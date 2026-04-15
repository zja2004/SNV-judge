"""
SNV Pathogenicity Predictor — v5
=================================
Ensemble meta-model (XGBoost + LightGBM Stacking, Isotonic-calibrated) integrating:
  Classical:    SIFT · PolyPhen-2 · AlphaMissense · CADD
  Foundation:   Evo2-40B (NVIDIA NIM) · Genos-10B (Stomics)
  Conservation: phyloP (Ensembl VEP)
  Population:   gnomAD v4 allele frequency

Trained on 2,000 ClinVar missense variants across 547 genes (5-fold CV).
AUROC = 0.9679 [0.959–0.974]  |  AUPRC = 0.9643 [0.953–0.972]
Holdout test (n=300): AUROC = 0.9547  |  F1 = 0.8946

New in v5:
  - Isotonic Regression calibration (Brier 0.0685 vs Platt 0.0743)
  - ACMG 5-tier classification badge
  - Variant query history with session-level comparison
  - Model reliability diagram (calibration curve)
  - AI report template selector (Chinese / English / Summary)

Run:
  export EVO2_API_KEY="nvapi-..."    # optional — enables Evo2 scoring
  export GENOS_API_KEY="sk-..."      # optional — enables Genos scoring
  export KIMI_API_KEY="sk-..."       # optional — enables AI clinical report (Kimi)
  streamlit run app.py
"""

import io
import os
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SNV Pathogenicity Predictor v5",
    page_icon="🧬",
    layout="wide",
)

# ── API keys (from environment variables) ─────────────────────────────────
EVO2_API_KEY  = os.environ.get("EVO2_API_KEY", "")
GENOS_API_KEY = os.environ.get("GENOS_API_KEY", "")
KIMI_API_KEY  = os.environ.get("KIMI_API_KEY", "")

# Inject KIMI_API_KEY into kimi_report module at import time
import kimi_report as _kimi_mod
if KIMI_API_KEY:
    _kimi_mod.KIMI_API_KEY = KIMI_API_KEY

# ── Genos-10B local embedding scoring (v5.2, optional) ────────────────────
# Import fetch_genos_embedding_score from the offline predict script.
# Used when GENOS_API_KEY is absent but a local ngrok endpoint is provided.
try:
    import sys as _sys
    _sys.path.insert(0, str(BASE / "skill"))
    from scripts.predict import fetch_genos_embedding_score as _fetch_genos_emb
    _GENOS_EMB_AVAILABLE = True
except Exception:
    _GENOS_EMB_AVAILABLE = False
EVO2_URL    = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/generate"
GENOS_URL   = "https://cloud.stomics.tech/api/aigateway/genos/variant_predict"
GNOMAD_URL  = "https://gnomad.broadinstitute.org/api"
ENSEMBL_SEQ_URL = "https://rest.ensembl.org/sequence/region/human"

# ── Feature definitions ───────────────────────────────────────────────────
FEATURE_COLS   = ["sift_score_inv", "polyphen_score", "am_pathogenicity",
                  "cadd_phred", "evo2_llr", "genos_path", "phylop", "gnomad_log_af"]
FEATURE_LABELS = ["SIFT (inv)", "PolyPhen-2", "AlphaMissense",
                  "CADD Phred", "Evo2 LLR", "Genos Score", "phyloP", "gnomAD log-AF"]
FEATURE_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
                  "#56B4E9", "#D55E00", "#75A025", "#E9ED4C"]

# ── Load model artefacts ──────────────────────────────────────────────────
BASE = Path(__file__).parent

@st.cache_resource
def load_artefacts():
    """Load model artefacts: v5 → v4 → v3 → v2 → v1 fallback chain.
    v5 uses Isotonic Regression calibration (better Brier score than Platt).
    """
    for suffix, ver in [("_v5", "v5"), ("_v4", "v4"), ("_v3", "v3"), ("_v2", "v2"), ("", "v1")]:
        try:
            with open(BASE / f"xgb_model{suffix}.pkl", "rb") as f:
                model = pickle.load(f)
            with open(BASE / f"train_medians{suffix}.pkl", "rb") as f:
                medians = pickle.load(f)
            with open(BASE / f"platt_scaler{suffix}.pkl", "rb") as f:
                calibrator = pickle.load(f)
            return model, medians, calibrator, ver
        except FileNotFoundError:
            continue
    raise RuntimeError("No model artefacts found. Run train.py first.")

MODEL, MEDIANS, PLATT, MODEL_VER = load_artefacts()

# ── ACMG 5-tier classification ────────────────────────────────────────────
# Thresholds calibrated on ClinVar gold-standard variants
ACMG_TIERS = [
    # (min_prob, label, short, color, text_color, confidence)
    (0.90, "Pathogenic",        "P",  "#c0392b", "#ffffff", "High confidence"),
    (0.70, "Likely Pathogenic", "LP", "#e74c3c", "#ffffff", "Moderate confidence"),
    (0.40, "VUS",               "VUS","#e67e22", "#ffffff", "Uncertain significance"),
    (0.20, "Likely Benign",     "LB", "#27ae60", "#ffffff", "Moderate confidence"),
    (0.00, "Benign",            "B",  "#1a7a4a", "#ffffff", "High confidence"),
]

def get_acmg_tier(prob: float) -> dict:
    """Return ACMG tier dict for a given calibrated probability."""
    for min_p, label, short, color, text_color, confidence in ACMG_TIERS:
        if prob >= min_p:
            return {"label": label, "short": short, "color": color,
                    "text_color": text_color, "confidence": confidence,
                    "prob": prob}
    return get_acmg_tier(0.0)

def render_acmg_badge(prob: float) -> str:
    """Return HTML for an ACMG classification badge."""
    t = get_acmg_tier(prob)
    return (
        f'<span style="background-color:{t["color"]};color:{t["text_color"]};'
        f'padding:4px 12px;border-radius:12px;font-weight:bold;font-size:14px;'
        f'letter-spacing:0.5px;">'
        f'{t["label"]} ({t["short"]})</span>'
        f'&nbsp;<span style="color:#666;font-size:12px;">{t["confidence"]} · '
        f'{prob:.1%}</span>'
    )

# ── Variant history (session-level) ──────────────────────────────────────
if "variant_history" not in st.session_state:
    st.session_state["variant_history"] = []   # list of result dicts

def _add_to_history(chrom, pos, ref, alt, scores, cal_prob, shap_vals,
                    evo2_llr, genos_path, gnomad_log_af):
    """Append a query result to session history (max 20 entries)."""
    import datetime
    entry = {
        "time":     datetime.datetime.now().strftime("%H:%M:%S"),
        "variant":  f"chr{chrom}:{pos} {ref}>{alt}",
        "gene":     scores.get("gene", "—"),
        "hgvsp":    scores.get("hgvsp", "—"),
        "prob":     round(cal_prob, 4),
        "acmg":     get_acmg_tier(cal_prob)["label"],
        "shap":     list(shap_vals),
        "scores":   {k: scores.get(k) for k in
                     ["sift_score", "polyphen_score", "am_pathogenicity",
                      "cadd_phred", "phylop"]},
        "evo2_llr":      evo2_llr,
        "genos_path":    genos_path,
        "gnomad_log_af": gnomad_log_af,
    }
    history = st.session_state["variant_history"]
    # Avoid exact duplicates
    if not history or history[-1]["variant"] != entry["variant"]:
        history.append(entry)
        if len(history) > 20:
            history.pop(0)
    st.session_state["variant_history"] = history

# ── VEP API ───────────────────────────────────────────────────────────────
VEP_URL = "https://rest.ensembl.org/vep/homo_sapiens/region"
VEP_HDR = {"Content-Type": "application/json", "Accept": "application/json"}

def fetch_vep_scores(chrom: str, pos: int, ref: str, alt: str) -> dict:
    """Call Ensembl VEP REST API and extract scores for a single variant."""
    variant_str = f"{chrom} {pos} . {ref} {alt} . . ."
    payload = {
        "variants":      [variant_str],
        "AlphaMissense": 1,
        "CADD":          1,
        "Conservation":  1,
        "canonical":     1,
        "mane":          1,
    }
    for attempt in range(3):
        try:
            r = requests.post(VEP_URL, headers=VEP_HDR, json=payload, timeout=30)
            if r.status_code == 200:
                break
            elif r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 5)))
            else:
                return {"error": f"VEP HTTP {r.status_code}: {r.text[:200]}"}
        except Exception as e:
            return {"error": str(e)}
    else:
        return {"error": "VEP API unavailable after 3 attempts"}

    data = r.json()
    if not data:
        return {"error": "No VEP results returned"}

    result = data[0]
    tcs = result.get("transcript_consequences", [])
    chosen = next((tc for tc in tcs if tc.get("mane_select")), None)
    if chosen is None:
        chosen = next((tc for tc in tcs if tc.get("canonical") == 1), None)
    if chosen is None and tcs:
        chosen = tcs[0]
    if chosen is None:
        return {"error": "No transcript consequences found"}

    csq = chosen.get("consequence_terms", [])
    am  = chosen.get("alphamissense", {})
    return {
        "consequence":    ", ".join(csq),
        "gene":           chosen.get("gene_symbol", ""),
        "transcript":     chosen.get("transcript_id", ""),
        "hgvsp":          chosen.get("hgvsp", ""),
        "sift_score":     chosen.get("sift_score"),
        "sift_pred":      chosen.get("sift_prediction"),
        "polyphen_score": chosen.get("polyphen_score"),
        "polyphen_pred":  chosen.get("polyphen_prediction"),
        "cadd_phred":     chosen.get("cadd_phred"),
        "am_pathogenicity": am.get("am_pathogenicity") if isinstance(am, dict) else None,
        "am_class":         am.get("am_class")         if isinstance(am, dict) else None,
        "phylop":           chosen.get("conservation"),
        "warning": ("Variant consequence is not missense_variant. "
                    "Scores may be absent or unreliable.")
                   if "missense_variant" not in csq else None,
    }

def fetch_gnomad_af(chrom: str, pos: int, ref: str, alt: str) -> float:
    """Fetch gnomAD v4 allele frequency for a variant. Returns log10(AF+1e-8)."""
    query = """
    query V($vid: String!) {
      variant(variantId: $vid, dataset: gnomad_r4) {
        exome  { af }
        genome { af }
      }
    }
    """
    vid = f"{chrom}-{pos}-{ref}-{alt}"
    for attempt in range(3):
        try:
            r = requests.post(
                GNOMAD_URL,
                json={"query": query, "variables": {"vid": vid}},
                headers={"Content-Type": "application/json"},
                timeout=15,
            )
            if r.status_code == 200:
                d = r.json().get("data", {}).get("variant")
                if d:
                    vals = []
                    if d.get("exome")  and d["exome"].get("af")  is not None:
                        vals.append(d["exome"]["af"])
                    if d.get("genome") and d["genome"].get("af") is not None:
                        vals.append(d["genome"]["af"])
                    af = max(vals) if vals else 0.0
                else:
                    af = 0.0  # variant absent from gnomAD (PM2 signal)
                return float(np.log10(af + 1e-8))
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return np.nan  # API unavailable


def fetch_vep_batch(variants: list[dict]) -> list[dict]:
    """Batch VEP call for up to 200 variants at once."""
    vep_strings = [f"{v['chrom']} {v['pos']} . {v['ref']} {v['alt']} . . ."
                   for v in variants]
    payload = {"variants": vep_strings, "AlphaMissense": 1,
               "CADD": 1, "canonical": 1, "mane": 1}
    for attempt in range(3):
        try:
            r = requests.post(VEP_URL, headers=VEP_HDR, json=payload, timeout=60)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 10)))
        except Exception:
            time.sleep(5)
    return []

# ── Genomic sequence fetch (for Evo2) ─────────────────────────────────────
def fetch_sequence_context(chrom: str, pos: int, flank: int = 50) -> str | None:
    """Fetch (2*flank+1) bp genomic context centred on pos via Ensembl."""
    start, end = pos - flank, pos + flank
    url = f"{ENSEMBL_SEQ_URL}/{chrom}:{start}..{end}:1"
    hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
    try:
        r = requests.get(url, headers=hdrs, timeout=15)
        if r.status_code == 200:
            return r.json().get("seq", "")
    except Exception:
        pass
    return None

# ── Evo2 scoring ──────────────────────────────────────────────────────────
def evo2_score_variant(ref_context: str, alt_context: str) -> float:
    """Zero-shot LLR via Evo2-40B NVIDIA NIM API."""
    if not EVO2_API_KEY or not ref_context or not alt_context:
        return np.nan
    mid    = len(ref_context) // 2
    prefix = ref_context[:mid]
    suffix = ref_context[mid + 1: mid + 6]
    hdrs   = {"Authorization": f"Bearer {EVO2_API_KEY}",
              "Content-Type": "application/json"}
    log_probs = {}
    for base, label in [(ref_context[mid], "ref"), (alt_context[mid], "alt")]:
        payload = {"sequence": prefix + base + suffix, "num_tokens": 1,
                   "top_k": 4, "enable_sampled_probs": True, "temperature": 0.001}
        for attempt in range(3):
            try:
                r = requests.post(EVO2_URL, headers=hdrs, json=payload, timeout=30)
                if r.status_code == 200:
                    p = r.json().get("sampled_probs", [None])[0]
                    log_probs[label] = np.log(p) if p and p > 0 else np.nan
                    break
                elif r.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                else:
                    time.sleep(1)
            except Exception:
                time.sleep(2)
        time.sleep(0.05)
    return log_probs.get("alt", np.nan) - log_probs.get("ref", np.nan)

# ── Genos scoring ─────────────────────────────────────────────────────────
def genos_score_variant(chrom: str, pos: int, ref: str, alt: str) -> dict:
    """Pathogenicity score from Genos (Zhejiang Lab) Stomics API."""
    if not GENOS_API_KEY:
        return {"genos_path": np.nan}
    hdrs    = {"Authorization": f"Bearer {GENOS_API_KEY}",
               "Content-Type": "application/json"}
    payload = {"assembly": "hg38", "chrom": f"chr{chrom}",
               "pos": int(pos), "ref": ref, "alt": alt}
    for attempt in range(3):
        try:
            r = requests.post(GENOS_URL, headers=hdrs, json=payload, timeout=30)
            if r.status_code == 200:
                res = r.json().get("result", {})
                return {"genos_path": res.get("score_Pathogenic", np.nan)}
            elif r.status_code == 429:
                time.sleep(5 * (attempt + 1))
            else:
                time.sleep(1)
        except Exception:
            time.sleep(2)
    return {"genos_path": np.nan}

def fetch_ai_scores(chrom: str, pos: int, ref: str, alt: str) -> tuple[float, float, float]:
    """Fetch Evo2 LLR, Genos score, and gnomAD log-AF in parallel.
    Returns (evo2_llr, genos_path, gnomad_log_af).

    Genos scoring priority (v5.2):
      1. Stomics Cloud API  — if GENOS_API_KEY is set
      2. Local embedding    — if genos_embed_url is set in session_state (ngrok)
      3. Training median    — fallback (imputed in predict())
    """
    evo2_llr      = np.nan
    genos_path    = np.nan
    gnomad_log_af = np.nan

    # Determine Genos scoring mode
    genos_embed_url = st.session_state.get("genos_embed_url", "").strip()
    use_embedding   = (not GENOS_API_KEY and genos_embed_url
                       and _GENOS_EMB_AVAILABLE)

    with ThreadPoolExecutor(max_workers=3) as ex:
        fut_gnomad = ex.submit(fetch_gnomad_af, chrom, pos, ref, alt)
        fut_seq    = ex.submit(fetch_sequence_context, chrom, pos, 50)

        if use_embedding:
            # Local Genos-10B embedding: cosine distance REF↔ALT (500bp context)
            fut_genos = ex.submit(
                _fetch_genos_emb, chrom, pos, ref, alt, genos_embed_url
            )
        else:
            fut_genos = ex.submit(genos_score_variant, chrom, pos, ref, alt)

        gnomad_log_af = fut_gnomad.result()

        if use_embedding:
            genos_path = fut_genos.result()   # float (cosine dist) or np.nan
        else:
            genos_result = fut_genos.result()
            genos_path   = genos_result.get("genos_path", np.nan)

        seq = fut_seq.result()
        if seq and len(seq) == 101 and EVO2_API_KEY:
            ref_ctx  = seq
            alt_ctx  = seq[:50] + alt + seq[51:]
            evo2_llr = evo2_score_variant(ref_ctx, alt_ctx)

    return evo2_llr, genos_path, gnomad_log_af

# ── Prediction ────────────────────────────────────────────────────────────
def predict(scores: dict, evo2_llr: float = np.nan,
            genos_path: float = np.nan,
            gnomad_log_af: float = np.nan) -> tuple[float, float, np.ndarray]:
    """Return (raw_prob, calibrated_prob, shap_values).

    Feature set adapts to loaded model version:
      v4: 8 features (SIFT, PolyPhen, AM, CADD, Evo2, Genos, phyloP, gnomAD log-AF)
      v3: 7 features (SIFT, PolyPhen, AM, CADD, Evo2, Genos, phyloP)
      v2: 6 features (SIFT, PolyPhen, AM, CADD, Evo2, Genos)
      v1: 4 features (SIFT, PolyPhen, AM, CADD)
    Missing values are imputed with training medians.
    """
    sift_inv = (1.0 - scores["sift_score"]) if scores.get("sift_score") is not None else None

    def _get(key):
        v = scores.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
        med = MEDIANS
        if hasattr(med, '__getitem__'):
            try:
                return float(med[key])
            except (KeyError, TypeError):
                pass
        return 0.0

    def _ai(val, key):
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            return float(val)
        med = MEDIANS
        if hasattr(med, '__getitem__'):
            try:
                return float(med[key])
            except (KeyError, TypeError):
                pass
        return 0.0

    base_feats = [
        sift_inv if sift_inv is not None else _get("sift_score_inv"),
        _get("polyphen_score"),
        _get("am_pathogenicity"),
        _get("cadd_phred"),
    ]

    if MODEL_VER in ("v2", "v3", "v4"):
        feature_vec = base_feats + [
            _ai(evo2_llr,   "evo2_llr"),
            _ai(genos_path, "genos_path"),
        ]
        if MODEL_VER in ("v3", "v4"):
            feature_vec.append(_get("phylop"))
        if MODEL_VER == "v4":
            feature_vec.append(_ai(gnomad_log_af, "gnomad_log_af"))
    else:
        feature_vec = base_feats

    X = np.array([feature_vec])

    # v4 model is a bundle dict {xgb, lgb, meta}
    if MODEL_VER == "v4" and isinstance(MODEL, dict):
        xgb_p = MODEL["xgb"].predict_proba(X)[0, 1]
        lgb_p = MODEL["lgb"].predict_proba(X)[0, 1]
        meta_X = np.array([[xgb_p, lgb_p]])
        raw_prob = float(MODEL["meta"].predict_proba(meta_X)[0, 1])
    else:
        raw_prob = float(MODEL.predict_proba(X)[0, 1])

    # Calibration: Isotonic (v5) uses predict() directly; Platt uses predict_proba on logit
    from sklearn.isotonic import IsotonicRegression
    if isinstance(PLATT, IsotonicRegression):
        cal_prob = float(np.clip(PLATT.predict([raw_prob])[0], 0.0, 1.0))
    else:
        logit_raw = np.log(raw_prob / (1 - raw_prob + 1e-9))
        cal_prob  = float(PLATT.predict_proba([[logit_raw]])[0, 1])

    # SHAP via TreeExplainer on XGB base model
    try:
        import shap
        xgb_model = MODEL["xgb"] if isinstance(MODEL, dict) else MODEL
        explainer = shap.TreeExplainer(xgb_model)
        shap_vals = explainer.shap_values(X)[0]
    except Exception:
        shap_vals = np.zeros(len(feature_vec))

    return raw_prob, cal_prob, shap_vals

# ── Figures ───────────────────────────────────────────────────────────────
def make_gauge(prob: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"aspect": "equal"})
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.15, 1.2); ax.axis("off")
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#e0e0e0", lw=18, solid_capstyle="round")
    theta_fill = np.linspace(np.pi, np.pi - prob * np.pi, 200)
    color = "#009E73" if prob < 0.4 else ("#E69F00" if prob < 0.7 else "#D55E00")
    ax.plot(np.cos(theta_fill), np.sin(theta_fill),
            color=color, lw=18, solid_capstyle="round")
    angle = np.pi - prob * np.pi
    ax.annotate("", xy=(0.72 * np.cos(angle), 0.72 * np.sin(angle)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2, mutation_scale=14))
    ax.plot(0, 0, "ko", ms=6)
    ax.text(-1.05, -0.08, "Benign",     ha="center", fontsize=8, color="#009E73")
    ax.text( 1.05, -0.08, "Pathogenic", ha="center", fontsize=8, color="#D55E00")
    ax.text(0, 0.45, f"{prob:.1%}", ha="center", va="center",
            fontsize=20, fontweight="bold", color=color)
    label = ("Likely Benign" if prob < 0.4 else
             "Uncertain"     if prob < 0.7 else "Likely Pathogenic")
    ax.text(0, 0.18, label, ha="center", fontsize=10, color=color, fontweight="bold")
    fig.tight_layout(pad=0)
    return fig

def make_score_bars(scores: dict, evo2_llr: float, genos_path: float,
                    gnomad_log_af: float = np.nan,
                    genos_label: str = "Genos Score") -> plt.Figure:
    # Dynamic Genos label: show source in parentheses
    feat_labels = list(FEATURE_LABELS) if MODEL_VER in ("v2", "v3", "v4") else list(FEATURE_LABELS[:4])
    if MODEL_VER in ("v2", "v3", "v4") and len(feat_labels) > 5:
        feat_labels[5] = genos_label
    sift_inv = (1.0 - scores["sift_score"]) if scores.get("sift_score") is not None else None
    vals  = [sift_inv, scores.get("polyphen_score"),
             scores.get("am_pathogenicity"), scores.get("cadd_phred")]
    maxes = [1, 1, 1, 60]
    if MODEL_VER in ("v2", "v3", "v4"):
        vals  += [evo2_llr  if not np.isnan(evo2_llr)   else None,
                  genos_path if not np.isnan(genos_path) else None]
        maxes += [3, 1]   # Evo2 LLR range ~[-3,3]; Genos [0,1]
    if MODEL_VER in ("v3", "v4"):
        phylop = scores.get("phylop")
        vals  += [float(phylop) if phylop is not None else None]
        maxes += [6]      # phyloP range ~[-6, 6]
    if MODEL_VER == "v4":
        gnomad_val = scores.get("gnomad_log_af")
        vals  += [float(gnomad_val) if gnomad_val is not None else None]
        maxes += [8]      # log10(AF+1e-8) range ~[-8, 0]

    n = len(feat_labels)
    fig, axes = plt.subplots(n, 1, figsize=(5, 0.85 * n + 0.5))
    if n == 1:
        axes = [axes]
    for ax, label, val, mx, col in zip(axes, feat_labels, vals, maxes, FEATURE_COLORS):
        ax.set_xlim(-mx if label == "Evo2 LLR" else 0, mx)
        ax.set_yticks([]); ax.set_xticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#cccccc")
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            ax.barh(0, val, color=col, height=0.6, alpha=0.85)
            ax.text(val + mx * 0.02, 0, f"{val:.3f}", va="center", fontsize=9)
        else:
            ax.text(mx * 0.02, 0, "N/A (imputed)", va="center",
                    fontsize=9, color="#999999", style="italic")
        ax.set_ylabel(label, rotation=0, ha="right", va="center",
                      fontsize=9, labelpad=5)
        if label in ("SIFT (inv)", "PolyPhen-2", "AlphaMissense", "Genos Score"):
            ax.axvline(0.5, color="#888888", lw=0.8, ls="--", alpha=0.6)
        elif label == "CADD Phred":
            ax.axvline(20, color="#888888", lw=0.8, ls="--", alpha=0.6)
        elif label == "Evo2 LLR":
            ax.axvline(0, color="#888888", lw=0.8, ls="--", alpha=0.6)
    fig.suptitle("Sub-model Scores", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout(pad=0.4)
    return fig

def make_shap_bars(shap_vals: np.ndarray) -> plt.Figure:
    labels = FEATURE_LABELS[:len(shap_vals)]
    order  = np.argsort(np.abs(shap_vals))
    labels_sorted = [labels[i] for i in order]
    vals_sorted   = [shap_vals[i] for i in order]
    colors = ["#D55E00" if v > 0 else "#0072B2" for v in vals_sorted]
    fig, ax = plt.subplots(figsize=(5, 0.6 * len(labels) + 1.2))
    bars = ax.barh(labels_sorted, vals_sorted, color=colors, height=0.55, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    for bar, val in zip(bars, vals_sorted):
        x  = val + (0.005 if val >= 0 else -0.005)
        ha = "left" if val >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=9)
    ax.set_xlabel("SHAP value (contribution to pathogenicity log-odds)", fontsize=9)
    ax.set_title("Feature Contributions (SHAP)", fontsize=11, fontweight="bold")
    red_patch  = mpatches.Patch(color="#D55E00", label="→ Pathogenic")
    blue_patch = mpatches.Patch(color="#0072B2", label="→ Benign")
    ax.legend(handles=[red_patch, blue_patch], fontsize=8, loc="lower right")
    plt.tight_layout()
    return fig

# ── VCF parser ────────────────────────────────────────────────────────────
def parse_vcf(file_bytes: bytes) -> list[dict]:
    """Parse a VCF file (plain or gzipped) and return list of variant dicts."""
    import gzip
    try:
        if file_bytes[:2] == b"\x1f\x8b":
            text = gzip.decompress(file_bytes).decode("utf-8", errors="replace")
        else:
            text = file_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        return []

    variants = []
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue
        chrom, pos, _, ref, alt = parts[0], parts[1], parts[2], parts[3], parts[4]
        chrom = chrom.replace("chr", "")
        # Only keep SNVs (single nucleotide)
        if len(ref) == 1 and len(alt) == 1 and ref != alt:
            try:
                variants.append({"chrom": chrom, "pos": int(pos),
                                  "ref": ref.upper(), "alt": alt.upper()})
            except ValueError:
                continue
    return variants

def score_vcf_variants(variants: list[dict],
                       progress_bar, status_text) -> list[dict]:
    """Score a list of variants: VEP → Evo2/Genos → predict."""
    results = []
    total   = len(variants)
    BATCH   = 50

    # Step 1: VEP in batches of 50
    vep_map = {}
    for i in range(0, total, BATCH):
        batch = variants[i: i + BATCH]
        status_text.text(f"VEP 注释中… {min(i+BATCH, total)}/{total}")
        progress_bar.progress(int(0.4 * min(i + BATCH, total) / total))
        raw_results = fetch_vep_batch(batch)
        for res in raw_results:
            parts = res.get("input", "").split()
            if len(parts) >= 5:
                key = (parts[0], int(parts[1]), parts[3], parts[4])
                vep_map[key] = res
        time.sleep(0.3)

    # Step 2: Extract VEP scores + AI scoring in parallel
    def process_one(v):
        key = (v["chrom"], v["pos"], v["ref"], v["alt"])
        vep_raw = vep_map.get(key, {})

        # Extract VEP fields
        tcs = vep_raw.get("transcript_consequences", [])
        chosen = next((tc for tc in tcs if tc.get("mane_select")), None)
        if chosen is None:
            chosen = next((tc for tc in tcs if tc.get("canonical") == 1), None)
        if chosen is None and tcs:
            chosen = tcs[0]

        scores = {"sift_score": None, "polyphen_score": None,
                  "am_pathogenicity": None, "cadd_phred": None}
        gene, transcript, hgvsp, consequence = "", "", "", ""
        if chosen:
            scores["sift_score"]     = chosen.get("sift_score")
            scores["polyphen_score"] = chosen.get("polyphen_score")
            scores["cadd_phred"]     = chosen.get("cadd_phred")
            am = chosen.get("alphamissense", {})
            scores["am_pathogenicity"] = am.get("am_pathogenicity") if isinstance(am, dict) else None
            gene        = chosen.get("gene_symbol", "")
            transcript  = chosen.get("transcript_id", "")
            hgvsp       = chosen.get("hgvsp", "")
            consequence = ", ".join(chosen.get("consequence_terms", []))

        # AI scores + gnomAD AF
        evo2_llr, genos_path, gnomad_log_af = np.nan, np.nan, np.nan
        if MODEL_VER in ("v2", "v3", "v4"):
            evo2_llr, genos_path, gnomad_log_af = fetch_ai_scores(
                v["chrom"], v["pos"], v["ref"], v["alt"])

        _, cal_prob, shap_vals = predict(scores, evo2_llr, genos_path, gnomad_log_af)

        return {
            "Chrom":       v["chrom"],
            "Pos":         v["pos"],
            "Ref":         v["ref"],
            "Alt":         v["alt"],
            "Gene":        gene,
            "Transcript":  transcript,
            "HGVSp":       hgvsp,
            "Consequence": consequence,
            "SIFT (inv)":  round(1 - scores["sift_score"], 4)
                           if scores["sift_score"] is not None else None,
            "PolyPhen-2":  scores["polyphen_score"],
            "AlphaMissense": scores["am_pathogenicity"],
            "CADD Phred":  scores["cadd_phred"],
            "Evo2 LLR":    None if np.isnan(evo2_llr)      else round(float(evo2_llr), 4),
            "Genos Score": None if np.isnan(genos_path)    else round(float(genos_path), 4),
            "gnomAD log-AF": None if np.isnan(gnomad_log_af) else round(float(gnomad_log_af), 4),
            "Pathogenicity Prob": round(cal_prob, 4),
            "Classification": ("Likely Pathogenic" if cal_prob >= 0.7
                               else "Uncertain"     if cal_prob >= 0.4
                               else "Likely Benign"),
        }

    # Use threads for AI scoring parallelism
    n_workers = 4 if (EVO2_API_KEY or GENOS_API_KEY) else 8
    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(process_one, v): v for v in variants}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                v = futs[fut]
                results.append({
                    "Chrom": v["chrom"], "Pos": v["pos"],
                    "Ref": v["ref"], "Alt": v["alt"],
                    "Gene": "", "Transcript": "", "HGVSp": "",
                    "Consequence": "error", "SIFT (inv)": None,
                    "PolyPhen-2": None, "AlphaMissense": None,
                    "CADD Phred": None, "Evo2 LLR": None, "Genos Score": None,
                    "Pathogenicity Prob": None, "Classification": "Error",
                })
            done += 1
            pct = 0.4 + 0.6 * done / total
            progress_bar.progress(int(pct * 100))
            status_text.text(f"评分中… {done}/{total}")

    # Sort by pathogenicity probability descending
    results.sort(key=lambda x: x.get("Pathogenicity Prob") or 0, reverse=True)
    return results

# ══════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════
_ver_labels = {"v4": "v4 (Evo2 + Genos + phyloP + gnomAD AF)",
               "v3": "v3 (Evo2 + Genos + phyloP)",
               "v2": "v2 (Evo2 + Genos)",
               "v1": "v1 (4-feature fallback)"}
model_badge = _ver_labels.get(MODEL_VER, MODEL_VER)
ai_status   = []
if EVO2_API_KEY:  ai_status.append("Evo2 ✓")
if GENOS_API_KEY: ai_status.append("Genos ✓")
if KIMI_API_KEY:  ai_status.append("Kimi ✓")
ai_badge = " · ".join(ai_status) if ai_status else "AI models disabled (no API keys)"

st.title("🧬 SNV Pathogenicity Predictor")
st.markdown(
    f"**Model**: {model_badge} &nbsp;|&nbsp; **AI**: {ai_badge}  \n"
    "Ensemble meta-model integrating **SIFT · PolyPhen-2 · AlphaMissense · CADD** "
    "+ **Evo2-40B** (NVIDIA NIM) + **Genos-10B** (Stomics) + **phyloP** (conservation) "
    "+ **gnomAD v4 AF** (population frequency).  \n"
    "Trained on 2,000 ClinVar missense variants across 547 genes · "
    "AUROC = 0.9679 · Holdout AUROC = 0.9547 · Isotonic calibration (Brier = 0.0685)"
)

# ── Model Info expander with reliability diagram ──────────────────────────
with st.expander("📊 Model Info & Calibration Curve", expanded=False):
    _cal_fig_path = BASE / "figures" / "fig_calibration_comparison.png"
    _val_fig_path = BASE / "figures" / "fig_validation_holdout.png"
    _info_col1, _info_col2 = st.columns(2)
    with _info_col1:
        st.markdown("**Performance Summary**")
        st.markdown("""
| Split | AUROC | AUPRC | F1 | Brier |
|---|---|---|---|---|
| 5-fold CV (n=2000) | 0.9679 | 0.9643 | 0.9022 | 0.0685 |
| Holdout test (n=300) | 0.9547 | 0.9527 | 0.8946 | 0.0789 |

**Calibration** (v5 vs v4):
| Method | Brier | ECE |
|---|---|---|
| Uncalibrated | 0.0807 | 0.0639 |
| Platt Scaling (v4) | 0.0743 | 0.0233 |
| **Isotonic Reg. (v5)** | **0.0685** | **~0** |

**ACMG Thresholds**:
- ≥ 0.90 → Pathogenic (P)
- 0.70–0.90 → Likely Pathogenic (LP)
- 0.40–0.70 → VUS
- 0.20–0.40 → Likely Benign (LB)
- < 0.20 → Benign (B)
        """)
    with _info_col2:
        if _cal_fig_path.exists():
            st.image(str(_cal_fig_path), caption="Reliability Diagram — Calibration Comparison",
                     use_container_width=True)
        else:
            st.info("Calibration figure not found. Run train.py to generate.")
    if _val_fig_path.exists():
        st.image(str(_val_fig_path), caption="Independent Holdout Validation (n=300)",
                 use_container_width=True)
    # LOGO-CV generalization figure
    _logo_fig_path = BASE / "figures" / "fig_logo_cv.png"
    if _logo_fig_path.exists():
        st.divider()
        st.markdown("**Generalization — Leave-One-Gene-Out CV (LOGO-CV)**")
        st.image(str(_logo_fig_path),
                 caption=(
                     "LOGO-CV across 16 disease gene families. Each gene's variants are held out "
                     "and a fresh stacking model is trained on all remaining genes. "
                     "Note: each gene has only 20–21 variants (10P/10B), so AUROC estimates "
                     "carry wide confidence intervals — especially for genes with AUROC = 1.00. "
                     "MYH7 (AUROC = 0.79) is a known weak point due to its gain-of-function "
                     "mechanism and atypical SIFT/PolyPhen score distribution."
                 ),
                 use_container_width=True)

st.divider()

# ── Tabs: Single variant | Batch VCF | History | AI Report ───────────────
tab_single, tab_batch, tab_history, tab_report = st.tabs([
    "🔬 Single Variant",
    "📂 Batch VCF",
    "📋 History",
    "🤖 AI Clinical Report",
])

# ════════════════════════════════════════════════════════════════════════
# TAB 1: Single variant
# ════════════════════════════════════════════════════════════════════════
with tab_single:
    with st.sidebar:
        st.header("Variant Input")
        st.markdown("Enter a **GRCh38** missense SNV:")
        chrom = st.text_input("Chromosome", value="17",
                              help="e.g. 17, X (no 'chr' prefix)")
        pos   = st.number_input("Position (1-based)", value=7674220, min_value=1)
        ref   = st.text_input("Reference allele", value="C").upper().strip()
        alt   = st.text_input("Alternate allele",  value="T").upper().strip()
        st.markdown("---")
        st.markdown("**Example variants**")
        examples = {
            "TP53 R175H (pathogenic)":   ("17", 7674220,  "C", "T"),
            "BRCA1 R1699W (pathogenic)": ("17", 43057062, "C", "T"),
            "BRCA2 N372H (benign)":      ("13", 32906729, "C", "A"),
        }
        for name, (ec, ep, er, ea) in examples.items():
            if st.button(name, use_container_width=True):
                chrom, pos, ref, alt = ec, ep, er, ea
                st.rerun()
        run = st.button("Predict", type="primary", use_container_width=True)

    if run:
        if not re.match(r"^[0-9XYM]+$", chrom.replace("chr", "")):
            st.error("Invalid chromosome. Use format: 17, X, Y, MT")
            st.stop()
        if not re.match(r"^[ACGT]$", ref) or not re.match(r"^[ACGT]$", alt):
            st.error("REF and ALT must each be a single nucleotide (A/C/G/T)")
            st.stop()
        if ref == alt:
            st.error("REF and ALT must be different")
            st.stop()

        chrom_clean = chrom.replace("chr", "")

        with st.spinner(f"Fetching VEP scores for {chrom_clean}:{pos} {ref}>{alt}…"):
            scores = fetch_vep_scores(chrom_clean, int(pos), ref, alt)

        if "error" in scores:
            st.error(f"VEP API error: {scores['error']}")
            st.stop()
        if scores.get("warning"):
            st.warning(scores["warning"])

        evo2_llr, genos_path, gnomad_log_af = np.nan, np.nan, np.nan
        if MODEL_VER in ("v2", "v3", "v4"):
            with st.spinner("Fetching Evo2 + Genos AI scores + gnomAD AF…"):
                evo2_llr, genos_path, gnomad_log_af = fetch_ai_scores(
                    chrom_clean, int(pos), ref, alt)

        raw_prob, cal_prob, shap_vals = predict(scores, evo2_llr, genos_path, gnomad_log_af)

        # ── Record to history ─────────────────────────────────────────────
        _add_to_history(chrom_clean, int(pos), ref, alt, scores, cal_prob,
                        shap_vals, evo2_llr, genos_path, gnomad_log_af)

        col1, col2, col3 = st.columns([1.2, 1.1, 1.1])

        with col1:
            st.subheader("Pathogenicity Score")
            # ACMG 5-tier badge
            st.markdown(render_acmg_badge(cal_prob), unsafe_allow_html=True)
            st.markdown("")
            st.pyplot(make_gauge(cal_prob), use_container_width=True)
            _cal_method = "Isotonic-calibrated" if MODEL_VER == "v5" else "Platt-calibrated"
            _ver_desc = {"v5": "v5 model (Isotonic cal.)",
                         "v4": "v4 model (Evo2 + Genos + phyloP + gnomAD AF)",
                         "v3": "v3 model (Evo2 + Genos + phyloP)",
                         "v2": "v2 model (Evo2 + Genos)",
                         "v1": "base 4-feature model"}
            st.caption(
                f"{_cal_method} probability (raw: {raw_prob:.1%}).  \n"
                f"Using {_ver_desc.get(MODEL_VER, MODEL_VER)}."
            )
            st.markdown("**Variant annotation**")
            for k, v in {
                "Gene":          scores.get("gene", "—"),
                "Transcript":    scores.get("transcript", "—"),
                "Consequence":   scores.get("consequence", "—"),
                "HGVSp":         scores.get("hgvsp") or "—",
                "AM class":      scores.get("am_class") or "—",
                "SIFT pred":     scores.get("sift_pred") or "—",
                "PolyPhen pred": scores.get("polyphen_pred") or "—",
            }.items():
                st.markdown(f"- **{k}**: {v}")

        with col2:
            st.subheader("Sub-model Scores")
            # Dynamic Genos label: show scoring source
            _embed_url_sv = st.session_state.get("genos_embed_url", "").strip()
            if GENOS_API_KEY:
                _genos_lbl = "Genos Score (API)"
            elif _embed_url_sv and _GENOS_EMB_AVAILABLE and not (isinstance(genos_path, float) and np.isnan(genos_path)):
                _genos_lbl = "Genos Score (embedding)"
            else:
                _genos_lbl = "Genos Score*"
            st.pyplot(make_score_bars(scores, evo2_llr, genos_path, gnomad_log_af,
                                      genos_label=_genos_lbl),
                      use_container_width=True)
            st.caption("Dashed line = pathogenicity threshold. N/A = score unavailable; "
                       "training median imputed for prediction. "
                       "\\* = Genos Score imputed with training median (0.676).")

        with col3:
            st.subheader("SHAP Feature Contributions")
            st.pyplot(make_shap_bars(shap_vals), use_container_width=True)
            st.caption("Red → pathogenic · Blue → benign · Length = magnitude")

        st.divider()

        # ── AI Clinical Report button (inline, inside single-variant results) ──
        st.subheader("🤖 AI 智能临床解读")
        if not KIMI_API_KEY:
            st.info(
                "设置 `KIMI_API_KEY` 环境变量以启用 Kimi AI 临床报告生成。  \n"
                "```bash\nexport KIMI_API_KEY='sk-...'\n```"
            )
        else:
            if st.button("✨ 生成 AI 临床解读报告", type="secondary", use_container_width=False):
                variant_info = {
                    "chrom": chrom_clean, "pos": int(pos), "ref": ref, "alt": alt,
                }
                # Store results in session state for the AI Report tab
                st.session_state["last_variant_info"]   = variant_info
                st.session_state["last_scores"]         = scores
                st.session_state["last_shap_vals"]      = shap_vals
                st.session_state["last_cal_prob"]       = cal_prob
                st.session_state["last_evo2_llr"]       = evo2_llr
                st.session_state["last_genos_path"]     = genos_path
                st.session_state["last_gnomad_log_af"]  = gnomad_log_af
                st.session_state["last_model_ver"]      = MODEL_VER
                st.session_state["trigger_report"]      = True

                with st.spinner("Kimi 正在生成临床解读报告…"):
                    report_placeholder = st.empty()
                    full_report = ""
                    try:
                        for chunk in _kimi_mod.generate_report_stream(
                            variant_info, scores, shap_vals, cal_prob,
                            evo2_llr, genos_path, gnomad_log_af, MODEL_VER,
                            template="chinese",
                        ):
                            full_report += chunk
                            report_placeholder.markdown(full_report + "▌")
                        report_placeholder.markdown(full_report)
                        st.session_state["last_report"] = full_report
                        st.success("报告生成完成！可在「🤖 AI Clinical Report」标签页查看完整报告。")
                    except Exception as e:
                        st.error(f"Kimi API 调用失败: {e}")

        st.divider()
        with st.expander("Raw scores table"):
            import pandas as pd
            feat_labels = FEATURE_LABELS[:len(shap_vals)]
            sift_inv = (1 - scores["sift_score"]) if scores.get("sift_score") is not None else None

            def _med(k, default=0.0):
                try: return float(MEDIANS[k])
                except Exception: return default

            raw_vals = [scores.get("sift_score"), scores.get("polyphen_score"),
                        scores.get("am_pathogenicity"), scores.get("cadd_phred")]
            used_vals = [
                sift_inv  if sift_inv is not None else _med("sift_score_inv", 0.5),
                scores.get("polyphen_score")    if scores.get("polyphen_score")    is not None else _med("polyphen_score"),
                scores.get("am_pathogenicity")  if scores.get("am_pathogenicity")  is not None else _med("am_pathogenicity"),
                scores.get("cadd_phred")        if scores.get("cadd_phred")        is not None else _med("cadd_phred"),
            ]
            if MODEL_VER in ("v2", "v3", "v4"):
                raw_vals  += [evo2_llr   if not np.isnan(evo2_llr)   else None,
                              genos_path if not np.isnan(genos_path)  else None]
                used_vals += [evo2_llr   if not np.isnan(evo2_llr)   else _med("evo2_llr"),
                              genos_path if not np.isnan(genos_path)  else _med("genos_path", 0.5)]
            if MODEL_VER in ("v3", "v4"):
                phylop_raw = scores.get("phylop")
                raw_vals  += [float(phylop_raw) if phylop_raw is not None else None]
                used_vals += [float(phylop_raw) if phylop_raw is not None else _med("phylop")]
            if MODEL_VER == "v4":
                raw_vals  += [gnomad_log_af if not np.isnan(gnomad_log_af) else None]
                used_vals += [gnomad_log_af if not np.isnan(gnomad_log_af) else _med("gnomad_log_af")]
            st.dataframe(pd.DataFrame({
                "Feature":       feat_labels,
                "Raw value":     raw_vals,
                "Used in model": [round(v, 4) if v is not None else None for v in used_vals],
                "SHAP value":    list(shap_vals),
                "Imputed?":      [v is None or (isinstance(v, float) and np.isnan(v))
                                  for v in raw_vals],
            }).round(4), use_container_width=True)
    else:
        st.info("Enter a variant in the sidebar and click **Predict** to get started.")
        st.markdown(f"""
### How it works
1. **Input**: GRCh38 chromosome, position, REF and ALT alleles
2. **VEP API**: Fetches SIFT, PolyPhen-2, AlphaMissense, CADD scores live
3. **AI models** *(if API keys set)*: Evo2-40B zero-shot LLR + Genos-10B pathogenicity score
4. **Stacking ensemble**: XGBoost + LightGBM meta-model combines all features
5. **SHAP**: Explains which features drove the prediction

### Model performance (v5, 5-fold CV + holdout, n=2,000)
| Model | AUROC | AUPRC | Calibration |
|---|---|---|---|
| **v5 (Isotonic cal.)** | **0.9679** | **0.9643** | **Brier=0.0685** |
| v4 (Platt cal.) | 0.9664 | 0.9671 | Brier=0.0743 |
| v3 (+ phyloP) | 0.9488 | 0.9447 | — |
| v2 (Evo2 + Genos) | 0.9373 | 0.9345 | — |
| AlphaMissense alone | 0.9109 | 0.9393 | — |

**Holdout test (n=300)**: AUROC=0.9547 · AUPRC=0.9527 · F1=0.8946

### API key setup
```bash
export EVO2_API_KEY="nvapi-..."    # https://build.nvidia.com/arc-institute/evo2
export GENOS_API_KEY="sk-..."      # https://cloud.stomics.tech
```
        """)

# ════════════════════════════════════════════════════════════════════════
# TAB 2: Batch VCF
# ════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("Batch VCF Scoring")
    st.markdown(
        "Upload a **VCF file** (GRCh38, `.vcf` or `.vcf.gz`) to score multiple variants at once.  \n"
        "Only **SNVs** (single nucleotide variants) are processed. "
        "Results are sorted by pathogenicity probability."
    )

    col_upload, col_info = st.columns([2, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "Upload VCF file", type=["vcf", "gz"],
            help="Standard VCF format. Multi-allelic sites are skipped."
        )
    with col_info:
        st.markdown("**Limits**")
        st.markdown("- Max **500 SNVs** per upload\n"
                    "- VEP batch: 50 variants/request\n"
                    "- AI scoring: parallel (4 workers)\n"
                    "- Estimated time: ~2 min / 100 variants")

    if uploaded is not None:
        file_bytes = uploaded.read()
        variants   = parse_vcf(file_bytes)

        if not variants:
            st.error("No valid SNVs found in the uploaded VCF file.")
        else:
            MAX_VARIANTS = 500
            if len(variants) > MAX_VARIANTS:
                st.warning(f"Found {len(variants):,} SNVs — processing first {MAX_VARIANTS}.")
                variants = variants[:MAX_VARIANTS]
            else:
                st.success(f"Found **{len(variants):,} SNVs** in VCF. Ready to score.")

            if st.button("🚀 Score All Variants", type="primary"):
                progress_bar = st.progress(0)
                status_text  = st.empty()

                results = score_vcf_variants(variants, progress_bar, status_text)

                progress_bar.progress(100)
                status_text.text(f"完成！共评分 {len(results)} 个变异体。")

                import pandas as pd
                df_results = pd.DataFrame(results)

                # Summary stats
                n_path = (df_results["Classification"] == "Likely Pathogenic").sum()
                n_unc  = (df_results["Classification"] == "Uncertain").sum()
                n_ben  = (df_results["Classification"] == "Likely Benign").sum()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total SNVs",          len(results))
                m2.metric("Likely Pathogenic",   n_path, delta=None)
                m3.metric("Uncertain",           n_unc)
                m4.metric("Likely Benign",       n_ben)

                st.divider()

                # Color-coded table
                def color_class(val):
                    if val == "Likely Pathogenic":
                        return "background-color: #fde8e8; color: #c0392b"
                    elif val == "Uncertain":
                        return "background-color: #fef9e7; color: #d68910"
                    elif val == "Likely Benign":
                        return "background-color: #e8f8f5; color: #1e8449"
                    return ""

                display_cols = ["Chrom", "Pos", "Ref", "Alt", "Gene", "HGVSp",
                                "AlphaMissense", "CADD Phred", "Evo2 LLR",
                                "Genos Score", "gnomAD log-AF",
                                "Pathogenicity Prob", "Classification"]
                if MODEL_VER not in ("v2", "v3", "v4"):
                    display_cols = [c for c in display_cols
                                    if c not in ("Evo2 LLR", "Genos Score", "gnomAD log-AF")]
                elif MODEL_VER != "v4":
                    display_cols = [c for c in display_cols if c != "gnomAD log-AF"]

                st.dataframe(
                    df_results[display_cols].style.applymap(
                        color_class, subset=["Classification"]
                    ).format({
                        "Pathogenicity Prob": "{:.3f}",
                        "AlphaMissense":      lambda x: f"{x:.3f}" if x is not None else "N/A",
                        "CADD Phred":         lambda x: f"{x:.1f}" if x is not None else "N/A",
                        "Evo2 LLR":           lambda x: f"{x:.3f}" if x is not None else "N/A",
                        "Genos Score":        lambda x: f"{x:.3f}" if x is not None else "N/A",
                    }),
                    use_container_width=True,
                    height=500,
                )

                # CSV download
                csv_buf = io.StringIO()
                df_results.to_csv(csv_buf, index=False)
                st.download_button(
                    label="⬇️ Download results as CSV",
                    data=csv_buf.getvalue(),
                    file_name=f"snv_judge_results_{uploaded.name.replace('.vcf','').replace('.gz','')}.csv",
                    mime="text/csv",
                )

                # Top pathogenic variants chart
                top_path = df_results[
                    df_results["Classification"] == "Likely Pathogenic"
                ].head(20)
                if not top_path.empty:
                    st.subheader(f"Top Likely Pathogenic Variants (n={len(top_path)})")
                    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(top_path))))
                    labels_plot = [
                        f"{r.Gene} {r.HGVSp.split(':')[-1] if r.HGVSp else f'{r.Chrom}:{r.Pos}'}"
                        for r in top_path.itertuples()
                    ]
                    probs = top_path["Pathogenicity Prob"].values
                    colors_plot = ["#D55E00" if p >= 0.7 else "#E69F00" for p in probs]
                    ax.barh(labels_plot[::-1], probs[::-1], color=colors_plot[::-1],
                            height=0.6, edgecolor="white")
                    ax.axvline(0.7, color="black", lw=0.8, ls="--", alpha=0.5)
                    ax.set_xlabel("Pathogenicity Probability", fontsize=10)
                    ax.set_xlim(0, 1.05)
                    ax.set_title("Top Likely Pathogenic Variants", fontsize=12, fontweight="bold")
                    import seaborn as sns
                    sns.despine(ax=ax)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)

    else:
        st.markdown("""
**Expected VCF format** (standard 4-column minimum):
```
##fileformat=VCFv4.2
#CHROM  POS     ID  REF  ALT  ...
17      7674220 .   C    T    ...
13      32906729 .  C    A    ...
```
        """)

# ════════════════════════════════════════════════════════════════════════
# TAB 3: Variant History
# ════════════════════════════════════════════════════════════════════════
with tab_history:
    st.subheader("📋 Variant Query History")
    history = st.session_state.get("variant_history", [])

    if not history:
        st.info("No variants queried yet in this session. "
                "Run a prediction in the **Single Variant** tab to start building history.")
    else:
        import pandas as pd

        # ── Summary table ─────────────────────────────────────────────
        hist_df = pd.DataFrame([{
            "Time":    h["time"],
            "Variant": h["variant"],
            "Gene":    h["gene"],
            "HGVSp":   h["hgvsp"],
            "Prob":    h["prob"],
            "ACMG":    h["acmg"],
        } for h in history])

        def _color_acmg(val):
            colors = {
                "Pathogenic":        "background-color:#fde8e8;color:#c0392b",
                "Likely Pathogenic": "background-color:#fef0f0;color:#e74c3c",
                "VUS":               "background-color:#fef9e7;color:#d68910",
                "Likely Benign":     "background-color:#eafaf1;color:#27ae60",
                "Benign":            "background-color:#d5f5e3;color:#1a7a4a",
            }
            return colors.get(val, "")

        st.dataframe(
            hist_df.style
                .applymap(_color_acmg, subset=["ACMG"])
                .format({"Prob": "{:.4f}"}),
            use_container_width=True,
            height=min(400, 60 + 35 * len(hist_df)),
        )

        # ── Export history ────────────────────────────────────────────
        csv_hist = hist_df.to_csv(index=False)
        st.download_button(
            "⬇️ Export history as CSV",
            data=csv_hist,
            file_name="snv_judge_history.csv",
            mime="text/csv",
        )

        st.divider()

        # ── Comparison view ───────────────────────────────────────────
        if len(history) >= 2:
            st.markdown("#### Compare Two Variants")
            variant_labels = [f"{h['time']} — {h['variant']} ({h['gene']})"
                              for h in history]
            col_a, col_b = st.columns(2)
            with col_a:
                sel_a = st.selectbox("Variant A", variant_labels,
                                     index=len(variant_labels)-2, key="hist_a")
            with col_b:
                sel_b = st.selectbox("Variant B", variant_labels,
                                     index=len(variant_labels)-1, key="hist_b")

            idx_a = variant_labels.index(sel_a)
            idx_b = variant_labels.index(sel_b)
            ha, hb = history[idx_a], history[idx_b]

            # Side-by-side comparison
            c1, c2 = st.columns(2)
            for col, h in [(c1, ha), (c2, hb)]:
                with col:
                    st.markdown(f"**{h['variant']}**")
                    st.markdown(render_acmg_badge(h["prob"]), unsafe_allow_html=True)
                    st.markdown("")
                    st.markdown(f"- Gene: **{h['gene']}**")
                    st.markdown(f"- HGVSp: `{h['hgvsp']}`")
                    st.markdown(f"- Probability: **{h['prob']:.4f}**")
                    st.markdown(f"- ACMG: **{h['acmg']}**")
                    # Mini score table
                    score_rows = []
                    for feat, key in [("SIFT (inv)", "sift_score"),
                                      ("PolyPhen-2", "polyphen_score"),
                                      ("AlphaMissense", "am_pathogenicity"),
                                      ("CADD Phred", "cadd_phred"),
                                      ("phyloP", "phylop")]:
                        v = h["scores"].get(key)
                        score_rows.append({"Feature": feat,
                                           "Value": f"{v:.3f}" if v is not None else "N/A"})
                    score_rows.append({"Feature": "Evo2 LLR",
                                       "Value": f"{h['evo2_llr']:.3f}"
                                                if h["evo2_llr"] is not None
                                                   and not (isinstance(h["evo2_llr"], float)
                                                            and np.isnan(h["evo2_llr"]))
                                                else "N/A"})
                    st.dataframe(pd.DataFrame(score_rows), use_container_width=True,
                                 hide_index=True)

            # SHAP comparison bar chart
            st.markdown("#### SHAP Contribution Comparison")
            feat_labels_hist = FEATURE_LABELS[:len(ha["shap"])]
            fig_cmp, axes_cmp = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
            for ax_c, h, title_c in [(axes_cmp[0], ha, sel_a.split("—")[1].strip()),
                                      (axes_cmp[1], hb, sel_b.split("—")[1].strip())]:
                shap_c = h["shap"][:len(feat_labels_hist)]
                order_c = np.argsort(np.abs(shap_c))
                labs_c  = [feat_labels_hist[i] for i in order_c]
                vals_c  = [shap_c[i] for i in order_c]
                cols_c  = ["#D55E00" if v > 0 else "#0072B2" for v in vals_c]
                ax_c.barh(labs_c, vals_c, color=cols_c, height=0.55, edgecolor="white")
                ax_c.axvline(0, color="black", lw=0.8)
                ax_c.set_title(title_c[:40], fontsize=10, fontweight="bold")
                ax_c.set_xlabel("SHAP value", fontsize=9)
                ax_c.grid(True, alpha=0.2, axis="x")
            plt.tight_layout()
            st.pyplot(fig_cmp, use_container_width=True)
            plt.close(fig_cmp)
        else:
            st.info("Query at least 2 variants to enable comparison.")

        st.divider()
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state["variant_history"] = []
            st.rerun()

# ════════════════════════════════════════════════════════════════════════
# TAB 4: AI Clinical Report
# ════════════════════════════════════════════════════════════════════════
with tab_report:
    st.subheader("🤖 AI 智能临床变异解读报告")
    st.markdown(
        "本模块由 **Kimi（Moonshot AI）** 驱动，将 SNV-judge v4 智能体系统的多源证据  \n"
        "（VEP · Evo2 · Genos · gnomAD · SHAP）转化为结构化 ACMG 风格临床解读报告。"
    )
    st.divider()

    # ── API key input (sidebar override) ─────────────────────────────────
    _cfg_col1, _cfg_col2 = st.columns(2)

    with _cfg_col1.expander("⚙️ Kimi API 设置", expanded=not bool(KIMI_API_KEY)):
        kimi_key_input = st.text_input(
            "Kimi API Key",
            value=KIMI_API_KEY,
            type="password",
            placeholder="sk-...",
            help="从 https://platform.moonshot.cn 获取 API Key",
        )
        if kimi_key_input and kimi_key_input != KIMI_API_KEY:
            _kimi_mod.KIMI_API_KEY = kimi_key_input
            st.success("API Key 已更新（本次会话有效）")

        if st.button("🔍 测试 Kimi API 连接"):
            ok, msg = _kimi_mod.check_kimi_available()
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    with _cfg_col2.expander("⚙️ Genos Embedding 设置 (v5.2)", expanded=False):
        st.markdown(
            "**本地 Genos-10B embedding 评分**  \n"
            "无需 Stomics Cloud API Key。通过本地 ngrok 端点计算 REF/ALT 序列余弦距离，"
            "替代 Genos Score 的训练集中位数填充。"
        )
        genos_url_input = st.text_input(
            "Genos 本地服务器 URL",
            value=st.session_state.get("genos_embed_url", ""),
            placeholder="https://xxx.ngrok-free.dev",
            help="启动本地 Genos-10B 服务后，将 ngrok URL 粘贴到此处。"
                 "服务不可用时自动降级回训练集中位数填充。",
            key="genos_url_input_field",
        )
        if genos_url_input != st.session_state.get("genos_embed_url", ""):
            st.session_state["genos_embed_url"] = genos_url_input.strip()
            if genos_url_input.strip():
                st.success(f"Genos embedding URL 已设置（本次会话有效）")
            else:
                st.info("Genos embedding URL 已清除，将使用训练集中位数填充。")

        # 显示当前 Genos 评分模式
        _embed_url = st.session_state.get("genos_embed_url", "").strip()
        if GENOS_API_KEY:
            st.caption("当前模式：**Stomics Cloud API**（GENOS_API_KEY 已设置，优先级最高）")
        elif _embed_url and _GENOS_EMB_AVAILABLE:
            st.caption(f"当前模式：**本地 Genos embedding**（余弦距离，500bp 上下文）  \n"
                       f"URL: `{_embed_url}`")
        else:
            st.caption("当前模式：**训练集中位数填充**（Genos Score* = 0.676）  \n"
                       "设置 URL 后可启用本地 embedding 评分。")

    st.divider()

    # ── Manual variant input for report generation ────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### 输入变异信息")
        st.markdown("*或在「🔬 Single Variant」标签页预测后点击「生成 AI 报告」按钮*")

        r_chrom = st.text_input("染色体", value="17", key="r_chrom")
        r_pos   = st.number_input("位置 (GRCh38)", value=7674220, min_value=1, key="r_pos")
        r_ref   = st.text_input("REF", value="C", key="r_ref").upper().strip()
        r_alt   = st.text_input("ALT", value="T", key="r_alt").upper().strip()

        st.markdown("**快速示例**")
        ex_cols = st.columns(3)
        if ex_cols[0].button("TP53 R175H", key="ex_tp53"):
            st.session_state.update({"r_chrom": "17", "r_pos": 7674220,
                                     "r_ref": "C", "r_alt": "T"})
            st.rerun()
        if ex_cols[1].button("BRCA1 R1699W", key="ex_brca1"):
            st.session_state.update({"r_chrom": "17", "r_pos": 43057062,
                                     "r_ref": "C", "r_alt": "T"})
            st.rerun()
        if ex_cols[2].button("BRCA2 N372H", key="ex_brca2"):
            st.session_state.update({"r_chrom": "13", "r_pos": 32906729,
                                     "r_ref": "C", "r_alt": "A"})
            st.rerun()

        generate_btn = st.button(
            "🚀 获取评分并生成 AI 报告",
            type="primary",
            use_container_width=True,
            key="gen_report_btn",
        )

    with col_right:
        st.markdown("#### 报告模板")
        report_template = st.radio(
            "选择报告语言/格式",
            options=["🇨🇳 中文临床报告", "🇬🇧 English Clinical Report", "📝 简版摘要"],
            index=0,
            key="report_template",
            help="中文报告适合国内临床遗传咨询；英文报告适合国际发表；简版摘要适合快速浏览。",
        )
        st.session_state["selected_template"] = report_template

        template_desc = {
            "🇨🇳 中文临床报告": "完整 ACMG 风格中文报告，含六节结构化内容",
            "🇬🇧 English Clinical Report": "Full ACMG-style report in English, 6-section structure",
            "📝 简版摘要": "3-5 句话的快速摘要，含分类结论和主要证据",
        }
        st.caption(template_desc.get(report_template, ""))
        st.markdown("---")
        st.markdown("""
**报告包含以下内容：**

1. **变异基本信息** — 坐标、基因、蛋白变化
2. **集成预测结果** — SNV-judge v5 校准概率 + ACMG 分类
3. **多维度证据分析**
   - 经典工具（SIFT/PP2/AM/CADD）→ PP3/BP4
   - 基因组基础模型（Evo2/Genos）
   - 进化保守性（phyloP）→ PP3/BP4
   - 人群频率（gnomAD）→ BA1/PM2
4. **SHAP 特征贡献** — 哪个工具最关键
5. **综合 ACMG 分类建议**
6. **临床意义与后续建议**

**模型**: `moonshot-v1-32k`  
**温度**: 0.3（低随机性，保证一致性）
        """)

    # ── Generate report ───────────────────────────────────────────────────
    if generate_btn or st.session_state.get("trigger_report"):
        st.session_state["trigger_report"] = False

        # Use session state data if available (from single-variant tab)
        if st.session_state.get("last_variant_info") and not generate_btn:
            v_info      = st.session_state["last_variant_info"]
            v_scores    = st.session_state["last_scores"]
            v_shap      = st.session_state["last_shap_vals"]
            v_prob      = st.session_state["last_cal_prob"]
            v_evo2      = st.session_state["last_evo2_llr"]
            v_genos     = st.session_state["last_genos_path"]
            v_gnomad    = st.session_state["last_gnomad_log_af"]
            v_model_ver = st.session_state["last_model_ver"]
        else:
            # Fresh fetch from manual input
            chrom_r = r_chrom.replace("chr", "")
            if not _kimi_mod.KIMI_API_KEY:
                st.error("请先设置 Kimi API Key（见上方设置面板）")
                st.stop()

            with st.spinner(f"正在获取 {chrom_r}:{r_pos} {r_ref}>{r_alt} 的评分…"):
                v_scores = fetch_vep_scores(chrom_r, int(r_pos), r_ref, r_alt)
                if "error" in v_scores:
                    st.error(f"VEP 错误: {v_scores['error']}")
                    st.stop()
                v_evo2, v_genos, v_gnomad = np.nan, np.nan, np.nan
                if MODEL_VER in ("v2", "v3", "v4"):
                    v_evo2, v_genos, v_gnomad = fetch_ai_scores(
                        chrom_r, int(r_pos), r_ref, r_alt)
                _, v_prob, v_shap = predict(v_scores, v_evo2, v_genos, v_gnomad)
                v_info      = {"chrom": chrom_r, "pos": int(r_pos),
                               "ref": r_ref, "alt": r_alt}
                v_model_ver = MODEL_VER

        # Check if we already have a cached report
        cached = st.session_state.get("last_report", "")
        if cached and not generate_btn:
            st.markdown(cached)
        else:
            if not _kimi_mod.KIMI_API_KEY:
                st.error("请先设置 Kimi API Key（见上方设置面板）")
                st.stop()

            st.markdown("---")
            st.markdown(f"**变异**: chr{v_info['chrom']}:{v_info['pos']} "
                        f"{v_info['ref']}>{v_info['alt']}  |  "
                        f"**基因**: {v_scores.get('gene', '?')}  |  "
                        f"**概率**: {v_prob:.1%}")
            st.markdown("---")

            report_area = st.empty()
            full_report = ""
            try:
                _tmpl = st.session_state.get("selected_template", "🇨🇳 中文临床报告")
                _tmpl_map = {
                    "🇨🇳 中文临床报告":          "chinese",
                    "🇬🇧 English Clinical Report": "english",
                    "📝 简版摘要":                 "summary",
                }
                with st.spinner("Kimi 正在分析多源证据，生成临床解读报告…"):
                    for chunk in _kimi_mod.generate_report_stream(
                        v_info, v_scores, v_shap, v_prob,
                        v_evo2, v_genos, v_gnomad, v_model_ver,
                        template=_tmpl_map.get(_tmpl, "chinese"),
                    ):
                        full_report += chunk
                        report_area.markdown(full_report + "▌")
                    report_area.markdown(full_report)
                    st.session_state["last_report"] = full_report

                # Download button
                st.download_button(
                    label="⬇️ 下载报告（Markdown）",
                    data=full_report,
                    file_name=f"snv_report_{v_info['chrom']}_{v_info['pos']}"
                              f"_{v_info['ref']}_{v_info['alt']}.md",
                    mime="text/markdown",
                )
            except Exception as e:
                st.error(f"Kimi API 调用失败: {str(e)}")
                st.info("请检查 API Key 是否正确，或稍后重试。")

    elif not st.session_state.get("last_report"):
        st.info(
            "👆 在左侧输入变异信息并点击「获取评分并生成 AI 报告」，  \n"
            "或在「🔬 Single Variant」标签页完成预测后点击「生成 AI 临床解读报告」按钮。"
        )
    else:
        # Show cached report from previous run
        st.markdown(st.session_state["last_report"])
        st.download_button(
            label="⬇️ 下载报告（Markdown）",
            data=st.session_state["last_report"],
            file_name="snv_report.md",
            mime="text/markdown",
        )

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data: ClinVar · Ensembl VEP · AlphaMissense (Google DeepMind) · CADD v1.7 · "
    "Evo2 (Arc Institute / NVIDIA) · Genos (Zhejiang Lab) · SIFT · PolyPhen-2 · gnomAD v4.  \n"
    "AI Report: Kimi (Moonshot AI, moonshot-v1-32k). "
    "Model trained on GRCh38 ClinVar gold-standard missense variants. "
    "**Not validated for clinical use.**"
)
