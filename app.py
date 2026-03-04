"""
SNV Pathogenicity Predictor
===========================
Ensemble meta-model (XGBoost) trained on BRCA1/BRCA2/TP53 ClinVar variants.
Features: SIFT, PolyPhen-2, AlphaMissense, CADD, REVEL — fetched live via
Ensembl VEP REST API.

Run:  streamlit run app.py
"""

import pickle
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SNV Pathogenicity Predictor",
    page_icon="🧬",
    layout="wide",
)

# ── Load model artefacts ──────────────────────────────────────────────────
BASE = Path(__file__).parent

@st.cache_resource
def load_artefacts():
    with open(BASE / "xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE / "train_medians.pkl", "rb") as f:
        medians = pickle.load(f)
    with open(BASE / "shap_explainer.pkl", "rb") as f:
        explainer = pickle.load(f)
    with open(BASE / "platt_scaler.pkl", "rb") as f:
        platt = pickle.load(f)
    return model, medians, explainer, platt

model, MEDIANS, EXPLAINER, PLATT = load_artefacts()

FEATURE_COLS   = ["sift_score_inv", "polyphen_score", "am_pathogenicity",
                  "cadd_phred", "revel"]
FEATURE_LABELS = ["SIFT (inv)", "PolyPhen-2", "AlphaMissense", "CADD Phred", "REVEL"]
SCORE_RANGES   = {
    "SIFT (inv)":    (0, 1,   "0 = tolerated → 1 = damaging"),
    "PolyPhen-2":    (0, 1,   "0 = benign → 1 = probably damaging"),
    "AlphaMissense": (0, 1,   "0 = benign → 1 = pathogenic"),
    "CADD Phred":    (0, 60,  "Phred-scaled; >20 = top 1%, >30 = top 0.1%"),
    "REVEL":         (0, 1,   "0 = benign → 1 = pathogenic"),
}

VEP_URL = "https://rest.ensembl.org/vep/homo_sapiens/region"
VEP_HDR = {"Content-Type": "application/json", "Accept": "application/json"}

# ── VEP fetch ─────────────────────────────────────────────────────────────
def fetch_vep_scores(chrom: str, pos: int, ref: str, alt: str) -> dict:
    """Call Ensembl VEP REST API and extract all 5 scores."""
    variant_str = f"{chrom} {pos} . {ref} {alt} . . ."
    payload = {
        "variants":      [variant_str],
        "AlphaMissense": 1,
        "CADD":          1,
        "REVEL":         1,
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
        return {"error": "No VEP results returned for this variant"}

    result = data[0]
    tcs = result.get("transcript_consequences", [])

    # Pick MANE Select > canonical > first
    chosen = None
    for tc in tcs:
        if tc.get("mane_select"):
            chosen = tc; break
    if chosen is None:
        for tc in tcs:
            if tc.get("canonical") == 1:
                chosen = tc; break
    if chosen is None and tcs:
        chosen = tcs[0]

    if chosen is None:
        return {"error": "No transcript consequences found"}

    # Check it's a missense variant
    csq = chosen.get("consequence_terms", [])
    if "missense_variant" not in csq:
        return {
            "warning": f"Variant consequence is '{', '.join(csq)}', not missense_variant. "
                       "Scores may be absent or unreliable.",
            "consequence": ", ".join(csq),
            "gene": chosen.get("gene_symbol", ""),
            "transcript": chosen.get("transcript_id", ""),
            "hgvsp": chosen.get("hgvsp", ""),
            "sift_score":       chosen.get("sift_score"),
            "sift_pred":        chosen.get("sift_prediction"),
            "polyphen_score":   chosen.get("polyphen_score"),
            "polyphen_pred":    chosen.get("polyphen_prediction"),
            "cadd_phred":       chosen.get("cadd_phred"),
            "cadd_raw":         chosen.get("cadd_raw"),
            "revel":            chosen.get("revel"),
            "am_pathogenicity": chosen.get("alphamissense", {}).get("am_pathogenicity")
                                if isinstance(chosen.get("alphamissense"), dict) else None,
            "am_class":         chosen.get("alphamissense", {}).get("am_class")
                                if isinstance(chosen.get("alphamissense"), dict) else None,
        }

    am = chosen.get("alphamissense", {})
    return {
        "consequence": ", ".join(csq),
        "gene":        chosen.get("gene_symbol", ""),
        "transcript":  chosen.get("transcript_id", ""),
        "hgvsp":       chosen.get("hgvsp", ""),
        "sift_score":       chosen.get("sift_score"),
        "sift_pred":        chosen.get("sift_prediction"),
        "polyphen_score":   chosen.get("polyphen_score"),
        "polyphen_pred":    chosen.get("polyphen_prediction"),
        "cadd_phred":       chosen.get("cadd_phred"),
        "cadd_raw":         chosen.get("cadd_raw"),
        "revel":            chosen.get("revel"),
        "am_pathogenicity": am.get("am_pathogenicity") if isinstance(am, dict) else None,
        "am_class":         am.get("am_class")         if isinstance(am, dict) else None,
    }

# ── Prediction ────────────────────────────────────────────────────────────
def predict(scores: dict) -> tuple[float, float, np.ndarray]:
    """Return (raw_prob, calibrated_prob, shap_values_array).

    Platt scaling (sigmoid calibration) corrects for the train/test
    prevalence mismatch: model trained at ~50% pathogenic prevalence,
    real-world BRCA2 prevalence ~5%.
    """
    sift_raw = scores.get("sift_score")
    sift_inv = (1.0 - sift_raw) if sift_raw is not None else None

    feature_vec = [
        sift_inv                        if sift_inv is not None  else MEDIANS["sift_score_inv"],
        scores.get("polyphen_score")    if scores.get("polyphen_score") is not None
                                        else MEDIANS["polyphen_score"],
        scores.get("am_pathogenicity")  if scores.get("am_pathogenicity") is not None
                                        else MEDIANS["am_pathogenicity"],
        scores.get("cadd_phred")        if scores.get("cadd_phred") is not None
                                        else MEDIANS["cadd_phred"],
        scores.get("revel")             if scores.get("revel") is not None
                                        else MEDIANS["revel"],
    ]
    X = np.array([feature_vec])
    raw_prob  = float(model.predict_proba(X)[0, 1])
    # Platt calibration: apply sigmoid scaler fitted on held-out BRCA2 test set
    logit_raw = np.log(raw_prob / (1 - raw_prob + 1e-9))
    cal_prob  = float(PLATT.predict_proba([[logit_raw]])[0, 1])
    shaps     = EXPLAINER.shap_values(X)[0]   # shape (5,)
    return raw_prob, cal_prob, shaps

# ── Gauge figure ──────────────────────────────────────────────────────────
def make_gauge(prob: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"aspect": "equal"})
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.15, 1.2)
    ax.axis("off")

    # Background arc (grey)
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#e0e0e0", lw=18, solid_capstyle="round")

    # Coloured fill up to prob
    theta_fill = np.linspace(np.pi, np.pi - prob * np.pi, 200)
    color = "#009E73" if prob < 0.4 else ("#E69F00" if prob < 0.7 else "#D55E00")
    ax.plot(np.cos(theta_fill), np.sin(theta_fill),
            color=color, lw=18, solid_capstyle="round")

    # Needle
    angle = np.pi - prob * np.pi
    ax.annotate("", xy=(0.72 * np.cos(angle), 0.72 * np.sin(angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="black",
                                lw=2, mutation_scale=14))
    ax.plot(0, 0, "ko", ms=6)

    # Labels
    ax.text(-1.05, -0.08, "Benign", ha="center", fontsize=8, color="#009E73")
    ax.text( 1.05, -0.08, "Pathogenic", ha="center", fontsize=8, color="#D55E00")
    ax.text(0, 0.45, f"{prob:.1%}", ha="center", va="center",
            fontsize=20, fontweight="bold", color=color)

    label = ("Likely Benign" if prob < 0.4
             else "Uncertain" if prob < 0.7
             else "Likely Pathogenic")
    ax.text(0, 0.18, label, ha="center", fontsize=10, color=color, fontweight="bold")

    fig.tight_layout(pad=0)
    return fig

# ── Score bar chart ───────────────────────────────────────────────────────
def make_score_bars(scores: dict) -> plt.Figure:
    vals = [
        (1.0 - scores["sift_score"]) if scores.get("sift_score") is not None else None,
        scores.get("polyphen_score"),
        scores.get("am_pathogenicity"),
        scores.get("cadd_phred"),
        scores.get("revel"),
    ]
    maxes = [1, 1, 1, 60, 1]
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]

    fig, axes = plt.subplots(5, 1, figsize=(5, 4.5))
    for i, (ax, label, val, mx, col) in enumerate(
            zip(axes, FEATURE_LABELS, vals, maxes, colors)):
        ax.set_xlim(0, mx)
        ax.set_yticks([]); ax.set_xticks([])
        ax.spines[["top","right","left"]].set_visible(False)
        ax.spines["bottom"].set_color("#cccccc")

        if val is not None:
            ax.barh(0, val, color=col, height=0.6, alpha=0.85)
            ax.text(val + mx * 0.02, 0, f"{val:.3f}", va="center", fontsize=9)
        else:
            ax.text(mx * 0.02, 0, "N/A (imputed)", va="center",
                    fontsize=9, color="#999999", style="italic")

        ax.set_ylabel(label, rotation=0, ha="right", va="center",
                      fontsize=9, labelpad=5)
        # Threshold line
        if label in ("SIFT (inv)", "PolyPhen-2", "AlphaMissense", "REVEL"):
            ax.axvline(0.5, color="#888888", lw=0.8, ls="--", alpha=0.6)
        elif label == "CADD Phred":
            ax.axvline(20, color="#888888", lw=0.8, ls="--", alpha=0.6)

    fig.suptitle("Sub-model Scores", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout(pad=0.4)
    return fig

# ── SHAP bar chart ────────────────────────────────────────────────────────
def make_shap_bars(shap_vals: np.ndarray) -> plt.Figure:
    order = np.argsort(np.abs(shap_vals))
    labels = [FEATURE_LABELS[i] for i in order]
    vals   = [shap_vals[i]       for i in order]
    colors = ["#D55E00" if v > 0 else "#0072B2" for v in vals]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.barh(labels, vals, color=colors, height=0.55, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    for bar, val in zip(bars, vals):
        x = val + (0.01 if val >= 0 else -0.01)
        ha = "left" if val >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=9)

    ax.set_xlabel("SHAP value (contribution to pathogenicity log-odds)", fontsize=9)
    ax.set_title("Feature Contributions (SHAP)", fontsize=11, fontweight="bold")
    red_patch  = mpatches.Patch(color="#D55E00", label="Pushes toward pathogenic")
    blue_patch = mpatches.Patch(color="#0072B2", label="Pushes toward benign")
    ax.legend(handles=[red_patch, blue_patch], fontsize=8, loc="lower right")
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════
st.title("🧬 SNV Pathogenicity Predictor")
st.markdown(
    "Ensemble meta-model (XGBoost + Platt calibration) integrating **SIFT · PolyPhen-2 · "
    "AlphaMissense · CADD · REVEL** scores fetched live from the Ensembl VEP API. "
    "Trained on 556 ClinVar gold-standard missense variants (BRCA1 + TP53); "
    "tested on held-out BRCA2 variants (XGBoost AUROC = 0.985 [95% CI 0.964–0.998])."
)

st.divider()

# ── Sidebar: input ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Variant Input")
    st.markdown("Enter a **GRCh38** missense SNV:")

    chrom = st.text_input("Chromosome", value="17",
                          help="e.g. 17, X (no 'chr' prefix)")
    pos   = st.number_input("Position (1-based)", value=7674220, min_value=1,
                            help="GRCh38 genomic position")
    ref   = st.text_input("Reference allele", value="C").upper().strip()
    alt   = st.text_input("Alternate allele",  value="T").upper().strip()

    st.markdown("---")
    st.markdown("**Example variants**")
    examples = {
        "TP53 R175H (pathogenic)":  ("17", 7674220,  "C", "T"),
        "BRCA1 R1699W (pathogenic)":("17", 43057062, "C", "T"),
        "BRCA2 N372H (benign)":     ("13", 32906729, "C", "A"),
    }
    for name, (ec, ep, er, ea) in examples.items():
        if st.button(name, use_container_width=True):
            chrom, pos, ref, alt = ec, ep, er, ea
            st.rerun()

    run = st.button("Predict", type="primary", use_container_width=True)

# ── Main panel ────────────────────────────────────────────────────────────
if run:
    # Validate inputs
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

    with st.spinner(f"Fetching VEP scores for {chrom_clean}:{pos} {ref}>{alt} ..."):
        scores = fetch_vep_scores(chrom_clean, int(pos), ref, alt)

    if "error" in scores:
        st.error(f"VEP API error: {scores['error']}")
        st.stop()

    if "warning" in scores:
        st.warning(scores["warning"])

    # Run prediction
    raw_prob, cal_prob, shap_vals = predict(scores)

    # ── Layout: 3 columns ─────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1.2, 1.1, 1.1])

    with col1:
        st.subheader("Pathogenicity Score")
        st.pyplot(make_gauge(cal_prob), use_container_width=True)
        st.caption(f"Platt-calibrated probability (raw XGBoost: {raw_prob:.1%}). "
                   "Calibration fitted on held-out BRCA2 test set to correct for "
                   "train/test prevalence mismatch.")

        # Variant annotation summary
        st.markdown("**Variant annotation**")
        ann_data = {
            "Gene":        scores.get("gene", "—"),
            "Transcript":  scores.get("transcript", "—"),
            "Consequence": scores.get("consequence", "—"),
            "HGVSp":       scores.get("hgvsp", "—") or "—",
            "AM class":    scores.get("am_class", "—") or "—",
            "SIFT pred":   scores.get("sift_pred", "—") or "—",
            "PolyPhen pred": scores.get("polyphen_pred", "—") or "—",
        }
        for k, v in ann_data.items():
            st.markdown(f"- **{k}**: {v}")

    with col2:
        st.subheader("Sub-model Scores")
        st.pyplot(make_score_bars(scores), use_container_width=True)
        st.caption("Dashed line = pathogenicity threshold (0.5 for probability scores; 20 for CADD Phred). "
                   "N/A = score unavailable; training median imputed for prediction.")

    with col3:
        st.subheader("SHAP Feature Contributions")
        st.pyplot(make_shap_bars(shap_vals), use_container_width=True)
        st.caption("Red bars push the prediction toward **pathogenic**; "
                   "blue bars push toward **benign**. "
                   "Bar length = magnitude of contribution.")

    st.divider()

    # ── Raw scores table ──────────────────────────────────────────────────
    with st.expander("Raw scores table"):
        import pandas as pd
        raw = {
            "Feature":       FEATURE_LABELS,
            "Raw value":     [
                scores.get("sift_score"),
                scores.get("polyphen_score"),
                scores.get("am_pathogenicity"),
                scores.get("cadd_phred"),
                scores.get("revel"),
            ],
            "Used in model": [
                (1 - scores["sift_score"]) if scores.get("sift_score") is not None
                    else MEDIANS["sift_score_inv"],
                scores.get("polyphen_score") if scores.get("polyphen_score") is not None
                    else MEDIANS["polyphen_score"],
                scores.get("am_pathogenicity") if scores.get("am_pathogenicity") is not None
                    else MEDIANS["am_pathogenicity"],
                scores.get("cadd_phred") if scores.get("cadd_phred") is not None
                    else MEDIANS["cadd_phred"],
                scores.get("revel") if scores.get("revel") is not None
                    else MEDIANS["revel"],
            ],
            "SHAP value":    list(shap_vals),
            "Imputed?":      [
                scores.get("sift_score") is None,
                scores.get("polyphen_score") is None,
                scores.get("am_pathogenicity") is None,
                scores.get("cadd_phred") is None,
                scores.get("revel") is None,
            ],
        }
        st.dataframe(pd.DataFrame(raw).round(4), use_container_width=True)

else:
    st.info("Enter a variant in the sidebar and click **Predict** to get started.")
    st.markdown("""
    ### How it works
    1. **Input**: GRCh38 chromosome, position, REF and ALT alleles
    2. **VEP API**: Fetches SIFT, PolyPhen-2, AlphaMissense, CADD, REVEL scores live
    3. **XGBoost**: Ensemble meta-model combines all 5 scores into a single pathogenicity probability
    4. **SHAP**: Explains which scores drove the prediction for this specific variant

    ### Model performance (BRCA2 held-out test set, n=286, 15 positives)
    | Model | AUROC [95% CI] | AUPRC [95% CI] |
    |---|---|---|
    | **XGBoost (calibrated)** | **0.985 [0.964–0.998]** | **0.847 [0.666–0.967]** |
    | Logistic Regression | 0.988 [0.971–1.000] | 0.889 [0.744–0.997] |
    | AlphaMissense alone | 0.985 [0.968–0.996] | 0.764 [0.528–0.940] |
    | CADD alone | 0.970 [0.933–0.993] | 0.750 [0.525–0.911] |

    > **Note:** CIs are wide due to only 15 positive cases in the test set. Logistic Regression
    > achieves slightly higher point-estimate AUPRC; the ensemble adds value through calibration
    > and SHAP interpretability. Bootstrap 95% CIs (n=2000 resamples).

    ### Limitations
    - Trained on BRCA1/TP53 variants only; cross-gene generalisation is not validated
    - Only 15 BRCA2 pathogenic variants in the test set — performance estimates have wide CIs
    - PolyPhen-2 coverage ~55% (requires structural data); missing values are median-imputed
    - Platt calibration fitted on the same test set used for evaluation (small n) — treat calibrated probabilities as indicative, not absolute
    - Not validated for clinical use
    """)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data sources: ClinVar · Ensembl VEP · AlphaMissense (Google DeepMind) · "
    "CADD v1.7 · REVEL · SIFT · PolyPhen-2. "
    "Model trained on GRCh38 ClinVar gold-standard missense variants."
)
