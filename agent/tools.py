"""
agent/tools.py — SNV-judge Plan C: LangChain Tool Definitions
==============================================================
Six tools that the Kimi-powered agent can call autonomously:

  1. vep_annotate          — Ensembl VEP: SIFT, PolyPhen-2, AlphaMissense, CADD, phyloP
  2. gnomad_frequency      — gnomAD v4 allele frequency + ACMG BA1/PM2 signal
  3. evo2_score            — Evo2-40B zero-shot LLR (NVIDIA NIM)
  4. genos_score           — Genos-10B pathogenicity score (Stomics)
  5. snv_judge_predict     — v5 ensemble prediction + SHAP + ACMG badge
  6. search_clinvar        — ClinVar/literature summary via Kimi web search
  7. save_patient_variant  — Persist variant result to patient profile (long-term memory)
  8. get_patient_history   — Retrieve patient's historical variants
  9. batch_vcf_analyze     — Parse VCF text, score all SNVs, return ranked summary
"""

import os
import json
import math
import time
import pickle
import numpy as np
import requests
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent          # repo root
PATIENT_DB = BASE / "agent" / "patient_profiles.json"

# ── API endpoints ──────────────────────────────────────────────────────────
VEP_URL     = "https://rest.ensembl.org/vep/homo_sapiens/region"
VEP_HDR     = {"Content-Type": "application/json", "Accept": "application/json"}
GNOMAD_URL  = "https://gnomad.broadinstitute.org/api"
EVO2_URL    = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/generate"
GENOS_URL   = "https://cloud.stomics.tech/api/aigateway/genos/variant_predict"
ENSEMBL_SEQ = "https://rest.ensembl.org/sequence/region/human"

# ── API keys (from environment, with fallback) ─────────────────────────────
EVO2_API_KEY  = os.environ.get("EVO2_API_KEY", "")
GENOS_API_KEY = os.environ.get("GENOS_API_KEY", "")

# ── Load v5 model artefacts ────────────────────────────────────────────────
def _load_model():
    for suffix, ver in [("_v5", "v5"), ("_v4", "v4"), ("_v3", "v3")]:
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
    raise RuntimeError("No model artefacts found.")

_MODEL, _MEDIANS, _CALIBRATOR, _MODEL_VER = _load_model()

ACMG_TIERS = [
    (0.90, "Pathogenic (P)",        "#c0392b"),
    (0.70, "Likely Pathogenic (LP)","#e74c3c"),
    (0.40, "VUS",                   "#e67e22"),
    (0.20, "Likely Benign (LB)",    "#27ae60"),
    (0.00, "Benign (B)",            "#1a7a4a"),
]

def _acmg_label(prob: float) -> str:
    for min_p, label, _ in ACMG_TIERS:
        if prob >= min_p:
            return label
    return "Benign (B)"

# ══════════════════════════════════════════════════════════════════════════
# TOOL 1: VEP Annotation
# ══════════════════════════════════════════════════════════════════════════
@tool
def vep_annotate(variant: str) -> str:
    """Annotate a missense SNV using Ensembl VEP REST API.
    Returns SIFT, PolyPhen-2, AlphaMissense, CADD Phred, phyloP conservation,
    gene symbol, transcript, HGVSp, and consequence.

    Args:
        variant: Variant in format "CHROM:POS:REF:ALT" e.g. "17:7674220:C:T"

    Returns:
        JSON string with annotation scores.
    """
    try:
        parts = variant.strip().replace("chr", "").split(":")
        if len(parts) != 4:
            return json.dumps({"error": "Invalid format. Use CHROM:POS:REF:ALT e.g. 17:7674220:C:T"})
        chrom, pos, ref, alt = parts[0], int(parts[1]), parts[2].upper(), parts[3].upper()
    except Exception as e:
        return json.dumps({"error": f"Parse error: {e}"})

    variant_str = f"{chrom} {pos} . {ref} {alt} . . ."
    payload = {
        "variants": [variant_str],
        "AlphaMissense": 1, "CADD": 1, "Conservation": 1,
        "canonical": 1, "mane": 1,
    }
    for attempt in range(3):
        try:
            r = requests.post(VEP_URL, headers=VEP_HDR, json=payload, timeout=30)
            if r.status_code == 200:
                break
            elif r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 5)))
            else:
                return json.dumps({"error": f"VEP HTTP {r.status_code}"})
        except Exception as e:
            if attempt == 2:
                return json.dumps({"error": str(e)})
            time.sleep(2)

    data = r.json()
    if not data:
        return json.dumps({"error": "No VEP results"})

    result = data[0]
    tcs = result.get("transcript_consequences", [])
    chosen = (next((tc for tc in tcs if tc.get("mane_select")), None)
              or next((tc for tc in tcs if tc.get("canonical") == 1), None)
              or (tcs[0] if tcs else None))

    if not chosen:
        return json.dumps({"error": "No transcript consequences found"})

    am = chosen.get("alphamissense", {})
    csq = chosen.get("consequence_terms", [])
    out = {
        "variant":        f"chr{chrom}:{pos} {ref}>{alt}",
        "gene":           chosen.get("gene_symbol", ""),
        "transcript":     chosen.get("transcript_id", ""),
        "hgvsp":          chosen.get("hgvsp", ""),
        "consequence":    ", ".join(csq),
        "sift_score":     chosen.get("sift_score"),
        "sift_pred":      chosen.get("sift_prediction"),
        "polyphen_score": chosen.get("polyphen_score"),
        "polyphen_pred":  chosen.get("polyphen_prediction"),
        "cadd_phred":     chosen.get("cadd_phred"),
        "am_pathogenicity": am.get("am_pathogenicity") if isinstance(am, dict) else None,
        "am_class":         am.get("am_class")         if isinstance(am, dict) else None,
        "phylop":           chosen.get("conservation"),
        "is_missense":    "missense_variant" in csq,
    }
    return json.dumps(out, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════
# TOOL 2: gnomAD Frequency
# ══════════════════════════════════════════════════════════════════════════
@tool
def gnomad_frequency(variant: str) -> str:
    """Query gnomAD v4 for population allele frequency of a variant.
    Returns AF, log10(AF), and ACMG population evidence (BA1/PM2).

    Args:
        variant: Variant in format "CHROM:POS:REF:ALT" e.g. "17:7674220:C:T"

    Returns:
        JSON string with AF and ACMG interpretation.
    """
    try:
        parts = variant.strip().replace("chr", "").split(":")
        chrom, pos, ref, alt = parts[0], int(parts[1]), parts[2].upper(), parts[3].upper()
    except Exception as e:
        return json.dumps({"error": f"Parse error: {e}"})

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
                    af = 0.0
                break
            time.sleep(2 ** attempt)
        except Exception:
            af = float("nan")
            time.sleep(2 ** attempt)
    else:
        af = float("nan")

    if math.isnan(af):
        return json.dumps({"error": "gnomAD API unavailable"})

    log_af = float(np.log10(af + 1e-8))

    # ACMG interpretation
    if af == 0.0:
        acmg = "PM2_Supporting — variant absent from gnomAD (may be novel)"
    elif af < 0.0001:
        acmg = "PM2_Supporting — very rare (AF < 0.01%)"
    elif af < 0.001:
        acmg = "PM2_Supporting — rare (AF < 0.1%)"
    elif af < 0.01:
        acmg = "Neutral — low frequency (AF < 1%)"
    elif af < 0.05:
        acmg = "BS1 — relatively common, weakly against pathogenicity"
    else:
        acmg = "BA1 — common variant (AF ≥ 5%), strong benign evidence"

    return json.dumps({
        "variant":   f"chr{chrom}:{pos} {ref}>{alt}",
        "af":        af,
        "af_str":    f"{af:.2e}" if af > 0 else "0 (absent)",
        "log10_af":  round(log_af, 4),
        "acmg_evidence": acmg,
    }, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════
# TOOL 3: Evo2 Score
# ══════════════════════════════════════════════════════════════════════════
@tool
def evo2_score(variant: str) -> str:
    """Score a variant using Evo2-40B genomic language model (NVIDIA NIM).
    Returns log-likelihood ratio (LLR): negative = alt allele less likely = pathogenic signal.
    Requires EVO2_API_KEY environment variable.

    Args:
        variant: Variant in format "CHROM:POS:REF:ALT" e.g. "17:7674220:C:T"

    Returns:
        JSON string with LLR and interpretation.
    """
    if not EVO2_API_KEY:
        return json.dumps({
            "llr": None,
            "note": "EVO2_API_KEY not set — Evo2 scoring unavailable. "
                    "Training median will be used for prediction."
        })

    try:
        parts = variant.strip().replace("chr", "").split(":")
        chrom, pos, ref, alt = parts[0], int(parts[1]), parts[2].upper(), parts[3].upper()
    except Exception as e:
        return json.dumps({"error": f"Parse error: {e}"})

    # Fetch genomic context
    start, end = pos - 50, pos + 50
    url = f"{ENSEMBL_SEQ}/{chrom}:{start}..{end}:1"
    try:
        r = requests.get(url, headers={"Accept": "application/json"}, timeout=15)
        seq = r.json().get("seq", "") if r.status_code == 200 else ""
    except Exception:
        seq = ""

    if not seq or len(seq) != 101:
        return json.dumps({"llr": None, "note": "Could not fetch genomic context for Evo2 scoring"})

    ref_ctx = seq
    alt_ctx = seq[:50] + alt + seq[51:]
    hdrs = {"Authorization": f"Bearer {EVO2_API_KEY}", "Content-Type": "application/json"}
    log_probs = {}
    for base, label in [(ref_ctx[50], "ref"), (alt_ctx[50], "alt")]:
        prefix = ref_ctx[:50]
        suffix = ref_ctx[51:56]
        payload = {"sequence": prefix + base + suffix, "num_tokens": 1,
                   "top_k": 4, "enable_sampled_probs": True, "temperature": 0.001}
        for attempt in range(3):
            try:
                r = requests.post(EVO2_URL, headers=hdrs, json=payload, timeout=30)
                if r.status_code == 200:
                    p = r.json().get("sampled_probs", [None])[0]
                    log_probs[label] = float(np.log(p)) if p and p > 0 else float("nan")
                    break
                elif r.status_code == 429:
                    time.sleep(5 * (attempt + 1))
            except Exception:
                time.sleep(2)

    llr = log_probs.get("alt", float("nan")) - log_probs.get("ref", float("nan"))
    if math.isnan(llr):
        return json.dumps({"llr": None, "note": "Evo2 API call failed"})

    if llr < -1.0:
        interp = "Strong pathogenic signal (alt allele evolutionarily disfavored)"
    elif llr < -0.3:
        interp = "Moderate pathogenic signal"
    elif llr < 0.3:
        interp = "Neutral"
    else:
        interp = "Benign signal (alt allele evolutionarily tolerated)"

    return json.dumps({"llr": round(llr, 4), "interpretation": interp}, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════
# TOOL 4: Genos Score
# ══════════════════════════════════════════════════════════════════════════
@tool
def genos_score(variant: str) -> str:
    """Score a variant using Genos-10B human genomic foundation model (Stomics/Zhejiang Lab).
    Returns pathogenicity score 0-1. Requires GENOS_API_KEY environment variable.

    Args:
        variant: Variant in format "CHROM:POS:REF:ALT" e.g. "17:7674220:C:T"

    Returns:
        JSON string with pathogenicity score and interpretation.
    """
    if not GENOS_API_KEY:
        return json.dumps({
            "score": None,
            "note": "GENOS_API_KEY not set — Genos scoring unavailable. "
                    "Training median will be used for prediction."
        })

    try:
        parts = variant.strip().replace("chr", "").split(":")
        chrom, pos, ref, alt = parts[0], int(parts[1]), parts[2].upper(), parts[3].upper()
    except Exception as e:
        return json.dumps({"error": f"Parse error: {e}"})

    hdrs = {"Authorization": f"Bearer {GENOS_API_KEY}", "Content-Type": "application/json"}
    payload = {"assembly": "hg38", "chrom": f"chr{chrom}",
               "pos": int(pos), "ref": ref, "alt": alt}
    for attempt in range(3):
        try:
            r = requests.post(GENOS_URL, headers=hdrs, json=payload, timeout=30)
            if r.status_code == 200:
                score = r.json().get("result", {}).get("score_Pathogenic", None)
                if score is not None:
                    if score > 0.7:
                        interp = "High pathogenicity"
                    elif score > 0.4:
                        interp = "Intermediate"
                    else:
                        interp = "Low pathogenicity"
                    return json.dumps({"score": round(float(score), 4),
                                       "interpretation": interp}, ensure_ascii=False)
            elif r.status_code == 429:
                time.sleep(5 * (attempt + 1))
        except Exception:
            time.sleep(2)

    return json.dumps({"score": None, "note": "Genos API unavailable"})


# ══════════════════════════════════════════════════════════════════════════
# TOOL 5: SNV-judge Predict
# ══════════════════════════════════════════════════════════════════════════
@tool
def snv_judge_predict(variant: str) -> str:
    """Run the full SNV-judge v5 prediction pipeline for a variant.
    Automatically calls VEP, gnomAD, Evo2, and Genos, then runs the
    ensemble model (XGBoost + LightGBM + Isotonic calibration).
    Returns calibrated probability, ACMG classification, and SHAP contributions.

    Args:
        variant: Variant in format "CHROM:POS:REF:ALT" e.g. "17:7674220:C:T"

    Returns:
        JSON string with full prediction result including ACMG badge and SHAP values.
    """
    import json as _json

    # Step 1: VEP
    vep_raw = json.loads(vep_annotate.invoke(variant))
    if "error" in vep_raw:
        return json.dumps({"error": f"VEP failed: {vep_raw['error']}"})

    # Step 2: gnomAD
    gnomad_raw = json.loads(gnomad_frequency.invoke(variant))
    gnomad_log_af = gnomad_raw.get("log10_af", float("nan"))
    if isinstance(gnomad_log_af, str):
        gnomad_log_af = float("nan")

    # Step 3: Evo2 + Genos (best-effort)
    evo2_raw   = json.loads(evo2_score.invoke(variant))
    genos_raw  = json.loads(genos_score.invoke(variant))
    evo2_llr   = evo2_raw.get("llr")   or float("nan")
    genos_path = genos_raw.get("score") or float("nan")

    # Step 4: Build feature vector
    def _med(key, default=0.0):
        try:
            return float(_MEDIANS[key])
        except Exception:
            return default

    def _val(v, med_key):
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            return float(v)
        return _med(med_key)

    sift_score = vep_raw.get("sift_score")
    sift_inv   = (1.0 - float(sift_score)) if sift_score is not None else _med("sift_score_inv", 0.5)

    feature_vec = [
        sift_inv,
        _val(vep_raw.get("polyphen_score"), "polyphen_score"),
        _val(vep_raw.get("am_pathogenicity"), "am_pathogenicity"),
        _val(vep_raw.get("cadd_phred"), "cadd_phred"),
        _val(evo2_llr, "evo2_llr"),
        _val(genos_path, "genos_path"),
        _val(vep_raw.get("phylop"), "phylop"),
        _val(gnomad_log_af, "gnomad_log_af"),
    ]
    X = np.array([feature_vec])

    # Step 5: Predict
    if isinstance(_MODEL, dict):
        xgb_p = _MODEL["xgb"].predict_proba(X)[0, 1]
        lgb_p = _MODEL["lgb"].predict_proba(X)[0, 1]
        meta_X = np.array([[xgb_p, lgb_p]])
        raw_prob = float(_MODEL["meta"].predict_proba(meta_X)[0, 1])
    else:
        raw_prob = float(_MODEL.predict_proba(X)[0, 1])

    # Step 6: Calibrate
    from sklearn.isotonic import IsotonicRegression
    if isinstance(_CALIBRATOR, IsotonicRegression):
        cal_prob = float(np.clip(_CALIBRATOR.predict([raw_prob])[0], 0.0, 1.0))
    else:
        logit = np.log(raw_prob / (1 - raw_prob + 1e-9))
        cal_prob = float(_CALIBRATOR.predict_proba([[logit]])[0, 1])

    # Step 7: SHAP
    try:
        import shap
        xgb_model = _MODEL["xgb"] if isinstance(_MODEL, dict) else _MODEL
        explainer = shap.TreeExplainer(xgb_model)
        shap_vals = explainer.shap_values(X)[0].tolist()
    except Exception:
        shap_vals = [0.0] * len(feature_vec)

    feat_labels = ["SIFT(inv)", "PolyPhen-2", "AlphaMissense", "CADD",
                   "Evo2 LLR", "Genos", "phyloP", "gnomAD log-AF"]
    shap_dict = {feat_labels[i]: round(shap_vals[i], 4)
                 for i in range(min(len(feat_labels), len(shap_vals)))}
    top_feature = max(shap_dict, key=lambda k: abs(shap_dict[k]))

    result = {
        "variant":          vep_raw.get("variant"),
        "gene":             vep_raw.get("gene", ""),
        "transcript":       vep_raw.get("transcript", ""),
        "hgvsp":            vep_raw.get("hgvsp", ""),
        "consequence":      vep_raw.get("consequence", ""),
        "model_version":    _MODEL_VER,
        "raw_probability":  round(raw_prob, 4),
        "calibrated_probability": round(cal_prob, 4),
        "acmg_classification":    _acmg_label(cal_prob),
        "scores": {
            "sift_inv":        round(sift_inv, 4),
            "polyphen":        vep_raw.get("polyphen_score"),
            "alphamissense":   vep_raw.get("am_pathogenicity"),
            "am_class":        vep_raw.get("am_class"),
            "cadd_phred":      vep_raw.get("cadd_phred"),
            "evo2_llr":        None if math.isnan(evo2_llr) else round(evo2_llr, 4),
            "genos_path":      None if math.isnan(genos_path) else round(genos_path, 4),
            "phylop":          vep_raw.get("phylop"),
            "gnomad_af":       gnomad_raw.get("af"),
            "gnomad_acmg":     gnomad_raw.get("acmg_evidence"),
        },
        "shap_contributions": shap_dict,
        "top_contributing_feature": top_feature,
    }
    return json.dumps(result, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════
# TOOL 6: ClinVar / Literature Search
# ══════════════════════════════════════════════════════════════════════════
@tool
def search_clinvar(query: str) -> str:
    """Search ClinVar for existing classifications of a variant or gene.
    Uses NCBI E-utilities to retrieve ClinVar records.

    Args:
        query: Search query e.g. "TP53 R175H" or "BRCA1 pathogenic missense" or
               "NM_000546.6:c.524G>A"

    Returns:
        JSON string with top ClinVar records (up to 5).
    """
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    try:
        # Search
        search_r = requests.get(
            f"{base}/esearch.fcgi",
            params={"db": "clinvar", "term": query, "retmax": 5,
                    "retmode": "json", "sort": "relevance"},
            timeout=15,
        )
        if search_r.status_code != 200:
            return json.dumps({"error": f"ClinVar search HTTP {search_r.status_code}"})

        ids = search_r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return json.dumps({"message": f"No ClinVar records found for: {query}",
                               "records": []})

        # Fetch summaries
        summary_r = requests.get(
            f"{base}/esummary.fcgi",
            params={"db": "clinvar", "id": ",".join(ids),
                    "retmode": "json"},
            timeout=15,
        )
        if summary_r.status_code != 200:
            return json.dumps({"error": "ClinVar summary fetch failed"})

        result_data = summary_r.json().get("result", {})
        records = []
        for uid in ids:
            rec = result_data.get(uid, {})
            if not rec:
                continue
            records.append({
                "clinvar_id":      uid,
                "title":           rec.get("title", ""),
                "clinical_significance": rec.get("clinical_significance", {}).get("description", ""),
                "review_status":   rec.get("clinical_significance", {}).get("review_status", ""),
                "gene":            rec.get("genes", [{}])[0].get("symbol", "") if rec.get("genes") else "",
                "variation_type":  rec.get("obj_type", ""),
                "last_evaluated":  rec.get("clinical_significance", {}).get("last_evaluated", ""),
            })

        return json.dumps({"query": query, "total_found": len(ids),
                           "records": records}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)})


# ══════════════════════════════════════════════════════════════════════════
# TOOL 7: Save Patient Variant (Long-term Memory)
# ══════════════════════════════════════════════════════════════════════════
@tool
def save_patient_variant(data: str) -> str:
    """Save a variant analysis result to a patient's long-term profile.
    Creates the patient profile if it doesn't exist.

    Args:
        data: JSON string with fields:
              - patient_id (str): Patient identifier e.g. "P001" or "Zhang San"
              - variant (str): "CHROM:POS:REF:ALT"
              - gene (str): Gene symbol
              - hgvsp (str): Protein change
              - acmg (str): ACMG classification
              - probability (float): Calibrated probability
              - notes (str, optional): Clinical notes

    Returns:
        Confirmation message.
    """
    try:
        d = json.loads(data)
    except Exception:
        return "Error: data must be a valid JSON string."

    required = ["patient_id", "variant", "acmg", "probability"]
    missing = [k for k in required if k not in d]
    if missing:
        return f"Error: missing required fields: {missing}"

    # Load existing profiles
    if PATIENT_DB.exists():
        with open(PATIENT_DB, "r", encoding="utf-8") as f:
            profiles = json.load(f)
    else:
        profiles = {}

    pid = str(d["patient_id"])
    if pid not in profiles:
        profiles[pid] = {"patient_id": pid, "variants": [], "created_at": _now()}

    import datetime
    entry = {
        "variant":     d["variant"],
        "gene":        d.get("gene", ""),
        "hgvsp":       d.get("hgvsp", ""),
        "acmg":        d["acmg"],
        "probability": d["probability"],
        "notes":       d.get("notes", ""),
        "saved_at":    _now(),
    }
    profiles[pid]["variants"].append(entry)
    profiles[pid]["last_updated"] = _now()

    PATIENT_DB.parent.mkdir(parents=True, exist_ok=True)
    with open(PATIENT_DB, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    return (f"Saved variant {d['variant']} ({d.get('gene','')}) "
            f"to patient profile '{pid}'. "
            f"ACMG: {d['acmg']}, Probability: {d['probability']:.1%}. "
            f"Total variants for this patient: {len(profiles[pid]['variants'])}.")


# ══════════════════════════════════════════════════════════════════════════
# TOOL 8: Get Patient History (Long-term Memory)
# ══════════════════════════════════════════════════════════════════════════
@tool
def get_patient_history(patient_id: str) -> str:
    """Retrieve all historical variant analyses for a patient.

    Args:
        patient_id: Patient identifier e.g. "P001" or "Zhang San"

    Returns:
        JSON string with patient profile and all historical variants.
    """
    if not PATIENT_DB.exists():
        return json.dumps({"message": "No patient profiles found yet.", "variants": []})

    with open(PATIENT_DB, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    pid = str(patient_id)
    if pid not in profiles:
        # Try case-insensitive search
        matches = [k for k in profiles if k.lower() == pid.lower()]
        if matches:
            pid = matches[0]
        else:
            all_ids = list(profiles.keys())
            return json.dumps({
                "message": f"Patient '{patient_id}' not found.",
                "available_patients": all_ids,
            })

    profile = profiles[pid]
    variants = profile.get("variants", [])

    # Summary stats
    acmg_counts = {}
    for v in variants:
        acmg_counts[v["acmg"]] = acmg_counts.get(v["acmg"], 0) + 1

    return json.dumps({
        "patient_id":    pid,
        "total_variants": len(variants),
        "acmg_summary":  acmg_counts,
        "created_at":    profile.get("created_at", ""),
        "last_updated":  profile.get("last_updated", ""),
        "variants":      variants,
    }, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════
# TOOL 9: Batch VCF Analysis
# ══════════════════════════════════════════════════════════════════════════
@tool
def batch_vcf_analyze(vcf_text: str) -> str:
    """Parse and analyze multiple variants from VCF-format text.
    Scores all SNVs using the SNV-judge v5 pipeline and returns a ranked summary.
    Top 3 high-risk variants are automatically analyzed in depth.

    Args:
        vcf_text: VCF-format text (with or without header lines).
                  Each data line: CHROM  POS  ID  REF  ALT  ...
                  Also accepts simplified format: one "CHROM:POS:REF:ALT" per line.

    Returns:
        JSON string with ranked variant list and summary statistics.
    """
    lines = vcf_text.strip().splitlines()
    variants = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Try tab/space-separated VCF format
        parts = line.replace("\t", " ").split()
        if len(parts) >= 5:
            chrom = parts[0].replace("chr", "")
            try:
                pos = int(parts[1])
                ref, alt = parts[3].upper(), parts[4].upper()
                if len(ref) == 1 and len(alt) == 1 and ref != alt and ref in "ACGT" and alt in "ACGT":
                    variants.append(f"{chrom}:{pos}:{ref}:{alt}")
            except ValueError:
                continue
        # Try colon-separated format
        elif ":" in line:
            variants.append(line)

    if not variants:
        return json.dumps({"error": "No valid SNVs found in input. "
                           "Expected VCF format or CHROM:POS:REF:ALT per line."})

    MAX_VARIANTS = 20
    if len(variants) > MAX_VARIANTS:
        note = f"Input has {len(variants)} variants; processing first {MAX_VARIANTS}."
        variants = variants[:MAX_VARIANTS]
    else:
        note = f"Processing {len(variants)} variants."

    results = []
    for var in variants:
        try:
            pred_raw = snv_judge_predict.invoke(var)
            pred = json.loads(pred_raw)
            if "error" not in pred:
                results.append({
                    "variant":     pred.get("variant", var),
                    "gene":        pred.get("gene", ""),
                    "hgvsp":       pred.get("hgvsp", ""),
                    "probability": pred.get("calibrated_probability", 0),
                    "acmg":        pred.get("acmg_classification", ""),
                    "top_feature": pred.get("top_contributing_feature", ""),
                    "gnomad_acmg": pred.get("scores", {}).get("gnomad_acmg", ""),
                })
        except Exception as e:
            results.append({"variant": var, "error": str(e)})

    # Sort by probability descending
    results.sort(key=lambda x: x.get("probability", 0), reverse=True)

    # Summary
    acmg_counts = {}
    for r in results:
        acmg = r.get("acmg", "Unknown")
        acmg_counts[acmg] = acmg_counts.get(acmg, 0) + 1

    top3 = [r for r in results if "error" not in r][:3]

    return json.dumps({
        "note":          note,
        "total_scored":  len(results),
        "acmg_summary":  acmg_counts,
        "top_3_high_risk": top3,
        "all_results":   results,
    }, ensure_ascii=False)


# ── Helper ─────────────────────────────────────────────────────────────────
def _now() -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── Export all tools ───────────────────────────────────────────────────────
ALL_TOOLS = [
    vep_annotate,
    gnomad_frequency,
    evo2_score,
    genos_score,
    snv_judge_predict,
    search_clinvar,
    save_patient_variant,
    get_patient_history,
    batch_vcf_analyze,
]
