"""
train.py — Full SNV-judge v2 training pipeline
===============================================
Reproduces the complete pipeline from data acquisition to model artefacts.

Steps:
  1. Fetch ClinVar gold-standard variants (multi-gene, ≥2-star review)
  2. Annotate via Ensembl VEP REST API (SIFT, PolyPhen-2, AlphaMissense, CADD)
  3. Score variants with Evo2 (NVIDIA NIM API) and Genos (Stomics API)
  4. Clean data + 5-fold cross-validation
  5. Train XGBoost + LightGBM Stacking ensemble
  6. Platt calibration
  7. SHAP analysis + figures
  8. Save all model artefacts

New in v2:
  - Expanded dataset: 10,542 ClinVar missense variants across 2,927 genes
  - Evo2 (Arc Institute / NVIDIA): zero-shot log-likelihood ratio scoring
    via NVIDIA NIM API (evo2-40b model)
  - Genos (Zhejiang Lab): human-centric genomic foundation model
    pathogenicity score via Stomics cloud API
  - Stacking ensemble: XGBoost + LightGBM meta-learner

Usage:
  # Set API keys as environment variables:
  export EVO2_API_KEY="nvapi-..."
  export GENOS_API_KEY="sk-..."
  python train.py

Outputs (saved to current directory):
  xgb_model_v2.pkl, platt_scaler_v2.pkl, shap_explainer_v2.pkl, train_medians_v2.pkl
  data/clinvar_raw_v2.csv, data/feature_matrix_v2.xlsx, data/model_metrics_v2.csv
  figures/model_curves_v2.png, figures/shap_analysis_v2.png
"""

import os, time, pickle, warnings
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import xgboost as xgb
import lightgbm as lgb
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)
warnings.filterwarnings("ignore")

# ── Directories ───────────────────────────────────────────────────────────
Path("data").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

# ── API Keys (set via environment variables) ──────────────────────────────
EVO2_API_KEY  = os.environ.get("EVO2_API_KEY", "")
GENOS_API_KEY = os.environ.get("GENOS_API_KEY", "")
EVO2_URL  = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/generate"
GENOS_URL = "https://cloud.stomics.tech/api/aigateway/genos/variant_predict"

FEATURE_COLS   = ["sift_score_inv", "polyphen_score", "am_pathogenicity",
                  "cadd_phred", "evo2_llr", "genos_path"]
FEATURE_LABELS = ["SIFT (inv)", "PolyPhen-2", "AlphaMissense", "CADD Phred",
                  "Evo2 LLR", "Genos Score"]

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Fetch ClinVar variants
# ══════════════════════════════════════════════════════════════════════════
def fetch_clinvar(gene: str, gene_id: str) -> list[dict]:
    """Fetch pathogenic/benign missense variants from ClinVar via NCBI eutils."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    # Search for variants
    search_url = f"{base}/esearch.fcgi"
    params = {
        "db": "clinvar", "retmax": 5000, "retmode": "json",
        "term": f"{gene}[gene] AND missense[molecular consequence] AND homo sapiens[organism]"
    }
    r = requests.get(search_url, params=params, timeout=30)
    ids = r.json()["esearchresult"]["idlist"]
    print(f"  {gene}: {len(ids)} ClinVar IDs found")

    records = []
    for i in range(0, len(ids), 100):
        batch = ids[i:i+100]
        r2 = requests.post(
            f"{base}/efetch.fcgi",
            data={"db": "clinvar", "rettype": "vcv", "retmode": "json",
                  "id": ",".join(batch)},
            timeout=60
        )
        if r2.status_code != 200:
            continue
        for vcv in r2.json().get("VariationArchive", []):
            try:
                clinsig = (vcv.get("ClassifiedRecord", {})
                             .get("Classifications", {})
                             .get("GermlineClassification", {})
                             .get("Description", "")).lower()
                if "pathogenic" in clinsig and "conflicting" not in clinsig:
                    label = "pathogenic"
                elif "benign" in clinsig and "conflicting" not in clinsig:
                    label = "benign"
                else:
                    continue

                loc = (vcv.get("ClassifiedRecord", {})
                          .get("SimpleAllele", {})
                          .get("Location", {})
                          .get("SequenceLocation", []))
                if isinstance(loc, dict):
                    loc = [loc]
                grch38 = next((l for l in loc
                               if l.get("Assembly") == "GRCh38"), None)
                if not grch38:
                    continue

                spdi = grch38.get("positionVCF")
                chrom = grch38.get("Chr", "")
                pos   = grch38.get("positionVCF")
                ref   = grch38.get("referenceAlleleVCF", "")
                alt   = grch38.get("alternateAlleleVCF", "")

                if not all([chrom, pos, ref, alt]):
                    continue
                if len(ref) != 1 or len(alt) != 1:
                    continue

                records.append({
                    "variation_id": vcv.get("VariationID", ""),
                    "gene": gene,
                    "chrom": chrom,
                    "pos": int(pos),
                    "ref": ref,
                    "alt": alt,
                    "clinsig": clinsig,
                    "label": label,
                })
            except Exception:
                continue
        time.sleep(0.4)
    return records


def step1_fetch_clinvar():
    print("\n=== Step 1: Fetching ClinVar variants ===")
    all_records = []
    for gene, gid in GENE_IDS.items():
        recs = fetch_clinvar(gene, gid)
        all_records.extend(recs)
        print(f"  {gene}: {len(recs)} usable variants")

    df = pd.DataFrame(all_records).drop_duplicates(subset=["chrom","pos","ref","alt"])
    df = df[df["label"].isin(["pathogenic","benign"])]
    df.to_csv("data/clinvar_raw.csv", index=False)
    print(f"  Total: {len(df)} variants saved to data/clinvar_raw.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Fetch VEP scores
# ══════════════════════════════════════════════════════════════════════════
VEP_URL = "https://rest.ensembl.org/vep/homo_sapiens/region"
VEP_HDR = {"Content-Type": "application/json", "Accept": "application/json"}

def vep_batch(variants_df: pd.DataFrame) -> list:
    vep_strings = [f"{r.chrom} {r.pos} . {r.ref} {r.alt} . . ."
                   for r in variants_df.itertuples()]
    payload = {"variants": vep_strings, "AlphaMissense": 1,
               "CADD": 1, "REVEL": 1, "canonical": 1, "mane": 1}
    for attempt in range(3):
        try:
            r = requests.post(VEP_URL, headers=VEP_HDR, json=payload, timeout=60)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 10)))
        except Exception as e:
            print(f"    VEP error: {e}")
            time.sleep(5)
    return []

def extract_scores(vep_result, chrom, pos, ref, alt) -> dict:
    row = {"chrom": chrom, "pos": pos, "ref": ref, "alt": alt,
           "sift_score": None, "polyphen_score": None,
           "am_pathogenicity": None, "cadd_phred": None, "revel": None}
    if not vep_result:
        return row
    tcs = vep_result.get("transcript_consequences", [])
    chosen = next((tc for tc in tcs if tc.get("mane_select")), None)
    if not chosen:
        chosen = next((tc for tc in tcs if tc.get("canonical") == 1), None)
    if not chosen and tcs:
        chosen = tcs[0]
    if not chosen:
        return row
    row["sift_score"]     = chosen.get("sift_score")
    row["polyphen_score"] = chosen.get("polyphen_score")
    row["cadd_phred"]     = chosen.get("cadd_phred")
    row["revel"]          = chosen.get("revel")
    am = chosen.get("alphamissense")
    if isinstance(am, dict):
        row["am_pathogenicity"] = am.get("am_pathogenicity")
    return row

def step2_fetch_vep(df_raw: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Step 2: Fetching VEP scores ===")
    records, batches = [], [df_raw.iloc[i:i+50] for i in range(0, len(df_raw), 50)]
    for idx, batch_df in enumerate(batches):
        results = vep_batch(batch_df)
        result_map = {}
        for res in results:
            parts = res.get("input", "").split()
            if len(parts) >= 5:
                result_map[(parts[0], int(parts[1]), parts[3], parts[4])] = res
        for row in batch_df.itertuples():
            key = (str(row.chrom), int(row.pos), row.ref, row.alt)
            sc  = extract_scores(result_map.get(key), row.chrom, row.pos, row.ref, row.alt)
            sc["variation_id"] = row.variation_id
            records.append(sc)
        if (idx + 1) % 5 == 0:
            print(f"  [{min((idx+1)*50, len(df_raw))}/{len(df_raw)}] done")
        time.sleep(0.5)

    df_scores = pd.DataFrame(records)
    print("  Coverage:")
    for col in ["sift_score","polyphen_score","am_pathogenicity","cadd_phred","revel"]:
        n = df_scores[col].notna().sum()
        print(f"    {col:20s}: {n}/{len(df_scores)} ({100*n/len(df_scores):.1f}%)")
    return df_scores


# ══════════════════════════════════════════════════════════════════════════
# STEP 3: Clean + split
# ══════════════════════════════════════════════════════════════════════════
def step3_clean_v2(df_raw: pd.DataFrame, df_scores: pd.DataFrame,
                   df_ai: pd.DataFrame) -> pd.DataFrame:
    """
    Merge VEP scores + AI scores, compute derived features, drop rows with
    all classical scores missing.  Returns a single DataFrame ready for CV.
    """
    print("\n=== Step 3: Cleaning + merging features ===")

    # Merge VEP scores
    vep_cols = ["variation_id", "sift_score", "polyphen_score",
                "am_pathogenicity", "cadd_phred"]
    df = df_raw.merge(df_scores[vep_cols], on="variation_id", how="left")

    # Merge AI scores
    if df_ai is not None and len(df_ai):
        df = df.merge(df_ai[["variation_id", "evo2_llr", "genos_path"]],
                      on="variation_id", how="left")
    else:
        df["evo2_llr"]   = np.nan
        df["genos_path"] = np.nan

    # Derived features
    df["sift_score_inv"] = 1.0 - df["sift_score"]
    df["label_bin"]      = (df["label"] == "pathogenic").astype(int)

    # Drop rows where ALL classical scores are missing
    classical = ["sift_score_inv", "polyphen_score", "am_pathogenicity", "cadd_phred"]
    df = df[~df[classical].isnull().all(axis=1)].copy()

    print(f"  Dataset: {len(df):,} variants "
          f"({df['label_bin'].mean()*100:.1f}% pathogenic, "
          f"{df['gene'].nunique()} genes)")
    print(f"  Feature coverage:")
    for col in FEATURE_COLS:
        n = df[col].notna().sum()
        print(f"    {col:20s}: {n:,}/{len(df):,} ({100*n/len(df):.1f}%)")

    df.to_csv("data/clinvar_v2_features.csv", index=False)
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 4: Train v2 model with 5-fold CV + Stacking ensemble
# ══════════════════════════════════════════════════════════════════════════
def bootstrap_metrics(y_true, y_prob, n_boot=200, seed=42):
    rng = np.random.default_rng(seed)
    aurocs, auprcs = [], []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        aurocs.append(roc_auc_score(yt, yp))
        auprcs.append(average_precision_score(yt, yp))
    return (np.percentile(aurocs, [2.5, 97.5]),
            np.percentile(auprcs, [2.5, 97.5]))


def _make_stacking_model(spw: float):
    """Build XGBoost + LightGBM stacking classifier."""
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0)
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=42, n_jobs=-1, verbose=-1)
    meta = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    return StackingClassifier(
        estimators=[("xgb", xgb_clf), ("lgb", lgb_clf)],
        final_estimator=meta,
        cv=3,
        passthrough=False,
        n_jobs=1,
    )


def step4_train_v2(df: pd.DataFrame):
    """
    5-fold stratified CV on the full dataset.
    Returns: (final_model, platt_scaler, medians, oof_probs, y_all, metrics_df)
    """
    print("\n=== Step 4: Training v2 Stacking model (5-fold CV) ===")

    # Impute missing values with column medians
    X_raw = df[FEATURE_COLS].values.astype(float)
    y     = df["label_bin"].values

    medians = {col: float(np.nanmedian(X_raw[:, i]))
               for i, col in enumerate(FEATURE_COLS)}
    for i, col in enumerate(FEATURE_COLS):
        mask = np.isnan(X_raw[:, i])
        X_raw[mask, i] = medians[col]

    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_probs = np.zeros(len(y))
    fold_aurocs, fold_auprcs = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_raw, y)):
        X_tr, X_va = X_raw[tr_idx], X_raw[va_idx]
        y_tr, y_va = y[tr_idx],     y[va_idx]

        model = _make_stacking_model(spw)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)[:, 1]
        oof_probs[va_idx] = proba

        auroc = roc_auc_score(y_va, proba)
        auprc = average_precision_score(y_va, proba)
        fold_aurocs.append(auroc)
        fold_auprcs.append(auprc)
        print(f"  Fold {fold+1}: AUROC={auroc:.4f}  AUPRC={auprc:.4f}")

    print(f"\n  CV mean AUROC = {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")
    print(f"  CV mean AUPRC = {np.mean(fold_auprcs):.4f} ± {np.std(fold_auprcs):.4f}")

    # Platt calibration on OOF predictions
    logit_oof = np.log(oof_probs / (1 - oof_probs + 1e-9))
    platt = LogisticRegression(max_iter=1000)
    platt.fit(logit_oof.reshape(-1, 1), y)
    cal_probs = platt.predict_proba(logit_oof.reshape(-1, 1))[:, 1]

    # Single-tool baselines (OOF normalised)
    single_probas = {}
    for i, lbl in enumerate(FEATURE_LABELS):
        sc = X_raw[:, i].copy()
        mn, mx = sc.min(), sc.max()
        single_probas[lbl] = np.clip((sc - mn) / (mx - mn + 1e-9), 0, 1)

    # Metrics table
    rows = []
    for name, prob in ([("XGBoost+LGB Stacking (calibrated)", cal_probs),
                        ("XGBoost+LGB Stacking (raw OOF)",    oof_probs)]
                       + list(single_probas.items())):
        auroc = roc_auc_score(y, prob)
        auprc = average_precision_score(y, prob)
        ci_r, ci_p = bootstrap_metrics(y, prob)
        rows.append({"Model": name,
                     "AUROC": f"{auroc:.4f} [{ci_r[0]:.3f}–{ci_r[1]:.3f}]",
                     "AUPRC": f"{auprc:.4f} [{ci_p[0]:.3f}–{ci_p[1]:.3f}]"})
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv("data/model_metrics_v2.csv", index=False)
    print("\n" + df_metrics.to_string(index=False))

    # Retrain final model on full dataset
    print("\n  Retraining final model on full dataset…")
    final_model = _make_stacking_model(spw)
    final_model.fit(X_raw, y)

    return final_model, platt, medians, cal_probs, oof_probs, y, single_probas, X_raw, df_metrics


# ══════════════════════════════════════════════════════════════════════════
# STEP 5: SHAP + figures (v2 — 6 features, OOF evaluation)
# ══════════════════════════════════════════════════════════════════════════
def step5_shap_figures_v2(final_model, X_all, y_all,
                          cal_probs, oof_probs, single_probas):
    """Generate SHAP beeswarm + ROC/PR curves using OOF predictions."""
    print("\n=== Step 5: SHAP + figures ===")
    sns.set_theme(style="ticks", font_scale=1.05)

    # Use the XGBoost base estimator for SHAP (StackingClassifier wraps it)
    xgb_base = final_model.named_estimators_["xgb"]
    explainer   = shap.TreeExplainer(xgb_base)
    shap_values = explainer.shap_values(X_all)
    mean_abs    = np.abs(shap_values).mean(axis=0)
    order_idx   = np.argsort(mean_abs)[::-1]
    feat_sorted = [FEATURE_LABELS[i] for i in order_idx]
    y_pos_map   = {f: r for r, f in enumerate(feat_sorted)}

    # ── SHAP figure ───────────────────────────────────────────────────────
    FEAT_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    vals = mean_abs[order_idx]
    bar_cols = [FEAT_COLORS[i] for i in order_idx[::-1]]
    bars = ax.barh(feat_sorted[::-1], vals[::-1], color=bar_cols, height=0.6)
    for bar, v in zip(bars, vals[::-1]):
        ax.text(v + vals.max() * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left", fontsize=10)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Feature Importance (Mean |SHAP|)\n5-fold OOF", fontsize=12, fontweight="bold")
    ax.set_xlim(0, vals.max() * 1.3)
    sns.despine(ax=ax)

    ax = axes[1]
    cmap = plt.cm.RdBu_r
    rng_ = np.random.default_rng(42)
    for i in order_idx:
        feat = FEATURE_LABELS[i]
        sv   = shap_values[:, i]
        fv   = X_all[:, i].copy()
        fv_norm = np.clip((fv - fv.min()) / (fv.max() - fv.min() + 1e-9), 0, 1)
        jitter = rng_.uniform(-0.3, 0.3, size=len(sv))
        ax.scatter(sv, y_pos_map[feat] + jitter, c=fv_norm, cmap=cmap,
                   s=12, alpha=0.6, linewidths=0, vmin=0, vmax=1)
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_yticks(range(len(feat_sorted)))
    ax.set_yticklabels(feat_sorted, fontsize=11)
    ax.set_xlabel("SHAP value", fontsize=10)
    ax.set_title("SHAP Beeswarm (5-fold OOF)", fontsize=12, fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Mid", "High"])
    sns.despine(ax=ax)
    fig.tight_layout(pad=2.0)
    fig.savefig("figures/shap_analysis_v2.png", dpi=150, bbox_inches="tight")
    fig.savefig("figures/shap_analysis_v2.svg", bbox_inches="tight")
    plt.close()
    print("  Saved figures/shap_analysis_v2.png")

    # ── ROC / PR curves ───────────────────────────────────────────────────
    COLORS = {
        "Stacking (calibrated)": "#0072B2",
        "Stacking (raw OOF)":    "#56B4E9",
        "AlphaMissense":         "#009E73",
        "CADD Phred":            "#CC79A7",
        "Evo2 LLR":              "#E69F00",
        "Genos Score":           "#D55E00",
        "PolyPhen-2":            "#999999",
        "SIFT (inv)":            "#000000",
    }
    LWIDTHS = {k: 2.5 if "Stacking" in k else 1.5 for k in COLORS}
    LSTYLES = {
        "Stacking (calibrated)": "-",
        "Stacking (raw OOF)":    "--",
        "AlphaMissense":         "-",
        "CADD Phred":            "-.",
        "Evo2 LLR":              ":",
        "Genos Score":           "--",
        "PolyPhen-2":            (0, (3, 1, 1, 1)),
        "SIFT (inv)":            (0, (5, 2)),
    }
    all_p = {"Stacking (calibrated)": cal_probs,
             "Stacking (raw OOF)":    oof_probs,
             **single_probas}
    curves = {}
    for name, prob in all_p.items():
        if name not in COLORS:
            continue
        fpr, tpr, _ = roc_curve(y_all, prob)
        prec, rec, _ = precision_recall_curve(y_all, prob)
        curves[name] = {"fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec,
                        "auroc": roc_auc_score(y_all, prob),
                        "auprc": average_precision_score(y_all, prob)}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, metric, xl, yl, title, lloc in [
        (axes[0], "auroc", "FPR",    "TPR",       "ROC Curves",  "lower right"),
        (axes[1], "auprc", "Recall", "Precision", "PR Curves",   "upper right"),
    ]:
        if metric == "auprc":
            ax.axhline(y_all.mean(), color="k", ls=":", lw=0.8, alpha=0.4,
                       label=f"Random (P={y_all.mean():.3f})")
        else:
            ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.4)
        for name, c in curves.items():
            v  = c[metric]
            xd = c["fpr"] if metric == "auroc" else c["rec"]
            yd = c["tpr"] if metric == "auroc" else c["prec"]
            ax.plot(xd, yd, color=COLORS[name], lw=LWIDTHS[name], ls=LSTYLES[name],
                    label=f"{name} ({'AUROC' if metric=='auroc' else 'AUPRC'}={v:.3f})")
        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel(yl, fontsize=11)
        ax.set_title(f"{title} — 5-fold OOF", fontsize=12, fontweight="bold")
        ax.legend(fontsize=7.5, loc=lloc, framealpha=0.92)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        sns.despine(ax=ax)
    fig.tight_layout(pad=2.0)
    fig.savefig("figures/model_curves_v2.png", dpi=150, bbox_inches="tight")
    fig.savefig("figures/model_curves_v2.svg", bbox_inches="tight")
    plt.close()
    print("  Saved figures/model_curves_v2.png")
    return explainer


# ══════════════════════════════════════════════════════════════════════════
# STEP 6: Save v2 artefacts
# ══════════════════════════════════════════════════════════════════════════
def step6_save_v2(final_model, platt, medians):
    """Save model artefacts with _v2 suffix."""
    print("\n=== Step 6: Saving v2 model artefacts ===")
    for obj, fname in [(final_model, "xgb_model_v2.pkl"),
                       (platt,       "platt_scaler_v2.pkl"),
                       (medians,     "train_medians_v2.pkl")]:
        with open(fname, "wb") as f:
            pickle.dump(obj, f)
        print(f"  Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 2b: Evo2 zero-shot variant scoring (NVIDIA NIM API)
# ══════════════════════════════════════════════════════════════════════════
def evo2_score_variant(ref_context: str, alt_context: str) -> float:
    """
    Zero-shot variant effect score using Evo2 log-likelihood ratio.

    Method: Feed (prefix + base + suffix) to Evo2, get P(next_token | context).
    Score = log P(continuation | alt_context) - log P(continuation | ref_context)
    Negative score → alt disrupts sequence → potentially pathogenic.

    Requires EVO2_API_KEY environment variable.
    """
    if not EVO2_API_KEY:
        return np.nan
    mid = len(ref_context) // 2
    prefix = ref_context[:mid]
    suffix = ref_context[mid+1:mid+6]
    hdrs = {"Authorization": f"Bearer {EVO2_API_KEY}",
            "Content-Type": "application/json"}
    log_probs = {}
    for base, label in [(ref_context[mid], 'ref'), (alt_context[mid], 'alt')]:
        payload = {"sequence": prefix + base + suffix, "num_tokens": 1,
                   "top_k": 4, "enable_sampled_probs": True, "temperature": 0.001}
        for attempt in range(3):
            try:
                r = requests.post(EVO2_URL, headers=hdrs, json=payload, timeout=30)
                if r.status_code == 200:
                    p = r.json().get('sampled_probs', [None])[0]
                    log_probs[label] = np.log(p) if p and p > 0 else np.nan
                    break
                elif r.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                else:
                    time.sleep(1)
            except Exception:
                time.sleep(2)
        time.sleep(0.05)
    return log_probs.get('alt', np.nan) - log_probs.get('ref', np.nan)


# ══════════════════════════════════════════════════════════════════════════
# STEP 2c: Genos variant pathogenicity scoring (Stomics API)
# ══════════════════════════════════════════════════════════════════════════
def genos_score_variant(chrom: str, pos: int, ref: str, alt: str) -> dict:
    """
    Genos (Zhejiang Lab) human-centric genomic foundation model score.
    Returns score_Pathogenic and score_Benign.

    Requires GENOS_API_KEY environment variable.
    """
    if not GENOS_API_KEY:
        return {'genos_path': np.nan, 'genos_benign': np.nan}
    hdrs = {"Authorization": f"Bearer {GENOS_API_KEY}",
            "Content-Type": "application/json"}
    payload = {"assembly": "hg38", "chrom": f"chr{chrom}",
               "pos": int(pos), "ref": ref, "alt": alt}
    for attempt in range(3):
        try:
            r = requests.post(GENOS_URL, headers=hdrs, json=payload, timeout=30)
            if r.status_code == 200:
                res = r.json().get('result', {})
                return {'genos_path':   res.get('score_Pathogenic', np.nan),
                        'genos_benign': res.get('score_Benign', np.nan)}
            elif r.status_code == 429:
                time.sleep(5 * (attempt + 1))
            else:
                time.sleep(1)
        except Exception:
            time.sleep(2)
    return {'genos_path': np.nan, 'genos_benign': np.nan}


def step2b_score_ai_features(df: pd.DataFrame, n_workers: int = 10) -> pd.DataFrame:
    """Score all variants with Evo2 and Genos in parallel."""
    print(f"\n=== Step 2b/c: Scoring {len(df):,} variants with Evo2 + Genos ===")

    def score_one(args):
        idx, row = args
        e = evo2_score_variant(row['ref_context'], row['alt_context'])
        g = genos_score_variant(row['chrom'], row['pos'], row['ref'], row['alt'])
        return idx, e, g

    results = {}
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(score_one, rv): rv[0] for rv in df.iterrows()}
        done = 0
        for fut in as_completed(futs):
            try:
                idx, e, g = fut.result()
                results[idx] = {'evo2_llr': e, **g}
            except Exception:
                idx = futs[fut]
                results[idx] = {'evo2_llr': np.nan, 'genos_path': np.nan,
                                'genos_benign': np.nan}
            done += 1
            if done % 200 == 0 or done == len(df):
                print(f"  {done:,}/{len(df):,} ({done/len(df)*100:.0f}%)")

    ai_df = pd.DataFrame.from_dict(results, orient='index')
    return df.join(ai_df)


# ══════════════════════════════════════════════════════════════════════════
# MAIN — v2 end-to-end pipeline
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import hashlib, datetime

    print("SNV-judge v2 Training Pipeline")
    print("=" * 60)
    print(f"  EVO2_API_KEY  : {'set' if EVO2_API_KEY  else 'NOT SET — Evo2 scoring disabled'}")
    print(f"  GENOS_API_KEY : {'set' if GENOS_API_KEY else 'NOT SET — Genos scoring disabled'}")
    print()

    # ── Step 1: ClinVar data ──────────────────────────────────────────────
    # The v2 pipeline expects a pre-downloaded ClinVar parquet/CSV with
    # columns: variation_id, gene, chrom, pos, ref, alt, label, ref_context, alt_context
    # (produced by the data preparation notebook / download script).
    # If the file exists, load it; otherwise fall back to NCBI eutils fetch.
    CLINVAR_V2 = Path("data/clinvar_v2_sample.parquet")
    CLINVAR_V2_CSV = Path("data/clinvar_v2_sample.csv")

    if CLINVAR_V2.exists():
        print(f"[Step 1] Loading pre-sampled ClinVar data from {CLINVAR_V2}")
        df_raw = pd.read_parquet(CLINVAR_V2)
    elif CLINVAR_V2_CSV.exists():
        print(f"[Step 1] Loading pre-sampled ClinVar data from {CLINVAR_V2_CSV}")
        df_raw = pd.read_csv(CLINVAR_V2_CSV)
    else:
        print("[Step 1] No pre-sampled data found — fetching from NCBI eutils…")
        print("  (For large-scale training, download variant_summary.txt.gz from ClinVar)")
        df_raw = step1_fetch_clinvar()
        df_raw.to_csv("data/clinvar_raw_v2.csv", index=False)

    # Record data provenance
    prov = {
        "date":     datetime.datetime.utcnow().isoformat(),
        "n_total":  len(df_raw),
        "n_path":   int((df_raw["label"] == "pathogenic").sum()),
        "n_benign": int((df_raw["label"] == "benign").sum()),
        "n_genes":  int(df_raw["gene"].nunique()),
    }
    print(f"  Loaded {prov['n_total']:,} variants "
          f"({prov['n_path']:,} P / {prov['n_benign']:,} B) "
          f"across {prov['n_genes']:,} genes")

    # ── Step 2a: VEP annotation ───────────────────────────────────────────
    VEP_CACHE = Path("data/vep_scores_v2.pkl")
    if VEP_CACHE.exists():
        print(f"\n[Step 2a] Loading cached VEP scores from {VEP_CACHE}")
        with open(VEP_CACHE, "rb") as f:
            df_vep = pickle.load(f)
    else:
        print("\n[Step 2a] Fetching VEP scores…")
        df_vep = step2_fetch_vep(df_raw)
        with open(VEP_CACHE, "wb") as f:
            pickle.dump(df_vep, f)

    # ── Step 2b/c: Evo2 + Genos scoring ──────────────────────────────────
    AI_CACHE = Path("data/ai_scores_v2.pkl")
    df_ai = None
    if AI_CACHE.exists():
        print(f"\n[Step 2b] Loading cached AI scores from {AI_CACHE}")
        with open(AI_CACHE, "rb") as f:
            df_ai = pickle.load(f)
    elif EVO2_API_KEY or GENOS_API_KEY:
        print("\n[Step 2b] Scoring variants with Evo2 + Genos…")
        # Requires ref_context / alt_context columns in df_raw
        if "ref_context" not in df_raw.columns:
            print("  WARNING: ref_context column missing — skipping AI scoring.")
            print("  Run the data preparation script to fetch genomic context sequences.")
        else:
            df_ai_scored = step2b_score_ai_features(df_raw)
            df_ai = df_ai_scored[["variation_id", "evo2_llr", "genos_path"]].copy()
            with open(AI_CACHE, "wb") as f:
                pickle.dump(df_ai, f)
    else:
        print("\n[Step 2b] No API keys set — skipping Evo2/Genos scoring.")
        print("  Set EVO2_API_KEY and GENOS_API_KEY to enable AI features.")

    # ── Step 3: Clean + merge ─────────────────────────────────────────────
    df_clean = step3_clean_v2(df_raw, df_vep, df_ai)

    # ── Step 4: Train v2 model ────────────────────────────────────────────
    (final_model, platt, medians,
     cal_probs, oof_probs, y_all,
     single_probas, X_all, df_metrics) = step4_train_v2(df_clean)

    # ── Step 5: SHAP + figures ────────────────────────────────────────────
    step5_shap_figures_v2(final_model, X_all, y_all,
                          cal_probs, oof_probs, single_probas)

    # ── Step 6: Save artefacts ────────────────────────────────────────────
    step6_save_v2(final_model, platt, medians)

    # Save provenance
    import json
    prov["model_metrics"] = df_metrics.to_dict(orient="records")
    with open("data/training_provenance_v2.json", "w") as f:
        json.dump(prov, f, indent=2)
    print("\n  Saved data/training_provenance_v2.json")

    print("\n" + "=" * 60)
    print("Done! Run the app with:  streamlit run app.py")
    print("=" * 60)
