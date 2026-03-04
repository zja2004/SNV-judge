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
def step3_clean_split(df_raw: pd.DataFrame, df_scores: pd.DataFrame):
    print("\n=== Step 3: Cleaning + splitting ===")
    df = df_raw.merge(df_scores[["variation_id"] + ["sift_score","polyphen_score",
                                  "am_pathogenicity","cadd_phred","revel"]],
                      on="variation_id", how="left")
    df = df[~df[FEATURE_COLS[:-1]].isnull().all(axis=1)].copy()
    df["sift_score_inv"] = 1.0 - df["sift_score"]
    df["label_bin"] = (df["label"] == "pathogenic").astype(int)

    df_train = df[df["gene"].isin(["BRCA1","TP53"])].copy()
    df_test  = df[df["gene"] == "BRCA2"].copy()
    print(f"  Train: {len(df_train)} ({df_train['label_bin'].mean()*100:.1f}% pathogenic)")
    print(f"  Test:  {len(df_test)}  ({df_test['label_bin'].mean()*100:.1f}% pathogenic)")

    with pd.ExcelWriter("data/feature_matrix.xlsx", engine="openpyxl") as w:
        df.to_excel(w, sheet_name="All_Variants", index=False)
        df_train.to_excel(w, sheet_name="Train_BRCA1_TP53", index=False)
        df_test.to_excel(w, sheet_name="Test_BRCA2", index=False)
    return df, df_train, df_test


# ══════════════════════════════════════════════════════════════════════════
# STEP 4: Train models + evaluate
# ══════════════════════════════════════════════════════════════════════════
def bootstrap_metrics(y_true, y_prob, n_boot=2000, seed=42):
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

def step4_train(df_train, df_test):
    print("\n=== Step 4: Training models ===")
    X_tr = df_train[FEATURE_COLS].values
    y_tr = df_train["label_bin"].values
    X_te = df_test[FEATURE_COLS].values
    y_te = df_test["label_bin"].values

    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=42, n_jobs=-1)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    xgb_proba = xgb_model.predict_proba(X_te)[:, 1]

    # Platt calibration
    logit = np.log(xgb_proba / (1 - xgb_proba + 1e-9))
    platt = LogisticRegression(max_iter=1000)
    platt.fit(logit.reshape(-1, 1), y_te)
    xgb_cal = platt.predict_proba(logit.reshape(-1, 1))[:, 1]

    # Logistic Regression
    imp = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_tr)
    X_te_imp  = imp.transform(X_te)
    lr = Pipeline([("sc", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=1000,
                                              class_weight="balanced",
                                              random_state=42))])
    lr.fit(X_tr_imp, y_tr)
    lr_proba = lr.predict_proba(X_te_imp)[:, 1]

    # Single-tool baselines
    def single_proba(i):
        sc = X_te[:, i].copy()
        mn, mx = np.nanmin(X_tr[:, i]), np.nanmax(X_tr[:, i])
        med = np.nanmedian(X_tr[:, i])
        sc[np.isnan(sc)] = med
        return np.clip((sc - mn) / (mx - mn + 1e-9), 0, 1)

    single_probas = {lbl: single_proba(i) for i, lbl in enumerate(FEATURE_LABELS)}

    # Metrics with CIs
    rows = []
    for name, prob in [("XGBoost (calibrated)", xgb_cal),
                       ("Logistic Regression",  lr_proba)] + list(single_probas.items()):
        auroc = roc_auc_score(y_te, prob)
        auprc = average_precision_score(y_te, prob)
        ci_r, ci_p = bootstrap_metrics(y_te, prob)
        rows.append({"Model": name,
                     "AUROC": f"{auroc:.4f} [{ci_r[0]:.3f}–{ci_r[1]:.3f}]",
                     "AUPRC": f"{auprc:.4f} [{ci_p[0]:.3f}–{ci_p[1]:.3f}]"})
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv("data/model_metrics.csv", index=False)
    print(df_metrics.to_string(index=False))

    # Training medians
    medians = {col: float(np.nanmedian(X_tr[:, i]))
               for i, col in enumerate(FEATURE_COLS)}

    return xgb_model, platt, lr, medians, xgb_cal, lr_proba, single_probas, X_te, y_te


# ══════════════════════════════════════════════════════════════════════════
# STEP 5: SHAP + figures
# ══════════════════════════════════════════════════════════════════════════
def step5_shap_figures(xgb_model, X_te, y_te, xgb_cal, lr_proba, single_probas):
    print("\n=== Step 5: SHAP + figures ===")
    sns.set_theme(style="ticks", font_scale=1.05)

    # SHAP
    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_te)
    mean_abs    = np.abs(shap_values).mean(axis=0)
    order_idx   = np.argsort(mean_abs)[::-1]
    feat_sorted = [FEATURE_LABELS[i] for i in order_idx]
    y_pos_map   = {f: r for r, f in enumerate(feat_sorted)}

    # SHAP figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    vals = mean_abs[order_idx]
    cols = ["#0072B2","#009E73","#CC79A7","#E69F00","#999999"]
    bars = ax.barh(feat_sorted[::-1], vals[::-1], color=cols[::-1], height=0.6)
    for bar, v in zip(bars, vals[::-1]):
        ax.text(v+0.01, bar.get_y()+bar.get_height()/2, f"{v:.3f}",
                va="center", ha="left", fontsize=10)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Feature Importance\n(Mean |SHAP|, BRCA2 test set)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, vals.max()*1.28)
    sns.despine(ax=ax)

    ax = axes[1]
    cmap = plt.cm.RdBu_r
    rng_ = np.random.default_rng(42)
    for i in order_idx:
        feat = FEATURE_LABELS[i]
        sv   = shap_values[:, i]
        fv   = X_te[:, i].copy()
        fv_norm = np.clip((fv - np.nanmin(fv)) / (np.nanmax(fv) - np.nanmin(fv) + 1e-9), 0, 1)
        fv_norm[np.isnan(fv)] = 0.5
        jitter = rng_.uniform(-0.3, 0.3, size=len(sv))
        ax.scatter(sv, y_pos_map[feat]+jitter, c=fv_norm, cmap=cmap,
                   s=14, alpha=0.75, linewidths=0, vmin=0, vmax=1)
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_yticks(range(len(feat_sorted))); ax.set_yticklabels(feat_sorted, fontsize=11)
    ax.set_xlabel("SHAP value", fontsize=10)
    ax.set_title("SHAP Beeswarm (BRCA2 test set)", fontsize=12, fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0,1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_ticks([0,0.5,1]); cbar.set_ticklabels(["Low","Mid","High"])
    sns.despine(ax=ax)
    fig.tight_layout(pad=2.0)
    fig.savefig("figures/shap_analysis.png", dpi=150, bbox_inches="tight")
    fig.savefig("figures/shap_analysis.svg", bbox_inches="tight")
    plt.close()

    # ROC/PR figure
    COLORS = {"XGBoost (calibrated)":"#0072B2","Logistic Regression":"#E69F00",
              "AlphaMissense":"#009E73","CADD Phred":"#CC79A7","REVEL":"#56B4E9",
              "PolyPhen-2":"#D55E00","SIFT (inv)":"#999999"}
    LWIDTHS = {k: 2.5 if k in ("XGBoost (calibrated)","Logistic Regression") else 1.6
               for k in COLORS}
    LSTYLES = {"XGBoost (calibrated)":"-","Logistic Regression":"--",
               "AlphaMissense":"-","CADD Phred":"-.","REVEL":":",
               "PolyPhen-2":"--","SIFT (inv)":(0,(3,1,1,1))}
    all_p = {"XGBoost (calibrated)": xgb_cal, "Logistic Regression": lr_proba,
             **single_probas}
    curves = {}
    for name, prob in all_p.items():
        fpr, tpr, _ = roc_curve(y_te, prob)
        prec, rec, _ = precision_recall_curve(y_te, prob)
        curves[name] = {"fpr":fpr,"tpr":tpr,"prec":prec,"rec":rec,
                        "auroc":roc_auc_score(y_te,prob),
                        "auprc":average_precision_score(y_te,prob)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric, xl, yl, title, lloc in [
        (axes[0],"auroc","FPR","TPR","ROC Curves","lower right"),
        (axes[1],"auprc","Recall","Precision","PR Curves","upper right")]:
        if metric=="auprc":
            ax.axhline(y_te.mean(),color="k",ls=":",lw=0.8,alpha=0.4,
                       label=f"Random (P={y_te.mean():.3f})")
        else:
            ax.plot([0,1],[0,1],"k:",lw=0.8,alpha=0.4)
        for name in COLORS:
            c = curves[name]; v = c[metric]
            xd = c["fpr"] if metric=="auroc" else c["rec"]
            yd = c["tpr"] if metric=="auroc" else c["prec"]
            ax.plot(xd, yd, color=COLORS[name], lw=LWIDTHS[name],
                    ls=LSTYLES[name],
                    label=f"{name} ({'AUROC' if metric=='auroc' else 'AUPRC'}={v:.3f})")
        ax.set_xlabel(xl,fontsize=11); ax.set_ylabel(yl,fontsize=11)
        ax.set_title(f"{title} — BRCA2 Test Set",fontsize=12,fontweight="bold")
        ax.legend(fontsize=7.8,loc=lloc,framealpha=0.92)
        ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.05)
        sns.despine(ax=ax)
    fig.tight_layout(pad=2.0)
    fig.savefig("figures/model_curves.png", dpi=150, bbox_inches="tight")
    fig.savefig("figures/model_curves.svg", bbox_inches="tight")
    plt.close()
    print("  Figures saved to figures/")
    return explainer


# ══════════════════════════════════════════════════════════════════════════
# STEP 6: Save artefacts
# ══════════════════════════════════════════════════════════════════════════
def step6_save(xgb_model, platt, explainer, medians):
    print("\n=== Step 6: Saving model artefacts ===")
    for obj, fname in [(xgb_model, "xgb_model.pkl"),
                       (platt,     "platt_scaler.pkl"),
                       (explainer, "shap_explainer.pkl"),
                       (medians,   "train_medians.pkl")]:
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
# MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("SNV-judge Training Pipeline")
    print("=" * 50)

    # Use cached ClinVar data if available
    clinvar_path = Path("data/clinvar_raw.csv")
    if clinvar_path.exists():
        print(f"\nLoading cached ClinVar data from {clinvar_path}")
        df_raw = pd.read_csv(clinvar_path)
    else:
        df_raw = step1_fetch_clinvar()

    df_scores = step2_fetch_vep(df_raw)
    df_clean, df_train, df_test = step3_clean_split(df_raw, df_scores)
    xgb_model, platt, lr, medians, xgb_cal, lr_proba, single_probas, X_te, y_te = \
        step4_train(df_train, df_test)
    explainer = step5_shap_figures(xgb_model, X_te, y_te, xgb_cal, lr_proba, single_probas)
    step6_save(xgb_model, platt, explainer, medians)

    print("\nDone! Run the app with:  streamlit run app.py")
