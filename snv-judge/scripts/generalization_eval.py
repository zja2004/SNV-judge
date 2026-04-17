"""
generalization_eval.py — SNV-judge v5 Generalization Validation
===============================================================
Leave-One-Gene-Out (LOGO) cross-validation to evaluate model generalization
across disease gene families beyond BRCA1/2.

Usage:
    from scripts.generalization_eval import run_logo_cv
    results = run_logo_cv("data/feature_matrix_v4.csv", model_dir=".")
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                              roc_auc_score, accuracy_score)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# Default genes to test (covers major disease categories)
DEFAULT_TEST_GENES = [
    "TP53",   # Tumor suppressor
    "MLH1",   # Lynch MMR
    "MSH2",   # Lynch MMR
    "MSH6",   # Lynch MMR
    "MYH7",   # Cardiomyopathy
    "FBN1",   # Connective tissue
    "LDLR",   # Metabolic
    "MECP2",  # Neurodevelopmental
    "RYR1",   # Musculoskeletal
    "GAA",    # Metabolic/Lysosomal
    "USH2A",  # Retinal dystrophy
    "RUNX1",  # Hematologic
    "SOS1",   # RASopathy
    "RAF1",   # RASopathy
    "BRCA1",  # Reference
    "BRCA2",  # Reference
]

FEATURE_MAP = {
    "sift_score_inv":   "sift_inv",
    "polyphen_score":   "polyphen",
    "am_pathogenicity": "alphamissense",
    "cadd_phred":       "cadd",
    "evo2_llr":         "evo2_llr",
    "genos_path":       "genos_score",
    "phylop":           "phylop",
    "gnomad_log_af":    "gnomad_log_af",
}
FEATURE_NAMES = list(FEATURE_MAP.values())


def load_feature_matrix(csv_path: str, medians: dict) -> tuple:
    """Load and impute feature matrix. Returns (X_df, y_series, gene_series)."""
    df = pd.read_csv(csv_path)

    # Rename columns to model feature names
    df_feat = df.copy()
    for src, dst in FEATURE_MAP.items():
        if src in df.columns:
            df_feat[dst] = df[src]

    # Impute missing values with training medians
    X = df_feat[FEATURE_NAMES].copy()
    for col in FEATURE_NAMES:
        X[col] = X[col].fillna(medians.get(col, 0.0))

    y = (df["label"] == "pathogenic").astype(int)
    genes = df["gene"]

    print(f"Loaded {len(df)} variants, {df['gene'].nunique()} genes")
    print(f"Missing after imputation: {X.isnull().sum().sum()}")
    return X, y, genes


def train_stacking_model(X_train: pd.DataFrame, y_train: pd.Series,
                          n_folds: int = 5) -> dict:
    """Train XGB + LGB stacking model with OOF meta-learner."""
    import xgboost as xgb
    import lightgbm as lgb

    xgb_m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                random_state=42, eval_metric="logloss", verbosity=0)
    lgb_m = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 random_state=42, verbose=-1)

    # OOF predictions for meta-learner
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_xgb = np.zeros(len(y_train))
    oof_lgb = np.zeros(len(y_train))

    for tr_idx, val_idx in skf.split(X_train, y_train):
        Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr = y_train.iloc[tr_idx]
        xgb_m.fit(Xtr, ytr)
        lgb_m.fit(Xtr, ytr)
        oof_xgb[val_idx] = xgb_m.predict_proba(Xval)[:, 1]
        oof_lgb[val_idx] = lgb_m.predict_proba(Xval)[:, 1]

    meta = LogisticRegression(random_state=42)
    meta.fit(np.column_stack([oof_xgb, oof_lgb]), y_train)

    # Retrain base models on full training set
    xgb_m.fit(X_train, y_train)
    lgb_m.fit(X_train, y_train)

    return {"xgb": xgb_m, "lgb": lgb_m, "meta": meta}


def predict_with_model(model_dict: dict, X: pd.DataFrame) -> np.ndarray:
    """Run stacking prediction (no calibration for LOGO-CV)."""
    xp = model_dict["xgb"].predict_proba(X)[:, 1]
    lp = model_dict["lgb"].predict_proba(X)[:, 1]
    return model_dict["meta"].predict_proba(np.column_stack([xp, lp]))[:, 1]


def run_logo_cv(feature_matrix_path: str,
                model_dir: str = ".",
                test_genes: list = None,
                n_folds: int = 5,
                verbose: bool = True) -> list[dict]:
    """
    Leave-One-Gene-Out Cross-Validation.

    For each test gene:
      - Train stacking model on all OTHER genes
      - Evaluate on held-out gene variants

    Args:
        feature_matrix_path: path to feature_matrix_v4.csv
        model_dir: path to directory with train_medians_v5.pkl
        test_genes: list of gene symbols to test (default: DEFAULT_TEST_GENES)
        n_folds: number of folds for OOF meta-learner training
        verbose: print progress

    Returns:
        List of dicts with gene, AUROC, AUPRC, F1, Accuracy, FP, FN, probs, y_true
    """
    if test_genes is None:
        test_genes = DEFAULT_TEST_GENES

    # Load medians for imputation
    base = Path(model_dir)
    medians_path = next(
        (base / f"train_medians{s}.pkl" for s in ["_v5", "_v4", "_v3", "_v2", ""]
         if (base / f"train_medians{s}.pkl").exists()),
        None
    )
    if medians_path is None:
        raise FileNotFoundError(f"No train_medians*.pkl found in {model_dir}")

    with open(medians_path, "rb") as f:
        medians = pickle.load(f)

    X_all, y_all, genes_all = load_feature_matrix(feature_matrix_path, medians)

    results = []
    for gene in test_genes:
        test_mask  = genes_all == gene
        train_mask = ~test_mask

        if test_mask.sum() == 0:
            if verbose:
                print(f"  {gene}: not found in dataset, skipping")
            continue

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test  = X_all[test_mask]
        y_test  = y_all[test_mask]

        if y_test.nunique() < 2:
            if verbose:
                print(f"  {gene}: only one class in test set, skipping")
            continue

        # Train model on all other genes
        model_dict = train_stacking_model(X_train, y_train, n_folds=n_folds)

        # Predict on held-out gene
        probs = predict_with_model(model_dict, X_test)
        preds = (probs >= 0.5).astype(int)

        auroc = roc_auc_score(y_test, probs)
        auprc = average_precision_score(y_test, probs)
        f1    = f1_score(y_test, preds)
        acc   = accuracy_score(y_test, preds)
        fp    = int(((preds == 1) & (y_test == 0)).sum())
        fn    = int(((preds == 0) & (y_test == 1)).sum())

        result = {
            "gene":     gene,
            "n":        int(len(y_test)),
            "n_P":      int(y_test.sum()),
            "n_B":      int((1 - y_test).sum()),
            "AUROC":    auroc,
            "AUPRC":    auprc,
            "F1":       f1,
            "Accuracy": acc,
            "FP":       fp,
            "FN":       fn,
            "probs":    probs,
            "y_true":   y_test.values,
            "is_brca":  gene in ("BRCA1", "BRCA2"),
        }
        results.append(result)

        if verbose:
            print(f"  {gene:<10}: AUROC={auroc:.4f}  F1={f1:.4f}  Acc={acc:.4f}  "
                  f"(n={len(y_test)}, FP={fp}, FN={fn})")

    # Summary
    if verbose and results:
        non_brca = [r for r in results if not r["is_brca"]]
        brca     = [r for r in results if r["is_brca"]]
        print(f"\nNon-BRCA mean AUROC: {np.mean([r['AUROC'] for r in non_brca]):.4f} "
              f"(n={len(non_brca)} genes)")
        if brca:
            print(f"BRCA1/2 mean AUROC:  {np.mean([r['AUROC'] for r in brca]):.4f}")

    return results


def results_to_dataframe(logo_results: list) -> pd.DataFrame:
    """Convert LOGO-CV results list to a clean DataFrame."""
    return pd.DataFrame([{
        k: v for k, v in r.items() if k not in ("probs", "y_true")
    } for r in logo_results]).round(4)


# ── CLI usage ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    matrix_path = sys.argv[1] if len(sys.argv) > 1 else "data/feature_matrix_v4.csv"
    model_dir   = sys.argv[2] if len(sys.argv) > 2 else "."

    print(f"Running LOGO-CV on {matrix_path}")
    results = run_logo_cv(matrix_path, model_dir=model_dir)

    df = results_to_dataframe(results)
    out_path = "generalization_logo_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
