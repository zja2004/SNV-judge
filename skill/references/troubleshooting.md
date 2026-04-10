# SNV-judge v5 Troubleshooting Guide

---

## Installation Issues

### `ModuleNotFoundError: No module named 'xgboost'`
```bash
pip install xgboost lightgbm scikit-learn shap numpy pandas requests
```

### `ModuleNotFoundError: No module named 'streamlit'`
```bash
pip install streamlit openai
```

---

## API Issues

### VEP API timeout or HTTP 500
- Ensembl REST API may be temporarily down
- Check status: https://rest.ensembl.org
- Retry automatically (3 attempts built-in)
- Fallback: use pre-computed `data/vep_scores.pkl` for training variants

### Evo2 API HTTP 429 (rate limit)
```python
# The script retries automatically with exponential backoff
# For batch scoring, use pre-computed scores:
import pickle
with open("data/scoring_ckpt.pkl", "rb") as f:
    precomputed = pickle.load(f)
```

### gnomAD GraphQL API unavailable
- Model imputes missing gnomAD AF with training median: log10(AF) = -4.48
- This is equivalent to AF ≈ 3.3×10⁻⁵ (rare variant, PM2 supporting)
- Check gnomAD status: https://gnomad.broadinstitute.org

### Kimi API `AuthenticationError`
```bash
export KIMI_API_KEY="sk-..."  # Get from https://platform.moonshot.cn
```

---

## Model Issues

### AUROC inflated when evaluating on training data
- **Problem**: Evaluating the pre-trained model on `feature_matrix_v4.csv` gives AUROC ~0.999 because the model was trained on these variants
- **Solution**: Use Leave-One-Gene-Out CV for unbiased evaluation:
```python
from scripts.generalization_eval import run_logo_cv
results = run_logo_cv("data/feature_matrix_v4.csv", model_dir=".")
```

### Model predicts all variants as pathogenic
- Check if gnomAD AF is NaN for all variants (shifts score distribution)
- Verify feature values are in expected ranges (see ACMG criteria reference)
- Check model version: `print(MODEL_VER)` — should be "v5" or "v4"

### `FileNotFoundError: xgb_model_v5.pkl`
- Run `python train.py --use-cache` to retrain from pre-computed scores
- Or download model artifacts from the GitHub repository

### Calibration seems off (probabilities too extreme)
- v5 uses Isotonic Regression (better calibration than v4 Platt scaling)
- Brier score should be ~0.0685 on held-out data
- If using v4 model, Brier score ~0.0743 is expected

---

## Data Issues

### VCF file not parsed correctly
- Ensure VCF is GRCh38 (not hg19/GRCh37)
- Only SNVs processed (REF and ALT must each be single nucleotides)
- Multi-allelic sites are skipped
- Check VCF header: must start with `##fileformat=VCFv4`

### Feature matrix column mismatch
Expected columns in `feature_matrix_v4.csv`:
```
allele_id, chrom, pos, ref, alt, gene, label,
sift_score, sift_score_inv, sift_pred,
polyphen_score, polyphen_pred,
am_pathogenicity, am_class,
cadd_phred, evo2_llr, genos_path, genos_benign, genos_pred,
phylop, gnomad_af, gnomad_log_af
```

### Missing values after imputation
- Training medians used for imputation:
  - sift_inv: 1.00 | polyphen: 0.744 | alphamissense: 0.374
  - cadd: 24.6 | evo2_llr: -0.074 | genos_score: 0.676
  - phylop: 1.66 | gnomad_log_af: -4.48

---

## Generalization Issues

### LOGO-CV takes too long
- Default: 16 genes × 5-fold OOF = 80 model fits (~5–10 min)
- Reduce test genes: `run_logo_cv(..., test_genes=["TP53", "MLH1"])`
- Reduce folds: `run_logo_cv(..., n_folds=3)`

### MYH7 AUROC consistently low (~0.79)
- This is expected — cardiomyopathy variants have atypical feature profiles
- Many MYH7 pathogenic variants have intermediate SIFT/PolyPhen scores
- Gain-of-function mechanism not captured by current feature set
- Potential improvement: add protein structure features (e.g., distance to myosin head)
