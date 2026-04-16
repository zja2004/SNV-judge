# Troubleshooting — SNV-judge v5.2

## Model loading errors

**`FileNotFoundError: xgb_model_v5.pkl`**
→ You are not in the SNV-judge repo root. Run `load_artifacts("/absolute/path/to/SNV-judge")`.

**`ModuleNotFoundError: xgboost / lightgbm / shap`**
```bash
pip install xgboost lightgbm scikit-learn shap numpy pandas requests
```

**`ModuleNotFoundError: streamlit`** (only needed for UI)
```bash
pip install streamlit
```

## API errors

**Ensembl VEP timeout or HTTP 500**
- Built-in retry (3 attempts). If still failing, check https://rest.ensembl.org
- Prediction will still run using median imputation for VEP-derived features

**gnomAD GraphQL unavailable**
- Model imputes missing AF with training median: log-AF = −4.48 (≈ 3.3×10⁻⁵)
- This is treated as a rare variant (PM2 supporting) — slightly biases toward pathogenic

**Evo2 HTTP 429 (rate limit)**
- Model automatically falls back to median imputation (evo2_llr = −0.074)
- AUROC impact: 0.9985 → 0.9892

**`resolve_protein_variant` returns None**
- Ensembl HGVS API could not map the protein change to genomic coordinates
- Try providing chrom/pos/ref/alt directly instead

## Prediction issues

**All variants predicted as Pathogenic**
- Check gnomAD AF: if all variants are absent from gnomAD, the model will lean pathogenic
- Verify feature values are in expected ranges (see acmg-criteria.md)

**AUROC inflated when testing on training data**
- `feature_matrix_v4.csv` IS the training set — evaluating on it gives ~0.999 (overfitting artifact)
- Use LOGO-CV for unbiased evaluation:
```python
from skill.scripts.generalization_eval import run_logo_cv
results = run_logo_cv("data/feature_matrix_v4.csv", model_dir=".")
# Expected: Non-BRCA mean AUROC ≈ 0.9642
```

**MYH7 predictions seem wrong**
- Expected behavior: MYH7 LOGO-CV AUROC = 0.79 (lowest of 16 genes)
- Gain-of-function mechanism → intermediate SIFT/PolyPhen scores → model underestimates pathogenicity
- Always flag MYH7 predictions as lower confidence

## VCF issues

**VCF not parsed**
- Must be GRCh38 (not hg19). Liftover first if needed.
- Only SNVs processed (single-nucleotide REF and ALT)
- Header must start with `##fileformat=VCFv4`

**Slow batch prediction**
- Default batch_size=50 for VEP. Max 200.
- For > 500 variants, split into chunks and run overnight

## Coordinate system

- All coordinates are **GRCh38 / hg38**, 1-based
- Chromosome format: `"17"` not `"chr17"`
- If user provides hg19 coordinates, tell them to liftover using UCSC or Ensembl
