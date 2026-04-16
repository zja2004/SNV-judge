---
name: snv-judge
description: Predict the pathogenicity of a human missense SNV (single nucleotide variant) using SNV-judge v5.2. Returns a calibrated probability (0–1), ACMG 5-tier classification (P/LP/VUS/LB/B), and per-feature SHAP contributions. Use this skill whenever a user asks to classify, score, or interpret a genetic variant, or wants to know whether a mutation is likely disease-causing.
---

# SNV-judge v5.2 — Variant Pathogenicity Prediction

## Overview

SNV-judge is a stacking ensemble (XGBoost + LightGBM + Isotonic calibration) trained on 2,000 ClinVar variants across 547 genes. It uses 8 features fetched from free public APIs (Ensembl VEP, gnomAD) plus optional AI scores (Evo2, Genos).

**No API keys required for basic prediction.** Evo2 and Genos scores are imputed from training medians when unavailable (AUROC impact: 0.9985 → 0.9892).

---

## Decision Tree: Which function to call

```
User request
├── Single variant (chrom/pos/ref/alt OR gene + protein change)
│   └── predict_variant()  →  then print_shap_summary()
│
├── Multiple variants at once
│   └── loop predict_variant()  →  collect results into a table
│
├── VCF file provided
│   └── predict_vcf()  →  saves CSV automatically
│
├── User wants to know WHY a variant scored high/low
│   └── predict_variant()  →  print_shap_summary()  →  explain top drivers
│
├── User provides gene name + amino acid change (e.g. "TP53 R175H")
│   └── resolve_protein_variant()  →  then predict_variant()
│
└── User wants ClinVar ground truth alongside prediction
    └── predict_variant()  +  fetch_clinvar_classification()
```

---

## Setup (run once per session)

```python
import sys
sys.path.insert(0, '.')  # run from SNV-judge repo root
from skill.scripts.predict import (
    load_artifacts,
    predict_variant,
    resolve_protein_variant,
    predict_vcf,
    fetch_clinvar_classification,
    print_result,
    print_shap_summary,
)

# Load model — do this ONCE and reuse artifacts for all predictions
artifacts = load_artifacts(".")
```

**If `load_artifacts` fails:** see [Troubleshooting](./references/troubleshooting.md#model-issues).

---

## Core Functions

### 1. Predict a single variant

```python
result = predict_variant(
    chrom="17",        # chromosome, no "chr" prefix, GRCh38
    pos=7674220,       # 1-based position
    ref="C",           # reference allele
    alt="T",           # alternate allele
    artifacts=artifacts,
    genos_url=None,    # optional: local ngrok URL for Genos embedding
)

# Quick summary
print_shap_summary(result)

# Full formatted output
print_result(result)
```

**Key result fields:**

| Field | Type | Description |
|---|---|---|
| `cal_prob` | float | Calibrated P(pathogenic), 0–1 |
| `acmg_class` | str | `Pathogenic` / `Likely Pathogenic` / `VUS` / `Likely Benign` / `Benign` |
| `acmg_confidence` | str | `High` / `Moderate` / `Low` |
| `shap_values` | list[float] | Per-feature SHAP contributions (log-odds) |
| `feature_labels` | list[str] | Feature names matching shap_values |
| `feature_vec` | list[float] | Actual feature values used |
| `top_shap_feature` | str | Feature with largest absolute SHAP |
| `genos_source` | str | `API` / `embedding` / `median_imputed` |
| `vep_scores` | dict | Gene, transcript, HGVSp, consequence from VEP |

### 2. Predict from gene + protein change

```python
# User says "TP53 R175H" — resolve to genomic coordinates first
coords = resolve_protein_variant("TP53", "R175H")
# coords = {"chrom": "17", "pos": 7674220, "ref": "C", "alt": "T", ...}

result = predict_variant(
    coords["chrom"], coords["pos"], coords["ref"], coords["alt"],
    artifacts=artifacts,
)
print_shap_summary(result)
```

### 3. Batch predict from VCF file

```python
results = predict_vcf(
    vcf_path="variants.vcf",   # .vcf or .vcf.gz, GRCh38
    artifacts=artifacts,
    output_csv="results.csv",  # auto-saved
    batch_size=50,             # VEP batch size, max 200
)
# results is a list of dicts, same structure as predict_variant()
```

### 4. Fetch ClinVar ground truth

```python
cv = fetch_clinvar_classification("17", 7674220, "C", "T")
print(cv["clinical_significance"])  # e.g. "Pathogenic"
print(cv["review_status"])          # e.g. "reviewed by expert panel"
print(cv["conditions"])             # list of associated diseases
```

---

## Interpreting Results

### ACMG probability thresholds

| cal_prob | Classification | Meaning |
|---|---|---|
| ≥ 0.90 | Pathogenic | Strong computational evidence of pathogenicity |
| 0.70–0.90 | Likely Pathogenic | Moderate evidence |
| 0.40–0.70 | VUS | Uncertain — borderline features |
| 0.20–0.40 | Likely Benign | Moderate evidence of benignity |
| < 0.20 | Benign | Strong evidence of benignity |

### SHAP interpretation

- **Positive SHAP** → feature pushes toward pathogenic
- **Negative SHAP** → feature pushes toward benign
- SHAP values are in **log-odds space** (not probability space)
- `top_shap_feature` is the single most influential feature for this variant
- `genos_source = "median_imputed"` means Genos/Evo2 were unavailable — marked with `*` in output

### Known limitations

- **MYH7 (cardiomyopathy)**: LOGO-CV AUROC = 0.79 (lowest of 16 tested genes). Gain-of-function mechanism causes intermediate SIFT/PolyPhen scores. Flag predictions on MYH7 variants as lower confidence.
- **Missense only**: Model trained on missense SNVs. Do not use for synonymous, intronic, or frameshift variants — VEP will warn if consequence is not missense.
- **GRCh38 only**: Coordinates must be hg38. If user provides hg19, tell them to liftover first.
- **Not for clinical use**: Provides computational evidence (PP3/BP4) only. Clinical classification requires functional data, segregation, de novo status, etc.

For full ACMG criteria details: [acmg-criteria.md](./references/acmg-criteria.md)

---

## Example: Complete workflow

```python
import sys
sys.path.insert(0, '.')
from skill.scripts.predict import *

artifacts = load_artifacts(".")

# Predict TP53 R175H
coords  = resolve_protein_variant("TP53", "R175H")
result  = predict_variant(coords["chrom"], coords["pos"],
                          coords["ref"], coords["alt"],
                          artifacts=artifacts)
cv      = fetch_clinvar_classification(coords["chrom"], coords["pos"],
                                       coords["ref"], coords["alt"])

print(f"P(pathogenic) : {result['cal_prob']:.1%}")
print(f"ACMG class    : {result['acmg_class']}")
print(f"ClinVar       : {cv['clinical_significance']}")
print_shap_summary(result)
```

Expected output:
```
P(pathogenic) : 100.0%
ACMG class    : Pathogenic
ClinVar       : Pathogenic/Likely pathogenic

SHAP Feature Contributions
──────────────────────────────────────────────────
  AlphaMissense          1.000   SHAP +2.627  ▲ pathogenic
  gnomAD log-AF         -8.000   SHAP +2.730  ▲ pathogenic
  CADD Phred            29.600   SHAP +0.898  ▲ pathogenic
  Genos-10B              0.974   SHAP +0.433  ▲ pathogenic
  phyloP                 2.550   SHAP +0.170  ▲ pathogenic
  SIFT (inv)             1.000   SHAP +0.171  ▲ pathogenic
  PolyPhen-2             0.998   SHAP +0.137  ▲ pathogenic
  Evo2 LLR*             -0.074   SHAP -1.281  ▼ benign
──────────────────────────────────────────────────
  Top driver: gnomAD log-AF  (SHAP +2.730)
  Calibrated P(pathogenic): 100.0%
  ACMG class: Pathogenic
```

---

## Reference Files

- [acmg-criteria.md](./references/acmg-criteria.md) — ACMG evidence criteria, per-tool thresholds (PP3/BP4/PM2/BA1)
- [troubleshooting.md](./references/troubleshooting.md) — API errors, model loading failures, VCF parsing issues
