# ACMG/AMP Variant Classification Criteria
## Relevant to SNV-judge v5 Features

Reference: Richards et al., Genetics in Medicine 2015

---

## Evidence Criteria Used by SNV-judge

### Pathogenic Evidence

| Criterion | Strength | SNV-judge Feature | Threshold |
|-----------|----------|-------------------|-----------|
| **PP3** | Supporting | SIFT, PolyPhen-2, AlphaMissense, CADD, Evo2, Genos, phyloP | Multiple tools predict damaging |
| **PM2** | Moderate | gnomAD log-AF | AF < 1×10⁻⁴ (absent or very rare) |

### Benign Evidence

| Criterion | Strength | SNV-judge Feature | Threshold |
|-----------|----------|-------------------|-----------|
| **BP4** | Supporting | SIFT, PolyPhen-2, AlphaMissense, CADD | Multiple tools predict benign |
| **BA1** | Stand-alone | gnomAD log-AF | AF > 5% in any population |

---

## Tool-Specific Thresholds

### SIFT (inverted in SNV-judge)
- Raw SIFT score: 0 = most damaging, 1 = tolerated
- SNV-judge uses `1 - SIFT` so higher = more damaging
- **PP3**: SIFT_inv > 0.5 (raw SIFT < 0.5, "deleterious")
- **BP4**: SIFT_inv < 0.5 (raw SIFT ≥ 0.5, "tolerated")

### PolyPhen-2
- Range: 0–1 (higher = more damaging)
- **PP3**: > 0.85 ("probably_damaging")
- **BP4**: < 0.15 ("benign")

### AlphaMissense (Google DeepMind)
- Range: 0–1 (higher = more pathogenic)
- **PP3**: > 0.564 ("likely_pathogenic")
- **BP4**: < 0.340 ("likely_benign")
- 0.340–0.564: "ambiguous"

### CADD Phred
- Higher = more deleterious
- **PP3**: > 20 (top 1% most deleterious variants)
- **PP3 strong**: > 30 (top 0.1%)

### Evo2-40B LLR
- Negative = alt allele less likely under evolutionary prior (damaging)
- **PP3 supporting**: LLR < -0.3
- **PP3 moderate**: LLR < -1.0

### Genos-10B
- Range: 0–1 (higher = more pathogenic)
- **PP3**: > 0.7

### phyloP
- Positive = conserved, negative = accelerated evolution
- **PP3**: > 2.0 (highly conserved)
- **BP4**: < -0.5 (evolutionarily accelerated)

### gnomAD v4 Allele Frequency
- SNV-judge stores as log10(AF + 1e-8)
- **BA1**: AF > 0.05 (log-AF > -1.30)
- **PM2**: AF < 1×10⁻⁴ (log-AF < -4.0) or absent (log-AF = -8.0)

---

## ACMG 5-Tier Classification (SNV-judge thresholds)

| Calibrated Probability | Classification | ACMG Code |
|------------------------|---------------|-----------|
| ≥ 0.90 | Pathogenic (P) | High confidence |
| 0.70–0.90 | Likely Pathogenic (LP) | Moderate confidence |
| 0.40–0.70 | VUS | Uncertain significance |
| 0.20–0.40 | Likely Benign (LB) | Moderate confidence |
| < 0.20 | Benign (B) | High confidence |

---

## Important Notes

1. SNV-judge provides **computational evidence only** (PP3/BP4 category)
2. Clinical classification requires integration with functional data, segregation, de novo status, etc.
3. The model is **not validated for clinical use** — research only
4. gnomAD AF alone achieves AUROC=0.96 on variants with data, but median imputation reduces full-dataset AUC to 0.74; XGBoost learns to use the missingness pattern itself as a PM2-like signal
