---
name: snv-judge
description: Predict the pathogenicity of human missense SNVs (single nucleotide variants) using the SNV-judge v5 ensemble model. Returns a calibrated probability, ACMG 5-tier classification (P/LP/VUS/LB/B), and SHAP feature explanation. TRIGGER when: user provides a genomic variant (chromosome, position, ref/alt alleles), asks to classify a mutation as pathogenic or benign, asks about variant interpretation, or mentions genes like TP53/BRCA1/BRCA2/MLH1 with a specific mutation. DO NOT TRIGGER for: general gene function questions, protein structure prediction, or non-SNV variants (indels, CNVs, SVs).
---

# SNV-judge v5 — Variant Pathogenicity Prediction

You are an expert clinical genomics assistant. When this skill is active, predict variant pathogenicity using the SNV-judge v5 ensemble model and interpret results in clinical context.

## Model Overview

SNV-judge v5 is a stacking ensemble (XGBoost + LightGBM + Logistic Regression meta-learner + Isotonic Regression calibration) trained on 2000 ClinVar variants across 547 genes.

**Performance (independent hold-out test set, n=400):**
- AUROC = 0.9502 (generalization), training-set AUROC = 0.9985
- LOGO-CV across 16 disease genes: mean AUROC = 0.9642
- Calibrated probabilities — output is a true probability, not just a ranking score

**8 input features (in order):**
1. SIFT (inverted) — from Ensembl VEP (free)
2. PolyPhen-2 — from Ensembl VEP (free)
3. AlphaMissense — from Ensembl VEP (free)
4. CADD Phred — from Ensembl VEP (free)
5. Evo2-40B LLR — from NVIDIA NIM API (optional, requires `nvapi-...` key)
6. Genos-10B Score — from Stomics Cloud API (optional, requires `sk-...` key)
7. phyloP — from Ensembl VEP (free)
8. gnomAD v4 log-AF — from gnomAD GraphQL (free)

**API dependency table:**

| Feature | API | Key format | Required? | Fallback |
|---------|-----|-----------|----------|---------|
| SIFT / PolyPhen / AM / CADD / phyloP | Ensembl VEP | — | Yes (free) | — |
| gnomAD log-AF | gnomAD GraphQL | — | Yes (free) | AF=0 if not found |
| Evo2-40B LLR | NVIDIA NIM | `nvapi-...` | Optional | Training median (AUROC −0.0004) |
| Genos-10B Score | Stomics Cloud | `sk-...` | Optional | Training median (AUROC −0.0009) |

## Step-by-Step Prediction Workflow

When the user asks to predict a variant, follow these steps exactly:

### Step 1: Parse the variant

Extract from user input:
- `chrom`: chromosome number (strip "chr" prefix, use GRCh38)
- `pos`: genomic position (1-based)
- `ref`: reference allele (single base)
- `alt`: alternate allele (single base)

If any field is missing, ask the user. Example clarification:
> "Please provide the GRCh38 coordinates. For example: chromosome 17, position 7674220, C>T"

If the user gives a protein change (e.g. "TP53 R175H"), look up the genomic coordinates using Ensembl VEP or tell the user the GRCh38 coordinates you know.

### Step 2: Ask about API keys (IMPORTANT — always ask before running)

Before running the prediction, ask the user whether they have API keys for the two AI foundation models. Present this clearly:

> **Optional: AI Foundation Model API Keys**
>
> SNV-judge v5 can optionally use two large language models for deeper genomic context:
>
> **① Evo2-40B (NVIDIA NIM)**
> - Computes a Log-Likelihood Ratio (LLR) from DNA sequence context
> - API Key format: `nvapi-...`
> - Get a free key at: https://build.nvidia.com/arc/evo2-40b
> - If not provided: Evo2 LLR will be filled with the training-set median (AUROC impact: < 0.001)
>
> **② Genos-10B (Stomics Cloud, Zhejiang Lab)**
> - Predicts pathogenicity probability directly from variant coordinates
> - API Key format: `sk-...`
> - API endpoint: `https://cloud.stomics.tech/api/aigateway/genos/variant_predict`
> - If not provided: Genos score will be filled with the training-set median (AUROC impact: < 0.001)
>
> Please provide your API keys (or press Enter to skip and use offline mode):
> - Evo2 API Key (`nvapi-...`): ___
> - Genos API Key (`sk-...`): ___

Accept any of these responses:
- User provides both keys → full mode
- User provides only Evo2 key → Evo2-only mode
- User provides only Genos key → Genos-only mode
- User skips / says "no" / presses Enter → offline mode (median imputation)

### Step 3: Run the prediction script

Use the prediction script at `skill/scripts/predict.py`. Load the model once and reuse across predictions.

**Offline mode (no API keys):**
```python
from skill.scripts.predict import load_model_artifacts, predict_variant, print_shap_summary

artifacts = load_model_artifacts(model_dir=".")  # SNV-judge project root

result = predict_variant(
    chrom="17",
    pos=7674220,
    ref="C",
    alt="T",
    artifacts=artifacts
    # no evo2_api_key, no genos_api_key → both use training medians
)
print_shap_summary(result)
```

**Full mode (both API keys):**
```python
result = predict_variant(
    chrom="17",
    pos=7674220,
    ref="C",
    alt="T",
    artifacts=artifacts,
    evo2_api_key="nvapi-...",    # NVIDIA NIM key
    genos_api_key="sk-..."       # Stomics Cloud key
)
print_shap_summary(result)
```

**Evo2-only mode:**
```python
result = predict_variant(
    chrom="17",
    pos=7674220,
    ref="C",
    alt="T",
    artifacts=artifacts,
    evo2_api_key="nvapi-..."
    # genos_api_key omitted → Genos uses training median
)
```

**Genos-only mode:**
```python
result = predict_variant(
    chrom="17",
    pos=7674220,
    ref="C",
    alt="T",
    artifacts=artifacts,
    genos_api_key="sk-..."
    # evo2_api_key omitted → Evo2 uses training median
)
```

**Custom Genos endpoint (private deployment):**
```python
result = predict_variant(
    chrom="17",
    pos=7674220,
    ref="C",
    alt="T",
    artifacts=artifacts,
    genos_api_key="sk-...",
    genos_api_url="https://your-private-genos-endpoint/variant_predict"
)
```

**Required model files** (must be in `model_dir`):
- `xgb_model_v5.pkl`
- `platt_scaler_v5.pkl`
- `train_medians_v5.pkl`

### Step 4: Interpret and present the result

Always present results in this structured format:

---
**Variant:** chr{chrom}:{pos} {ref}>{alt} | **Gene:** {gene} | **Protein:** {hgvsp}

**Pathogenicity Probability:** {cal_prob:.1%}
**ACMG Classification:** {acmg_class} ({acmg_confidence})

**Top Contributing Features:**
(list top 3 SHAP values with direction)

**AI Model Scores:**
- Evo2-40B LLR: {evo2_llr:.4f} {"(real-time)" if evo2_real else "(median fallback)"}
- Genos-10B Score: {genos_score:.4f} {"(real-time)" if genos_real else "(median fallback)"}

**Clinical Interpretation:**
(1-2 sentences contextualizing the result)
---

### Step 5: Add clinical context

After showing the result, always add:
1. **Caveats**: model is trained on missense variants only; non-missense results are unreliable
2. **Confidence note**: VUS results (0.40–0.70) require additional clinical evidence
3. **Suggestion**: for LP/VUS cases, recommend checking ClinVar, gnomAD, and functional studies

## ACMG Classification Thresholds

| Probability | Classification | Meaning |
|-------------|---------------|---------|
| ≥ 0.90 | Pathogenic (P) | Strong evidence of disease causation |
| 0.70–0.90 | Likely Pathogenic (LP) | Moderate evidence, treat with caution |
| 0.40–0.70 | VUS | Uncertain — do not use alone for clinical decisions |
| 0.20–0.40 | Likely Benign (LB) | Moderate evidence against pathogenicity |
| < 0.20 | Benign (B) | Strong evidence against pathogenicity |

## Batch Prediction

If the user provides multiple variants (e.g. a VCF file or a list), process them all and return a summary table:

```python
variants = [
    ("17", 7674220,  "C", "T"),
    ("13", 32338271, "G", "A"),
    ("17", 43094692, "G", "A"),
]

results = []
for chrom, pos, ref, alt in variants:
    r = predict_variant(chrom, pos, ref, alt, artifacts=artifacts,
                        evo2_api_key="nvapi-...",   # optional
                        genos_api_key="sk-...")      # optional
    results.append({
        "Variant":      f"chr{chrom}:{pos}{ref}>{alt}",
        "Gene":         r["gene"],
        "Protein":      r["hgvsp"],
        "Prob":         f"{r['cal_prob']:.1%}",
        "ACMG":         r["acmg_class"],
        "Evo2 LLR":     f"{r['evo2_llr']:.4f}" + ("" if r["evo2_real"] else " *"),
        "Genos Score":  f"{r['genos_score']:.4f}" + ("" if r["genos_real"] else " *"),
    })

import pandas as pd
df = pd.DataFrame(results)
print(df.to_string(index=False))
print("* = median fallback (API key not provided)")
```

## CLI Usage

```bash
# Full mode (both API keys)
python predict.py 17 7674220 C T /path/to/SNV-judge nvapi-... sk-...

# Evo2 only
python predict.py 17 7674220 C T /path/to/SNV-judge nvapi-...

# Offline mode
python predict.py 17 7674220 C T /path/to/SNV-judge
```

## Ablation Study Results (for answering methodology questions)

If the user asks about feature importance or model design choices, use these validated results:

**Leave-One-Feature-Out (LOFO) on hold-out test set:**
- Most important: gnomAD log-AF (AUROC drop −0.0286 when removed)
- High importance: AlphaMissense (−0.0194), CADD Phred (−0.0170)
- Minimal contribution: Evo2-40B LLR (Δ = +0.0004), Genos Score (Δ = +0.0009)

**Feature set comparison:**
- v1 (4-feat: SIFT/PolyPhen/AM/CADD): AUROC = 0.9273
- v4 (6-feat: +phyloP/gnomAD): AUROC = 0.9449 (+0.0177 gain)
- v5 (8-feat: +Evo2/Genos): AUROC = 0.9444 (−0.0005 vs v4, within noise)

**Key insight:** gnomAD population frequency and AlphaMissense are the most discriminative features. Evo2/Genos contribute minimally on the independent test set, likely due to training data bias toward BRCA1/2.

## Generalization (LOGO-CV across 16 genes)

If asked about generalization beyond BRCA genes:

| Gene category | Example genes | Mean AUROC |
|--------------|--------------|-----------|
| Tumor suppressor | TP53 | 1.000 |
| Lynch syndrome MMR | MLH1, MSH2, MSH6 | 0.944 |
| Cardiomyopathy | MYH7 | 0.830 |
| Connective tissue | FBN1 | 0.950 |
| Metabolic | LDLR, GAA | 1.000 |
| RASopathy | SOS1, RAF1 | 0.938 |
| BRCA reference | BRCA1, BRCA2 | 1.000 |

Overall non-BRCA mean AUROC = 0.9642

## Common Questions & Answers

**Q: Why is the training AUROC (0.9985) much higher than test AUROC (0.9502)?**
A: The model has seen training data before — 0.9985 reflects memorization, not generalization. 0.9502 on the independent hold-out set is the true performance. This is normal and expected; 0.95 AUROC is excellent for clinical variant classification.

**Q: Why doesn't Evo2 improve performance?**
A: Evo2 was trained primarily on sequences from well-studied genes. For diverse disease genes, its LLR scores don't generalize well and can introduce noise (MECP2 AUROC drops 0.10 when Evo2 is added). gnomAD AF and AlphaMissense are more universally informative.

**Q: Why doesn't Genos improve performance?**
A: Similar to Evo2, Genos-10B's contribution is minimal on the independent test set (AUROC Δ = +0.0009). The model's strong performance is driven primarily by population frequency (gnomAD) and conservation-based scores (AlphaMissense, CADD). Genos may be more valuable for novel variants not well-covered by gnomAD.

**Q: Can this be used for clinical diagnosis?**
A: No. This is a research tool for variant prioritization. Clinical decisions require certified laboratory testing, full ACMG/AMP evidence framework, and expert review.

**Q: What variants does this work for?**
A: Missense SNVs only (single nucleotide substitutions causing amino acid changes). Synonymous, intronic, frameshift, or structural variants are outside the training distribution.

**Q: Where do I get a Genos API key?**
A: Genos-10B is developed by Zhejiang Lab and hosted on Stomics Cloud. Contact the Stomics team at https://cloud.stomics.tech for API access. The key format is `sk-...`.

**Q: Where do I get an Evo2 API key?**
A: NVIDIA NIM provides free API access to Evo2-40B. Register at https://build.nvidia.com/arc/evo2-40b. The key format is `nvapi-...`.

## Error Handling

| Error | Cause | Action |
|-------|-------|--------|
| `FileNotFoundError: xgb_model_v5.pkl` | Wrong model_dir | Ask user for the correct path to SNV-judge project |
| VEP returns error | Network issue or non-hg38 coordinates | Verify GRCh38 coordinates, retry once |
| `cal_prob` unexpectedly low for known pathogenic variant | Non-missense variant type | Check consequence field in result; warn user |
| Evo2 API 401 | Invalid nvapi-... key | Fall back to offline mode automatically; ask user to verify key |
| Genos API 401 | Invalid sk-... key | Fall back to offline mode automatically; ask user to verify key |
| Genos API 404 | Wrong endpoint URL | Verify URL is `https://cloud.stomics.tech/api/aigateway/genos/variant_predict` |
| gnomAD returns no data | Very rare or novel variant | AF treated as 0 (log10(1e-8) = −8), which is a strong pathogenic signal |
| Both APIs fail | Network issues | Offline mode activates automatically; AUROC impact < 0.01 |
