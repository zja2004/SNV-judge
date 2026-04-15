# SNV-judge v5.2

> 🌐 Language: **English** | [中文](README_zh.md)

An interpretable ensemble meta-model for human missense SNV pathogenicity prediction. Integrates classical scoring tools with genomic foundation models, evolutionary conservation, and population allele frequency — with a universal LLM-powered clinical report generator.

Trained on **2,000 ClinVar missense variants** across **547 genes** (1:1 P/B, expert-panel reviewed, GRCh38).

![SHAP Analysis](figures/shap_analysis_v4.png)

---

## Key Numbers

| Metric | Value |
|--------|-------|
| 5-fold CV AUROC | **0.9679** [0.959–0.974] |
| 5-fold CV AUPRC | **0.9643** [0.953–0.972] |
| Holdout AUROC (n=300) | **0.9547** |
| Holdout F1 | **0.8946** |
| Brier score | **0.0685** (Isotonic calibration) |
| Training variants | 2,000 (547 genes) |
| Features | 8 |

---

## What's New

### v5.2
- **`skill/scripts/predict.py`** — offline prediction CLI (1,163 lines, 18 functions): VEP annotation, gnomAD AF, ClinVar lookup, SHAP waterfall, batch VCF scoring, `print_shap_summary()` terminal output
- **Universal LLM backend** — AI Clinical Report now supports any OpenAI-compatible provider (Alibaba DashScope, Moonshot, OpenAI, DeepSeek, etc.); in-app provider selector + auto model-list fetch
- **Genos local embedding mode** — use Genos-10B without a Stomics Cloud API key via a local ngrok endpoint (cosine distance of REF/ALT embeddings)
- **LOGO-CV figure** in Model Info panel — generalization chart across 16 disease gene families

### v5.0
- Isotonic Regression calibration (Brier 0.0685 vs Platt 0.0743, −7.8%)
- ACMG 5-tier badge (P / LP / VUS / LB / B) with confidence level
- Session-level variant history with side-by-side SHAP comparison
- Reliability diagram (calibration curve) in Model Info panel
- AI report template selector: Chinese / English / Summary

### Version history

| | v1 | v2 | v3 | v4 | v5 | **v5.2** |
|---|---|---|---|---|---|---|
| Training variants | 842 | 1,800 | 2,000 | 2,000 | 2,000 | **2,000** |
| Features | 4 | 6 | 7 | 8 | 8 | **8** |
| Calibration | — | Platt | Platt | Platt | Isotonic | **Isotonic** |
| AUROC (CV) | ~0.91 | 0.9373 | 0.9488 | 0.9664 | 0.9679 | **0.9679** |
| Offline CLI | — | — | — | — | — | **✓** |
| Universal LLM | — | — | — | — | — | **✓** |

---

## Features

- **Live VEP annotation** — SIFT, PolyPhen-2, AlphaMissense, CADD, phyloP via Ensembl VEP REST API
- **Evo2-40B scoring** — zero-shot log-likelihood ratio (9.3T token DNA foundation model) [1]
- **Genos-10B scoring** — human-centric genomic foundation model pathogenicity score [2]; local embedding mode supported (v5.2)
- **gnomAD v4 AF** — population allele frequency (ACMG BA1/PM2 signal)
- **Stacking ensemble** — XGBoost + LightGBM base learners, logistic regression meta-learner
- **Isotonic calibration** — better probability calibration than Platt scaling
- **ACMG 5-tier badge** — automatic P/LP/VUS/LB/B classification
- **SHAP interpretability** — per-variant feature contribution chart + `print_shap_summary()` CLI output
- **🤖 AI Clinical Report** — any OpenAI-compatible LLM synthesizes all tool outputs into a structured ACMG-style report (Chinese / English / Summary); in-app provider/model selector with auto model-list fetch
- **Offline CLI** — `skill/scripts/predict.py` for batch VCF scoring without the Streamlit UI
- **Streamlit UI** — pathogenicity gauge, score bars, SHAP chart, variant history, AI report tab

---

## Model Performance

### Cross-validation (5-fold, n=2,000)

| Model | AUROC [95% CI] | AUPRC [95% CI] | Brier |
|---|---|---|---|
| **v5: Isotonic calibration** | **0.9679 [0.959–0.974]** | **0.9643 [0.953–0.972]** | **0.0685** |
| v4: Platt calibration | 0.9664 [0.958–0.972] | 0.9671 [0.956–0.973] | 0.0743 |
| v3: + phyloP | 0.9488 [0.938–0.958] | 0.9447 [0.933–0.955] | — |
| v2: + Evo2 + Genos | 0.9373 [0.927–0.947] | 0.9345 [0.921–0.946] | — |
| AlphaMissense alone | 0.9109 [0.898–0.923] | 0.9393 [0.927–0.948] | — |
| CADD alone | 0.9039 [0.886–0.916] | 0.9220 [0.903–0.938] | — |
| Genos alone | 0.6478 [0.619–0.674] | 0.7231 [0.683–0.749] | — |

### Independent holdout (n=300, 150P/150B)

| Split | AUROC | AUPRC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|---|
| 5-fold CV OOF | 0.9679 | 0.9643 | 0.9022 | — | — |
| **Holdout** | **0.9547** | **0.9527** | **0.8946** | **0.9333** | **0.8467** |

![Calibration Comparison](figures/fig_calibration_comparison.png)
![Holdout Validation](figures/fig_validation_holdout.png)

### ROC & PR Curves

![Model Curves](figures/fig1_roc_comparison.png)

### Ablation Study

![Ablation](figures/fig2_ablation.png)

### Generalization — Leave-One-Gene-Out CV (LOGO-CV)

Model trained on all genes except one, evaluated on the held-out gene (n ≈ 20 per gene, 10P/10B).

![LOGO-CV](figures/fig_logo_cv.png)

| Gene | Disease | AUROC | F1 |
|------|---------|-------|----|
| BRCA1, BRCA2 | Hereditary breast/ovarian cancer | 1.000 | 0.952 / 0.909 |
| TP53, FBN1, LDLR, GAA, USH2A, RUNX1 | Tumor suppressor / Connective / Metabolic / Retinal / Hematologic | 1.000 | 0.91–1.00 |
| SOS1, RAF1, MSH2, MSH6 | RASopathy / Lynch MMR | 0.980–0.991 | 0.857–0.952 |
| RYR1, MECP2 | Musculoskeletal / Neurodevelopmental | 0.930–0.955 | 0.750–0.857 |
| MLH1 | Lynch MMR | 0.873 | 0.870 |
| **MYH7** | **Cardiomyopathy** | **0.790** | **0.800** |

**Non-BRCA mean AUROC = 0.9642** (14 genes). MYH7 is the weakest gene — see [Limitations](#limitations).

> Each gene has only ~20 variants in the test set; AUROC estimates carry wide CIs (~±0.10). AUROC = 1.00 for several genes reflects small-sample perfect separation.

---

## AI Clinical Report

Supports **any OpenAI-compatible LLM** — configure in-app without restarting:

```
Tool outputs (VEP + Evo2 + Genos + gnomAD + SHAP)
        ↓
  kimi_report.py — evidence formatting
        ↓
  Your chosen LLM (Qwen / Kimi / GPT-4o / DeepSeek / ...)
        ↓
  Structured ACMG-style report (streaming Markdown)
```

**Supported providers (built-in presets):**

| Provider | Base URL |
|----------|----------|
| Moonshot (Kimi) | `https://api.moonshot.cn/v1` |
| Alibaba DashScope | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| OpenAI | `https://api.openai.com/v1` |
| DeepSeek | `https://api.deepseek.com/v1` |
| Custom | any OpenAI-compatible endpoint |

**In-app setup**: AI Report tab → ⚙️ LLM Settings → select provider → enter API key → click "Get Model List" → select model → generate report.

**Report templates**: Chinese clinical report / English clinical report / Brief summary

---

## Future Vision: Plan C — Multi-Agent System

```
User natural language query
        ↓
  Planner Agent (LangGraph)
  ├── VEP Agent      → SIFT · PP2 · AM · CADD · phyloP
  ├── Evo2 Agent     → NVIDIA NIM LLR scoring
  ├── Genos Agent    → Cloud / local embedding score
  ├── gnomAD Agent   → GraphQL population frequency
  ├── SNV-judge Agent → Ensemble prediction + SHAP
  └── Report Agent   → ACMG-structured clinical report
        ↓
  Multi-turn dialogue · Batch VCF · Long-term memory
```

![Plan C Architecture](figures/figB3_plan_c_vision.png)

---

## Repository Structure

```
SNV-judge/
├── app.py                          # Streamlit web app (v5.2)
├── kimi_report.py                  # Universal LLM report backend
├── train.py                        # Full training pipeline
├── requirements.txt
│
├── xgb_model_v5.pkl                # Stacking classifier (v5)
├── platt_scaler_v5.pkl             # Isotonic Regression calibrator (v5)
├── train_medians_v5.pkl            # Training-set medians (NaN imputation)
├── feature_cols_v5.pkl             # Feature column names
│
├── data/
│   ├── feature_matrix_v4.csv/xlsx  # 2,000-variant feature matrix
│   ├── calibration_metrics_v5.csv  # Calibration comparison
│   ├── model_metrics_v5.csv        # CV + holdout metrics
│   ├── scoring_ckpt.pkl            # Pre-computed Evo2 LLR + Genos scores
│   ├── vep_scores.pkl              # Pre-computed VEP scores
│   ├── phylop_cache.pkl            # Pre-computed phyloP scores
│   └── gnomad_af_cache.pkl         # Pre-computed gnomAD v4 AF
│
├── skill/                          # Offline prediction package
│   ├── SKILL.md                    # API docs & quick-start
│   ├── scripts/
│   │   ├── predict.py              # Offline CLI (v5.2, 1,163 lines, 18 functions)
│   │   └── generalization_eval.py  # LOGO-CV evaluation script
│   └── references/
│       ├── acmg-criteria.md
│       └── troubleshooting.md
│
└── figures/
    ├── fig_logo_cv.png/svg         # LOGO-CV generalization (16 genes)
    ├── fig_calibration_comparison.png/svg
    ├── fig_validation_holdout.png/svg
    ├── fig1_roc_comparison.png/svg
    ├── fig2_ablation.png/svg
    ├── figB3_plan_c_vision.png/svg
    └── shap_analysis_v4.png/svg
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/zja2004/SNV-judge.git
cd SNV-judge
pip install -r requirements.txt
```

### 2. API keys (all optional)

```bash
# Scoring models
export EVO2_API_KEY="nvapi-..."      # NVIDIA NIM — enables Evo2 scoring
export GENOS_API_KEY="sk-..."        # Stomics Cloud — enables Genos scoring

# LLM for AI Clinical Report — any OpenAI-compatible provider
export LLM_API_KEY="sk-..."          # your provider's API key
export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export LLM_MODEL="qwen-plus"
```

> **No keys needed to run**: Evo2/Genos fall back to training-set medians; LLM report tab is hidden; all other features (VEP, gnomAD, SHAP, ACMG badge) work normally.  
> All LLM settings can be configured **directly in the app** — no restart needed.

### 3. Run the app

```bash
streamlit run app.py
# → http://localhost:8501
```

### 4. Offline CLI

```bash
# Single variant
python skill/scripts/predict.py 17 7674220 C T /path/to/SNV-judge

# With local Genos embedding (no API key needed)
python skill/scripts/predict.py 17 7674220 C T /path/to/SNV-judge \
    --genos-url https://xxx.ngrok-free.dev

# Batch VCF
python skill/scripts/predict.py --vcf variants.vcf /path/to/SNV-judge \
    --output results.csv
```

### 5. Python API

```python
from skill.scripts.predict import load_model_artifacts, predict_variant, print_shap_summary

artifacts = load_model_artifacts("/path/to/SNV-judge")
result = predict_variant("17", 7674220, "C", "T", artifacts=artifacts)

print(f"Pathogenicity: {result['prob_pathogenic']:.1%}")
print(f"ACMG tier:     {result['acmg_tier']}")
print_shap_summary(result)
```

### 6. Example variants

| Variant | Gene | Expected |
|---------|------|----------|
| chr17:7674220 C>T | TP53 R175H | Pathogenic |
| chr17:43057062 C>T | BRCA1 R1699W | Pathogenic |
| chr13:32906729 C>A | BRCA2 N372H | Benign |

---

## Pre-computed Scores

All scores used to train v5 are in `data/` — no API keys needed to retrain:

| File | Contents | Coverage |
|------|----------|----------|
| `scoring_ckpt.pkl` | Evo2-40B LLR + Genos-10B scores | 1,677/2,000 (83.9%) |
| `vep_scores.pkl` | SIFT · PolyPhen-2 · AlphaMissense · CADD | 95–100% |
| `phylop_cache.pkl` | phyloP conservation scores | 1,922/2,000 (96.1%) |
| `gnomad_af_cache.pkl` | gnomAD v4 allele frequencies | 2,000 entries |
| `feature_matrix_v4.xlsx` | Complete feature matrix | 2,000 × 22 |

```bash
python train.py --use-cache   # retrain without any API keys
```

**Data provenance** (all scores generated March 2026):

| Score | Source | Version |
|-------|--------|---------|
| Evo2 LLR | NVIDIA NIM `evo2-40b` | `health.api.nvidia.com` |
| Genos Score | Stomics Cloud `variant_predict` | `cloud.stomics.tech` |
| SIFT / PolyPhen-2 / AlphaMissense / CADD / phyloP | Ensembl VEP REST | GRCh38 / e113 |
| gnomAD AF | gnomAD GraphQL | gnomAD r4 |
| ClinVar variants | ClinVar FTP | March 2026 |

---

## Methods

### Data
- **Source**: ClinVar (March 2026), missense SNVs with ≥2-star review status
- **Training set**: 2,000 variants (1,000P + 1,000B), 547 genes, max 10 per gene
- **Genome build**: GRCh38

### Features

| Feature | Tool | Direction | Coverage |
|---------|------|-----------|----------|
| SIFT (inverted) | SIFT4G | ↑ more damaging | 94% |
| PolyPhen-2 | PolyPhen-2 | ↑ more damaging | 88% |
| AlphaMissense | Google DeepMind | ↑ more pathogenic | 87% |
| CADD Phred | CADD v1.7 | ↑ more deleterious | 100% |
| Evo2 LLR | Arc Institute / NVIDIA [1] | ↓ more pathogenic | 84% |
| Genos Score | Zhejiang Lab [2] | ↑ more pathogenic | 84% |
| phyloP | Ensembl VEP / UCSC | ↑ more conserved | 96% |
| gnomAD log-AF | gnomAD v4 | ↓ rarer = more pathogenic | 37% |

Missing values imputed with training-set medians.

### Model
- **Algorithm**: XGBoost + LightGBM stacking, logistic regression meta-learner
- **Calibration**: Isotonic Regression on out-of-fold predictions
- **Evaluation**: 5-fold stratified CV
- **Hyperparameters**: XGBoost/LightGBM (n_estimators=300, max_depth=4, lr=0.05)

### Evo2 LLR
Evo2 is a 40B-parameter DNA language model trained on 9.3T nucleotide tokens [1]. LLR = log P(alt context) − log P(ref context) over a 101 bp window. Negative LLR → alt allele less likely under evolutionary prior → functional disruption.

### Genos Score
Genos is a 1.2B–10B human-centric genomic foundation model [2]. We query `variant_predict` with GRCh38 coordinates for a direct pathogenicity probability. In v5.2, a local embedding mode is also supported: cosine distance between REF/ALT sequence embeddings from the `/extract` endpoint.

---

## Limitations

- Training set is a 2,000-variant sample from ClinVar; performance on rare/novel variants may differ
- Evo2/Genos API calls add ~2–5 s per variant; use batch mode for large VCFs
- Genos standalone AUROC is modest (0.65); it contributes mainly through feature interactions
- gnomAD AF coverage is 37% in training (many ClinVar pathogenic variants absent from gnomAD); XGBoost learns to use the missingness pattern as a signal
- **MYH7 (cardiomyopathy) is a known weak gene** — LOGO-CV AUROC = 0.79, lowest of 16 tested genes. Three reasons:
  1. **Gain-of-function mechanism**: MYH7 pathogenic variants act via dominant gain-of-function (altered myosin ATPase kinetics), not loss-of-function. SIFT/PolyPhen-2 are calibrated on LoF variants and systematically underestimate MYH7 pathogenicity.
  2. **Intermediate scores**: MYH7 pathogenic variants (myosin head domain, residues 167–931) often have intermediate SIFT (0.01–0.1) and PolyPhen-2 (0.5–0.85) scores overlapping with benign variants.
  3. **Small test set**: n=20 per gene → AUROC CI ≈ ±0.10.
  - **Planned (v6)**: protein structure features (distance to ATP-binding site, AlphaFold2 inter-domain contact energy).
- **Not validated for clinical use**

---

## Citation

- **[1] Evo2**: Brixi et al., *bioRxiv* 2025. https://arcinstitute.org/manuscripts/Evo2
- **[2] Genos**: Zhejiang Lab, 2024. https://github.com/zhejianglab/Genos
- **AlphaMissense**: Cheng et al., *Science* 2023
- **CADD**: Kircher et al., *Nature Genetics* 2014; Rentzsch et al., *NAR* 2019
- **SIFT**: Ng & Henikoff, *Genome Research* 2001
- **PolyPhen-2**: Adzhubei et al., *Nature Methods* 2010
- **Ensembl VEP**: McLaren et al., *Genome Biology* 2016
- **gnomAD v4**: Karczewski et al., *Nature* 2020; Chen et al., *bioRxiv* 2023
- **ClinVar**: Landrum et al., *NAR* 2016

---

## License

MIT. Individual tool licenses apply: AlphaMissense (CC-BY 4.0), CADD (non-commercial free).

## Author

Junow Chow
