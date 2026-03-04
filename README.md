# SNV-judge v3: Ensemble SNV Pathogenicity Predictor with Genomic Foundation Models

An interpretable ensemble meta-model that integrates classical pathogenicity scoring tools with **state-of-the-art genomic foundation models** — [Evo2](https://arcinstitute.org/manuscripts/Evo2) (Arc Institute / NVIDIA) and [Genos](https://github.com/zhejianglab/Genos) (Zhejiang Lab) — plus **phyloP evolutionary conservation** to predict the pathogenicity of human missense SNVs.

Trained on **2,000 ClinVar missense variants** across **547 genes** (balanced 1:1 P/B, expert-panel reviewed).

![SHAP Analysis](figures/shap_analysis_v3.png)

---

## What's New in v3

| | v1 | v2 | v3 |
|---|---|---|---|
| Training variants | 842 (BRCA1/2/TP53) | 1,800 (547 genes) | **2,000 (547 genes)** |
| Features | SIFT, PolyPhen, AM, CADD | + Evo2 LLR, Genos Score | + **phyloP conservation** |
| Model | XGBoost | XGBoost + LightGBM Stacking | XGBoost + LightGBM Stacking |
| AUROC | ~0.91 | 0.9373 | **0.9386** |
| Batch VCF | — | — | **Yes (up to 500 SNVs)** |

---

## Features

- **Live annotation**: Fetches SIFT, PolyPhen-2, AlphaMissense, CADD scores via Ensembl VEP API
- **Evo2 scoring**: Zero-shot log-likelihood ratio from Evo2-40B (9.3T token DNA foundation model) [2]
- **Genos scoring**: Human-centric genomic foundation model pathogenicity score [3]
- **Stacking ensemble**: XGBoost + LightGBM with logistic regression meta-learner
- **Interpretability**: Per-variant SHAP feature contribution chart
- **Interactive UI**: Streamlit app with pathogenicity gauge, score bar chart, and SHAP visualization
- **Reproducible**: Full training pipeline included (`train.py`)

---

## Model Performance

5-fold cross-validation on 1,800 ClinVar missense variants (expert panel reviewed, multi-gene).  
Bootstrap 95% confidence intervals (n=200 resamples).

| Model | AUROC [95% CI] | AUPRC [95% CI] |
|---|---|---|
| **v3: XGBoost + Evo2 + Genos + phyloP** | **0.9386 [0.929–0.949]** | **0.9328 [0.918–0.945]** |
| v2: XGBoost + Evo2 + Genos | 0.9373 [0.927–0.947] | 0.9345 [0.921–0.946] |
| AlphaMissense alone | 0.9109 [0.898–0.923] | 0.9393 [0.927–0.948] |
| CADD alone | 0.9039 [0.886–0.916] | 0.9220 [0.903–0.938] |
| Genos alone | 0.6478 [0.619–0.674] | 0.7231 [0.683–0.749] |

> Adding phyloP conservation to v2 improves AUROC by **+0.0013** (pathogenic variants: mean phyloP = 1.82 vs. 0.12 for benign).

### ROC & Precision-Recall Curves

![Model Curves](figures/model_curves_v3.png)

---

## SHAP Feature Importance

AlphaMissense remains the dominant contributor, followed by CADD and PolyPhen-2. **Genos Score**, **Evo2 LLR**, and **phyloP** all contribute meaningfully — phyloP captures evolutionary constraint orthogonal to the sequence-based tools.

![SHAP Analysis](figures/shap_analysis_v3.png)

---

## Repository Structure

```
SNV-judge/
├── app.py                      # Streamlit web application (v3)
├── train.py                    # Full training pipeline (data → model → evaluation)
├── requirements.txt            # Python dependencies
├── xgb_model_v3.pkl            # Trained stacking classifier (v3, 7 features)
├── platt_scaler_v3.pkl         # Platt calibration scaler (v3)
├── train_medians_v3.pkl        # Training set medians (for NaN imputation, v3)
├── feature_cols_v3.pkl         # Feature column names (v3)
├── xgb_model_v2.pkl            # v2 model (6 features, kept for fallback)
├── platt_scaler_v2.pkl         # v2 Platt scaler
├── train_medians_v2.pkl        # v2 medians
├── data/
│   ├── model_metrics_v3.csv    # v3 AUROC/AUPRC with bootstrap 95% CIs
│   └── model_metrics_v2.csv    # v2 metrics (legacy)
└── figures/
    ├── model_curves_v3.png/svg     # ROC + PR curves (v3 vs v2)
    ├── shap_analysis_v3.png/svg    # SHAP beeswarm + bar plots (7 features)
    ├── model_curves_v2.png/svg     # v2 curves (legacy)
    └── shap_analysis_v2.png/svg    # v2 SHAP (legacy)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API keys (required for Evo2 + Genos scoring)

```bash
export EVO2_API_KEY="your-nvidia-nim-api-key"    # https://build.nvidia.com/arc-institute/evo2
export GENOS_API_KEY="your-stomics-api-key"       # https://cloud.stomics.tech
```

> Without API keys, the app falls back to the 4-feature base model (SIFT, PolyPhen, AlphaMissense, CADD).

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### 4. Example variants to try

| Variant | Gene | Expected |
|---|---|---|
| chr17:7674220 C>T | TP53 R175H | Pathogenic |
| chr17:43057062 C>T | BRCA1 R1699W | Pathogenic |
| chr13:32906729 C>A | BRCA2 N372H | Benign |

---

## Retrain the Model

To reproduce the full pipeline from scratch:

```bash
export EVO2_API_KEY="..."
export GENOS_API_KEY="..."
python train.py
```

This will:
1. Download ClinVar `variant_summary.txt.gz` and filter for high-quality missense SNVs (≥2-star, GRCh38)
2. Fetch 101 bp genomic context for each variant via Ensembl REST API
3. Annotate variants via Ensembl VEP REST API (SIFT, PolyPhen-2, AlphaMissense, CADD)
4. Score variants with **Evo2** (zero-shot LLR via NVIDIA NIM) and **Genos** (Stomics cloud API)
5. Train XGBoost + LightGBM stacking ensemble with 5-fold cross-validation
6. Compute SHAP values and generate figures
7. Save all model artefacts (`*_v2.pkl`)

---

## Methods

### Data
- **Source**: ClinVar (accessed March 2026), filtered for missense SNVs with ≥2-star review status
- **Total high-quality variants**: 61,498 (38,237 benign + 23,261 pathogenic) across 2,927 genes
- **Training set**: 2,000 variants (1,000 P + 1,000 B) sampled from 547 genes (max 10 per gene, expert-panel prioritised)
- **Genome build**: GRCh38

### Features
| Feature | Tool | Direction | Coverage |
|---|---|---|---|
| SIFT (inverted) | SIFT4G | Higher = more damaging | 94% |
| PolyPhen-2 score | PolyPhen-2 | Higher = more damaging | 88% |
| AlphaMissense score | Google DeepMind | Higher = more pathogenic | 87% |
| CADD Phred score | CADD v1.7 | Higher = more deleterious | 100% |
| **Evo2 LLR** | Arc Institute / NVIDIA | Negative = more pathogenic | 100% |
| **Genos Score** | Zhejiang Lab | Higher = more pathogenic | 100% |
| **phyloP score** | Ensembl VEP / UCSC | Higher = more conserved | 67% |

Classical scores and phyloP fetched via Ensembl VEP REST API (`Conservation=1`). Missing values imputed with training-set medians.

#### Evo2 Log-Likelihood Ratio
Evo2 is a 40-billion parameter DNA language model trained on 9.3 trillion nucleotide tokens across all domains of life [1]. For each variant, we compute:

```
LLR = log P(continuation | alt_context) − log P(continuation | ref_context)
```

using a 101 bp genomic window centred on the variant. A negative LLR indicates the alternate allele is less likely under the evolutionary prior, suggesting functional disruption.

#### Genos Pathogenicity Score
Genos is a 1.2B–10B parameter human-centric genomic foundation model trained on the human reference genome and population variation data [2]. We query the `variant_predict` endpoint with GRCh38 coordinates to obtain a direct pathogenicity probability.

### Model
- **Algorithm**: Stacking ensemble — XGBoost + LightGBM base learners, logistic regression meta-learner
- **Calibration**: Platt scaling (sigmoid) fitted on out-of-fold predictions
- **Evaluation**: 5-fold cross-validation (stratified by label) to prevent data leakage
- **Hyperparameters**: XGBoost (n_estimators=300, max_depth=4, lr=0.05); LightGBM (n_estimators=300, max_depth=4, lr=0.05)

### Interpretability
- SHAP TreeExplainer computed on out-of-fold held-out samples

---

## Limitations

- Training set is a 2,000-variant sample from ClinVar; performance on rare/novel variants may differ
- Evo2 and Genos API calls add ~2–5 seconds per variant; batch scoring recommended for large VCFs
- Genos standalone AUROC is modest (0.65); it contributes primarily through interaction with classical features
- phyloP coverage is 67% (VEP `conservation` field); missing values are imputed with training median
- phyloP and CADD are partially correlated (both capture conservation); marginal gain from phyloP is modest (+0.0013 AUROC)
- **Not validated for clinical use**

---

## Citation

If you use this project, please cite the underlying tools:

- **[1] Evo2**: Brixi et al., *bioRxiv* 2025. Arc Institute / NVIDIA. https://arcinstitute.org/manuscripts/Evo2
- **[2] Genos**: Zhejiang Lab, 2024. https://github.com/zhejianglab/Genos
- **AlphaMissense**: Cheng et al., *Science* 2023
- **CADD**: Kircher et al., *Nature Genetics* 2014; Rentzsch et al., *Nucleic Acids Research* 2019
- **SIFT**: Ng & Henikoff, *Genome Research* 2001
- **PolyPhen-2**: Adzhubei et al., *Nature Methods* 2010
- **Ensembl VEP**: McLaren et al., *Genome Biology* 2016
- **ClinVar**: Landrum et al., *Nucleic Acids Research* 2016

---

## License

MIT License. Note that individual scoring tools have their own licenses:
- REVEL and ClinPred: non-commercial use only
- AlphaMissense: CC-BY 4.0
- CADD: free for non-commercial use

---

## Author

Junow Chow
