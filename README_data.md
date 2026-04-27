# SNV-judge: Data & Results Branch

This branch (`data-and-results`) contains all experimental data, trained model artifacts, figures, and documents generated during the SNV-judge undergraduate thesis project (Southeast University, 2026).

## Branch Structure

```
data/                              # Raw & processed datasets
  feature_matrix.xlsx              # Final 8-feature matrix (842 variants, BRCA1/2)
  feature_matrix_v4.csv            # Feature matrix v4 (CSV format)
  clinvar_raw.csv                  # Raw ClinVar query results (BRCA1/2 missense)
  genos_scores_cache.csv           # Genos-10B pathogenicity scores (842 variants)
  vep_gnomad_phylop_cache.csv      # VEP annotation: gnomAD AF + phyloP scores
  vep_brca_annotation.csv          # VEP full annotation for BRCA1/2 variants
  generalization_ext_features.csv  # External validation set (TP53/MLH1/MSH2/PTEN)
  generalization_results.csv       # Generalization validation performance summary
  calibration_final_cv_results.csv # Calibration method comparison (5-fold CV)
  model_metrics_final.csv          # Final model performance metrics
  gpn_vep_results.csv              # GPN embedding + VEP comparison results

figures/                           # All experimental figures (PNG + SVG)
  fig_roc_pr_final.*               # ROC and PR curves (final model)
  fig_ablation_shap_final.*        # Ablation study + SHAP feature importance
  fig_calibration_final.*          # Calibration comparison (Platt vs Isotonic)
  fig_generalization_validation.*  # Generalization validation (4-gene panel)
  fig_planc_shap_comparison.*      # Plan C SHAP comparison (case studies)
  ...

planc_reports/                     # Plan C ACMG interpretation reports
  report_case1_TP53_pathogenic_v2.md   # Case 1: TP53 (Pathogenic, P=93.9%)
  report_case2_BRCA2_benign_v2.md      # Case 2: BRCA2 (Likely Benign, P=3.9%)

docs/                              # Thesis and presentation documents
  SNV-judge_毕业论文_周嘉诺.docx    # Full undergraduate thesis (Word)
  SNV_judge_opening_defense_v2.pptx  # Opening defense presentation
  开题报告_周嘉诺_SNV-judge.docx    # Thesis proposal (Word)
```

## Key Results

| Metric | Value |
|--------|-------|
| Dataset | 842 variants (BRCA1/2, ClinVar Dec 2024) |
| Pathogenic / Benign | 289 / 553 |
| AUROC (5-fold CV) | 0.9797 (95% CI [0.9674, 0.9896]) |
| AUPRC | 0.9644 |
| F1 Score | 0.9233 |
| Brier Score | 0.0439 |
| ECE (Platt Scaling) | 0.0257 |

### Generalization Validation

| Gene | n | AUROC | AUPRC | F1 |
|------|---|-------|-------|-----|
| TP53 | 341 | 0.824 | 0.940 | 0.798 |
| MLH1 | 111 | 0.682 | 0.853 | 0.695 |
| MSH2 | 174 | 0.766 | 0.773 | 0.680 |
| PTEN | 370 | 0.600 | 0.662 | 0.714 |

## Citation

Zhou Jianuo. SNV-judge: An Ensemble Learning Framework for Pathogenicity Prediction of Single Nucleotide Variants. Undergraduate Thesis, Southeast University, 2026.
