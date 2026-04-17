# ACMG Evidence Criteria — SNV-judge v5.2

Reference: Richards et al., Genetics in Medicine 2015

## What SNV-judge covers

SNV-judge provides **computational evidence only** — equivalent to PP3 (pathogenic supporting) and BP4 (benign supporting) in the ACMG framework. It does NOT evaluate functional studies, segregation, de novo status, or allelic data.

## Per-feature thresholds

| Feature | PP3 (pathogenic) | BP4 (benign) | Notes |
|---|---|---|---|
| SIFT (inv) | > 0.5 | < 0.5 | Raw SIFT inverted; higher = more damaging |
| PolyPhen-2 | > 0.85 | < 0.15 | "probably_damaging" vs "benign" |
| AlphaMissense | > 0.564 | < 0.340 | 0.340–0.564 = ambiguous |
| CADD Phred | > 20 (top 1%) | — | > 30 = top 0.1% |
| Evo2-40B LLR | < -0.3 (supporting) | — | < -1.0 = moderate |
| Genos-10B | > 0.7 | — | Range 0–1 |
| phyloP | > 2.0 (conserved) | < -0.5 (accelerated) | UCSC 100-way |
| gnomAD log-AF | < -4.0 → PM2 | > -1.30 → BA1 | Absent (−8.0) = strongest PM2 |

## ACMG 5-tier output mapping

| cal_prob | ACMG class | Evidence strength |
|---|---|---|
| ≥ 0.90 | Pathogenic | PP3 strong (multiple tools concordant) |
| 0.70–0.90 | Likely Pathogenic | PP3 moderate |
| 0.40–0.70 | VUS | Conflicting or insufficient evidence |
| 0.20–0.40 | Likely Benign | BP4 moderate |
| < 0.20 | Benign | BP4 strong |

## gnomAD AF signal

gnomAD AF is the single strongest feature (SHAP rank #1 in most variants). Key points:
- 37% of training variants have gnomAD data; the rest are imputed at log-AF = −4.48
- XGBoost learns the **missingness pattern** itself as a PM2-like signal (absent from gnomAD → likely pathogenic)
- BA1 (stand-alone benign): AF > 5% in any population → log-AF > −1.30

## Important caveats for agent responses

1. Always state that SNV-judge is **not validated for clinical use**
2. For VUS (0.40–0.70), explicitly note that additional evidence is needed
3. For MYH7 variants, add: "MYH7 is a known weak gene for this model (AUROC 0.79); interpret with caution"
4. If `genos_source = "median_imputed"`, note that Evo2/Genos scores were unavailable and imputed
