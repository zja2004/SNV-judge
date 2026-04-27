# SNV-judge Plan C — ACMG智能解读报告（v2）

| 字段 | 值 |
|------|----|
| 变异 | BRCA2 chr13:32340561 T>C (GRCh38) |
| SNV-judge致病概率 | 3.9% |
| ClinVar标注（验证用） | Likely benign |
| 生成时间 | 2026-03-19 |
| 模型 | moonshot-v1-32k |
| Tokens | 1747 |

---

## 变异基本信息

变异位于基因组坐标chr13:32340561，基因BRCA2，核苷酸变化为T>C，导致蛋白质变化为p.? (missense)。

## SNV-judge集成预测

SNV-judge集成预测显示，该变异的致病概率为3.9%，该预测是基于XGBoost和LightGBM模型的集成，并通过Platt Scaling校准。

## 多维度证据分析

1. **[PM2_weak]** NEUTRAL: gnomAD中未收录（AF≈0），但SNV-judge概率=3.9%偏低，gnomAD未收录不足以单独支持致病（良性变异同样可能未被收录）。
   
   触发原因：gnomAD AF=0.00e+00，SNV-judge概率=3.9%。

2. **[BP4]** BENIGN (supporting): 多个计算工具预测良性：AlphaMissense=0.045, CADD=15.5, REVEL=0.155, PolyPhen-2=0.000。

   触发原因：AlphaMissense=0.0447, CADD Phred=15.49, REVEL=0.155, PolyPhen-2=0.0。

## SHAP特征贡献

本次预测中，对变异分类影响最大的特征按绝对值排序如下：

1. AlphaMissense: SHAP=-1.5760 (→良性)
2. REVEL: SHAP=-1.3971 (→良性)
3. gnomAD AF: SHAP=-1.0111 (→良性)
4. CADD Phred: SHAP=-0.8781 (→良性)
5. Genos-10B: SHAP=-0.5270 (→良性)
6. SIFT: SHAP=-0.5266 (→良性)
7. phyloP: SHAP=+0.1460 (→致病)
8. PolyPhen-2: SHAP=+0.0460 (→致病)

## ACMG综合分类

根据上述证据，我们可以得出以下结论：

- **[BP4]** BENIGN (supporting): 多个计算工具预测良性。
- **[PM2_weak]** NEUTRAL: gnomAD中未收录，不足以单独支持致病。

综合以上证据，该变异被分类为**可能良性（LP）**。支持证据汇总表如下：

| ACMG证据条目 | 证据描述 | 触发原因 | 证据权重 |
|--------------|----------|----------|----------|
| BP4          | 多个计算工具预测良性 | AlphaMissense=0.0447, CADD Phred=15.49, REVEL=0.155, PolyPhen-2=0.0 | 支持 |
| PM2_weak     | gnomAD中未收录 | gnomAD AF=0.00e+00, SNV-judge概率=3.9% | 中性 |

## 临床意义与局限性

该变异被分类为可能良性（LP），主要基于多个计算工具的预测结果和gnomAD数据库中未收录的情况。然而，需要注意的是，gnomAD数据库的缺失并不能排除该变异可能是良性的可能性，因为良性变异同样可能未被收录。此外，由于SNV-judge预测的致病概率较低，这进一步支持了该变异可能为良性的结论。

尽管如此，我们仍建议对患者进行进一步的临床评估和随访，以确定该变异的临床意义。此外，由于遗传变异的复杂性，单一的计算预测和数据库信息可能不足以完全确定变异的致病性，因此可能需要结合其他临床信息和家族史来做出最终的诊断。