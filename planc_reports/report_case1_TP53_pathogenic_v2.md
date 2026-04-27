# SNV-judge Plan C — ACMG智能解读报告（v2）

| 字段 | 值 |
|------|----|
| 变异 | TP53 chr17:7673781 C>G (GRCh38) |
| SNV-judge致病概率 | 93.9% |
| ClinVar标注（验证用） | Pathogenic/Likely pathogenic |
| 生成时间 | 2026-03-19 |
| 模型 | moonshot-v1-32k |
| Tokens | 1740 |

---

## 变异基本信息

该变异位于人类基因组第17号染色体上，具体位置为7673781，由胞嘧啶（C）突变为鸟嘌呤（G）。该变异影响的基因是TP53，这是一种与肿瘤抑制相关的基因。蛋白质变化为错义突变，导致氨基酸序列发生改变，但具体改变的氨基酸位置未给出。

## SNV-judge集成预测

根据SNV-judge集成预测模型，该变异的致病概率为93.9%，模型由XGBoost和LightGBM集成，并通过Platt Scaling校准。这一高概率表明该变异很可能是致病的。

## 多维度证据分析

1. **[PM2] PATHOGENIC (moderate)**: gnomAD数据库中未收录该变异（AF≈0），结合SNV-judge模型的高概率（93.9%），支持该变异为极罕见致病变异。
   
2. **[PP3] PATHOGENIC (supporting)**: 多个计算工具预测该变异为有害。AlphaMissense得分为0.9997，CADD Phred得分为29.6，REVEL得分为0.94，SIFT得分为1.0（倒数）。这些高得分表明该变异很可能影响蛋白质功能，支持其致病性。

3. **[PP3_cons] PATHOGENIC (supporting)**: phyloP得分为2.55，大于等于2.0，表明该变异位点在进化上高度保守，支持其致病性。

4. **[PP3_fm] PATHOGENIC (supporting)**: Genos-10B致病性评分为0.9735，大于等于0.8，表明基础模型支持该变异为致病。

## SHAP特征贡献

SHAP特征贡献显示，AlphaMissense、CADD Phred、REVEL对本次预测致病性的贡献最大，分别为+3.4153、+1.6542、+1.6523。这些高正值表明这些特征强烈支持该变异的致病性。而gnomAD AF的SHAP值为-1.0419，PolyPhen-2的SHAP值为-0.9599，这两个负值表明这些特征对预测的致病性贡献较小，但考虑到PolyPhen-2得分为nan，其SHAP值可能不具代表性。

## ACMG综合分类

根据上述证据，我们可以得出以下支持证据汇总表：

| ACMG证据条目 | 描述 | 权重 |
|--------------|------|------|
| PM2          | gnomAD中未收录，模型高概率 | moderate |
| PP3          | 多个计算工具预测有害 | supporting |
| PP3_cons     | 高度进化保守 | supporting |
| PP3_fm       | Genos-10B致病性评分高 | supporting |

综合考虑，该变异被分类为**Pathogenic**。

## 临床意义与局限性

该变异在TP53基因上，这是一个与多种癌症相关的基因。根据多维度计算证据和SHAP特征贡献分析，该变异很可能是致病的。然而，需要注意的是，尽管计算工具和数据库提供了强有力的证据，但它们不能替代实验验证。因此，对于临床决策，建议结合患者的临床表现和其他遗传信息进行综合评估。此外，gnomAD数据库中未收录该变异，虽然不足以单独支持致病，但结合模型高概率，我们认为这增加了该变异致病的可能性。