# SNV-judge v5 — 临床 SNV 致病性预测 Skill

## 概述

SNV-judge v5 是一个基于机器学习的单核苷酸变异（SNV）致病性预测工具，
专为临床遗传学和科研场景设计。

**核心特点：**
- 直接调用训练好的模型文件，**无需 Evo2 / Genos API Key**
- 仅依赖免费公共 API（Ensembl VEP + gnomAD GraphQL）
- 输出 ACMG 5 级分类（P/LP/VUS/LB/B）+ 校准概率 + SHAP 解释
- 支持单变异预测、批量 VCF 处理、Leave-One-Gene-Out 泛化验证

---

## 模型架构

```
输入特征（8维）
├── SIFT (inv)         ← Ensembl VEP（免费）
├── PolyPhen-2         ← Ensembl VEP（免费）
├── AlphaMissense      ← Ensembl VEP（免费）
├── CADD Phred         ← Ensembl VEP（免费）
├── Evo2-40B LLR       ← 训练集中位数填充（离线模式）★
├── Genos-10B Score    ← 训练集中位数填充（离线模式）★
├── phyloP             ← Ensembl VEP（免费）
└── gnomAD v4 log-AF   ← gnomAD GraphQL（免费）

★ 离线模式：Evo2/Genos 使用训练集中位数（-0.074 / 0.676）填充
  AUROC 影响：0.9985 → 0.9892（下降 0.0093，可接受）

集成模型
├── XGBoost（主分类器）
├── LightGBM（辅助分类器）
├── Logistic Regression（Meta-learner）
└── Isotonic Regression（概率校准）
```

---

## 性能指标

| 评估场景 | AUROC |
|---------|-------|
| 训练集（完整 8 特征） | 0.9985 |
| **离线模式（Evo2/Genos 中位数填充）** | **0.9892** |
| LOGO-CV 非 BRCA 基因均值 | 0.9642 |
| BRCA1/2 参考 | 1.000 |
| 最弱基因（MYH7） | 0.790 |

**LOGO-CV 验证基因（16 个）：**
TP53, MLH1, MSH2, MSH6, MYH7, FBN1, LDLR, MECP2, RYR1, GAA, USH2A, RUNX1, SOS1, RAF1, BRCA1, BRCA2

---

## 快速开始

### 1. 安装依赖

```bash
pip install xgboost lightgbm scikit-learn shap requests numpy pandas
```

### 2. 单变异预测（推荐）

```python
from scripts.predict import load_model_artifacts, predict_variant, print_shap_summary

# 加载模型（只需一次，可复用）
artifacts = load_model_artifacts(model_dir="/path/to/SNV-judge")

# 预测 TP53 R175H（chr17:7674220 C>T，GRCh38）
result = predict_variant("17", 7674220, "C", "T", artifacts=artifacts)

print(f"致病概率: {result['cal_prob']:.1%}")
print(f"ACMG 分级: {result['acmg_class']}")
print_shap_summary(result)
```

### 3. 批量 VCF 预测

```python
from scripts.predict import load_model_artifacts, predict_variant
import pandas as pd

artifacts = load_model_artifacts("/path/to/SNV-judge")

variants = [
    ("17", 7674220, "C", "T"),   # TP53 R175H
    ("13", 32338271, "G", "A"),  # BRCA2 N372H
    ("17", 43094692, "G", "A"),  # BRCA1 R1699W
]

results = []
for chrom, pos, ref, alt in variants:
    r = predict_variant(chrom, pos, ref, alt, artifacts=artifacts)
    results.append({
        "variant": f"chr{chrom}:{pos}{ref}>{alt}",
        "gene": r["gene"],
        "hgvsp": r["hgvsp"],
        "cal_prob": r["cal_prob"],
        "acmg_class": r["acmg_class"],
    })

df = pd.DataFrame(results)
print(df.to_string(index=False))
```

### 4. 命令行使用

```bash
# 格式: python predict.py <chrom> <pos> <ref> <alt> [model_dir]
python scripts/predict.py 17 7674220 C T /path/to/SNV-judge
```

---

## 文件结构

```
snv-judge-skill/
├── SKILL.md                    # 本文件
├── scripts/
│   ├── predict.py              # 核心预测脚本（离线模式，无需 Evo2/Genos）
│   └── generalization_eval.py  # LOGO-CV 泛化验证脚本
└── references/
    ├── acmg-criteria.md        # ACMG 证据标准参考
    └── troubleshooting.md      # 常见问题排查
```

**依赖的模型文件（来自 SNV-judge 项目）：**
```
SNV-judge/
├── xgb_model_v5.pkl            # XGBoost + LightGBM + LR 集成模型
├── platt_scaler_v5.pkl         # Isotonic Regression 校准器
├── train_medians_v5.pkl        # 训练集特征中位数（用于填充缺失值）
└── data/feature_matrix_v4.csv  # 训练集特征矩阵（2000 变异 × 22 列）
```

---

## API 依赖说明

| API | 用途 | 是否需要 Key | 限速 |
|-----|------|------------|------|
| Ensembl VEP REST | SIFT/PolyPhen/AM/CADD/phyloP | ❌ 无需 | 15 req/s |
| gnomAD GraphQL | 人群频率 | ❌ 无需 | 宽松 |
| Evo2-40B | LLR 评分 | ~~需要~~ → **中位数填充** | — |
| Genos-10B | 致病性评分 | ~~需要~~ → **中位数填充** | — |

---

## 输出字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `cal_prob` | float | 校准后致病概率 [0, 1] |
| `acmg_class` | str | ACMG 分级（P/LP/VUS/LB/B） |
| `acmg_confidence` | str | 置信度描述 |
| `shap_values` | list | 各特征 SHAP 贡献值 |
| `top_shap_feature` | str | 贡献最大的特征名 |
| `gene` | str | 基因符号（来自 VEP） |
| `hgvsp` | str | 蛋白变化（如 p.Arg175His） |
| `consequence` | str | 变异后果类型 |
| `offline_mode` | bool | True（Evo2/Genos 使用中位数填充） |

---

## 注意事项

1. **离线模式精度**：Evo2/Genos 中位数填充导致 AUROC 从 0.9985 降至 0.9892，
   对于 VUS 边界案例（概率 0.4–0.6）建议结合临床证据综合判断。

2. **变异类型**：模型针对 missense 变异训练，对 synonymous/intronic 变异
   预测结果不可靠（VEP 会给出警告）。

3. **坐标系**：使用 GRCh38/hg38，染色体不含 "chr" 前缀。

4. **网络要求**：需要访问 rest.ensembl.org 和 gnomad.broadinstitute.org。
   如在内网环境，可预先缓存 VEP 结果。

---

## 引用

如使用本工具，请引用：
- SNV-judge v5（本项目）
- Ensembl VEP: McLaren et al., Genome Biology 2016
- AlphaMissense: Cheng et al., Science 2023
- gnomAD v4: Chen et al., Nature 2024
- CADD: Rentzsch et al., Nucleic Acids Research 2019
