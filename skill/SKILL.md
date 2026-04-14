# SNV-judge v5 — 临床 SNV 致病性预测 Skill

## 概述

SNV-judge v5 是一个基于机器学习的单核苷酸变异（SNV）致病性预测工具，
专为临床遗传学和科研场景设计。

**核心特点：**
- 直接调用训练好的模型文件，**无需 Evo2 / Genos API Key**
- 仅依赖免费公共 API（Ensembl VEP + gnomAD GraphQL）
- 输出 ACMG 5 级分类（P/LP/VUS/LB/B）+ 校准概率 + SHAP 解释
- 支持单变异预测、VCF 批量处理、蛋白变异名称解析、ClinVar 实时查询

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

### 3. 蛋白变异名称解析（新功能）

```python
from scripts.predict import resolve_protein_variant, predict_variant

# 直接用基因名 + 蛋白变化，自动解析为基因组坐标
coords = resolve_protein_variant("TP53", "R175H")
# → {"chrom": "17", "pos": 7674220, "ref": "C", "alt": "T", ...}

result = predict_variant(coords["chrom"], coords["pos"],
                         coords["ref"], coords["alt"], artifacts=artifacts)
```

### 4. VCF 批量预测（新功能）

```python
from scripts.predict import load_model_artifacts, predict_vcf

artifacts = load_model_artifacts("/path/to/SNV-judge")

# 批量预测 VCF 文件（支持 .vcf 和 .vcf.gz）
results = predict_vcf("variants.vcf", artifacts=artifacts, output_csv="results.csv")

# 结果自动保存为 CSV，包含 cal_prob、acmg_class、gene、hgvsp 等字段
```

### 5. SHAP 瀑布图可视化（新功能）

```python
from scripts.predict import predict_variant, plot_shap_waterfall

result = predict_variant("17", 7674220, "C", "T", artifacts=artifacts)

# 生成并保存 SHAP 瀑布图
fig = plot_shap_waterfall(result, save_path="shap_waterfall.svg")
```

### 6. ClinVar 实时查询（新功能）

```python
from scripts.predict import fetch_clinvar_classification

cv = fetch_clinvar_classification("17", 7674220, "C", "T")
print(f"ClinVar: {cv['clinical_significance']}")
print(f"证据级别: {cv['review_status']}")
print(f"相关疾病: {cv['conditions']}")
```

### 7. 命令行使用

```bash
# 单变异预测
python scripts/predict.py 17 7674220 C T /path/to/SNV-judge

# VCF 批量预测 + CSV 输出
python scripts/predict.py --vcf variants.vcf /path/to/SNV-judge --output results.csv

# 蛋白变异名称解析
python scripts/predict.py --gene TP53 --protein R175H /path/to/SNV-judge

# 附带 ClinVar 查询 + SHAP 图
python scripts/predict.py 17 7674220 C T /path/to/SNV-judge --clinvar --shap shap.svg
```

---

## 文件结构

```
snv-judge-skill/
├── SKILL.md                    # 本文件
├── scripts/
│   ├── predict.py              # 核心预测脚本（含全部新功能）
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
| NCBI E-utilities | ClinVar 查询 | ❌ 无需（有 Key 可提速） | 3 req/s |
| Ensembl HGVS | 蛋白变异坐标解析 | ❌ 无需 | 15 req/s |
| Evo2-40B | LLR 评分 | ~~需要~~ → **中位数填充** | — |
| Genos-10B | 致病性评分 | ~~需要~~ → **中位数填充** | — |

---

## 新功能说明（v5.1）

### ① 批量 VEP 注释（fetch_vep_batch）
- 最多 200 变异/次，比逐个调用快 10–50 倍
- 自动处理限速（429 响应），支持重试

### ② VCF 文件支持（parse_vcf + predict_vcf）
- 支持 `.vcf` 和 `.vcf.gz` 格式
- 自动过滤非 SNV（indel、多等位基因取第一个 ALT）
- 并行 gnomAD 查询（4 线程），批量 VEP 注释

### ③ SHAP 瀑布图（plot_shap_waterfall）
- 水平条形图，红色→致病贡献，蓝色→良性贡献
- 自动生成标题（含变异坐标、基因、预测概率）
- 支持保存为 SVG/PNG

### ④ 蛋白变异名称解析（resolve_protein_variant）
- 输入：基因名 + 蛋白变化（如 "TP53", "R175H"）
- 自动转换为三字母氨基酸码（R175H → p.Arg175His）
- 通过 Ensembl HGVS API 解析为 GRCh38 坐标

### ⑤ 模型版本特征数量验证
- 自动检测 `n_features_in_`，与版本期望值比对
- v5 期望 8 特征，v4 期望 6 特征
- 不匹配时自动调整特征列表并给出警告

### ⑥ 基因组上下文缓存（lru_cache）
- `fetch_genomic_context()` 使用 `@lru_cache(maxsize=512)`
- 批量预测时避免重复 API 调用，显著提升速度

### ⑦ ClinVar 实时查询（fetch_clinvar_classification）
- 通过 NCBI E-utilities 查询临床意义分类
- 返回：clinical_significance、review_status、conditions、last_evaluated
- 完全免费，无需 API Key

### ⑧ CSV 批量输出（save_results_csv + --output）
- 输出字段：chrom、pos、ref、alt、gene、hgvsp、cal_prob、acmg_class 等
- CLI 支持 `--output results.csv` 参数

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

5. **VCF 批量预测**：建议 batch_size=50（默认），最大 200。
   过大批次可能触发 VEP 限速。

---

## 引用

如使用本工具，请引用：
- SNV-judge v5（本项目）
- Ensembl VEP: McLaren et al., Genome Biology 2016
- AlphaMissense: Cheng et al., Science 2023
- gnomAD v4: Chen et al., Nature 2024
- CADD: Rentzsch et al., Nucleic Acids Research 2019
