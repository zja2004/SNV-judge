# SNV-judge v5

**基于多源异构特征融合的单核苷酸变异（SNV）致病性智能预测系统**

> 面向临床遗传学和科研场景的 AI 辅助变异解读工具。融合 8 维异构特征（经典注释工具 + 基因组基础模型 + 进化保守性 + 人群频率），输出 ACMG 5 级分类、校准概率、SHAP 可解释性分析，以及 LLM 生成的中文临床报告。

---

## 目录

- [性能指标](#性能指标)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [数据集](#数据集)
- [模型文件](#模型文件)
- [API 依赖](#api-依赖)
- [快速开始](#快速开始)
- [功能模块](#功能模块)
- [文件结构](#文件结构)
- [已知局限](#已知局限)
- [开发日志](#开发日志)

---

## 性能指标

| 评估场景 | AUROC | AUPRC | F1 | Brier |
|---------|-------|-------|-----|-------|
| 5-fold CV OOF（n=2000） | **0.9679** | 0.9643 | 0.9022 | 0.0685 |
| Holdout Test（n=300） | 0.9547 | 0.9527 | 0.8946 | 0.0789 |
| 离线模式（Evo2/Genos 中位数填充） | 0.9892 | — | — | — |
| LOGO-CV 非 BRCA 均值（14 基因） | 0.9642 | — | — | — |

**校准质量（Isotonic Regression）：**

| 方法 | Brier Score | ECE |
|------|-------------|-----|
| 未校准 | 0.0807 | 0.0639 |
| Platt Scaling | 0.0743 | 0.0233 |
| **Isotonic Regression（当前）** | **0.0685** | **≈ 0** |

**LOGO-CV 各基因 AUROC（16 基因，每基因 20–21 个变异）：**

| 基因 | 疾病类别 | AUROC |
|------|----------|-------|
| TP53, FBN1, LDLR, GAA, USH2A, RUNX1, BRCA1, BRCA2 | 多类别 | 1.0000 |
| SOS1 | RASopathy | 0.9909 |
| MSH2, MSH6 | 错配修复 | 0.9900 |
| RAF1 | RASopathy | 0.9800 |
| RYR1 | 肌肉骨骼 | 0.9545 |
| MECP2 | 神经发育 | 0.9300 |
| MLH1 | 错配修复 | 0.8727 |
| **MYH7** | **心肌病** | **0.7900** |

---

## 系统架构

```
输入：染色体坐标（GRCh38）或蛋白变异名称（如 TP53 R175H）
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                   特征提取层（8 维）                      │
│                                                         │
│  免费公共 API（无需 Key）          基础模型（可选）        │
│  ├── SIFT (inv)      ─── VEP      ├── Evo2-40B LLR      │
│  ├── PolyPhen-2      ─── VEP      │   (NVIDIA NIM API)  │
│  ├── AlphaMissense   ─── VEP      └── Genos-10B Score   │
│  ├── CADD Phred      ─── VEP          (本地 ngrok /     │
│  ├── phyloP          ─── VEP           Stomics Cloud)   │
│  └── gnomAD v4 AF    ─── gnomAD GraphQL                 │
│                                                         │
│  ★ 离线模式：Evo2/Genos 使用训练集中位数填充              │
│    Evo2 中位数 = -0.074，Genos 中位数 = 0.676            │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                   集成模型层                              │
│                                                         │
│  XGBoost ──┐                                            │
│            ├── Logistic Regression (Meta-learner)       │
│  LightGBM ─┘         │                                  │
│                       ▼                                 │
│              Isotonic Regression（概率校准）              │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                   输出层                                  │
│  ├── 校准概率（0–1）                                      │
│  ├── ACMG 5 级分类（P / LP / VUS / LB / B）              │
│  ├── SHAP 特征贡献（瀑布图 / 蜂群图）                     │
│  ├── ClinVar 实时查询（rsID / review_status / conditions）│
│  └── LLM 中文临床报告（6 章节，OpenAI-compatible API）    │
└─────────────────────────────────────────────────────────┘
```

**ACMG 5 级分类阈值：**

| 分级 | 阈值 | 置信度 |
|------|------|--------|
| Pathogenic (P) | ≥ 0.90 | High confidence |
| Likely Pathogenic (LP) | ≥ 0.70 | Moderate confidence |
| VUS | ≥ 0.40 | Uncertain significance |
| Likely Benign (LB) | ≥ 0.20 | Moderate confidence |
| Benign (B) | < 0.20 | High confidence |

---

## 技术栈

### 核心机器学习

| 库 | 版本 | 用途 |
|----|------|------|
| `xgboost` | ≥ 2.0.0 | 主分类器 |
| `lightgbm` | ≥ 4.0.0 | 辅助分类器 |
| `scikit-learn` | ≥ 1.3.0 | Stacking meta-learner、Isotonic 校准、交叉验证 |
| `shap` | ≥ 0.45.0 | SHAP 可解释性分析（瀑布图、蜂群图） |
| `numpy` | ≥ 1.24.0 | 数值计算 |
| `pandas` | ≥ 2.0.0 | 数据处理 |

### Web 应用

| 库 | 版本 | 用途 |
|----|------|------|
| `streamlit` | ≥ 1.32.0 | 交互式 Web 界面（`app.py`） |
| `matplotlib` | ≥ 3.7.0 | SHAP 图、校准曲线、LOGO-CV 图 |

### API 与网络

| 库 | 版本 | 用途 |
|----|------|------|
| `requests` | ≥ 2.31.0 | Ensembl VEP、gnomAD GraphQL、ClinVar efetch |
| `openai` | ≥ 1.0.0 | LLM 临床报告（任意 OpenAI-compatible 端点） |

### Python 版本

- **推荐**: Python 3.11+
- **最低**: Python 3.10（需要 `str | None` 类型注解语法）

---

## 数据集

### 训练集

| 属性 | 值 |
|------|-----|
| 来源 | ClinVar（2024 年 3 月快照） |
| 筛选条件 | 错义变异（missense）、review status ≥ 2 星、P/LP 或 B/LB |
| 总变异数 | **2,000 个**（1,000 P/LP + 1,000 B/LB） |
| 基因数 | 2,927 个基因 |
| 参考基因组 | GRCh38 |
| 文件 | `data/feature_matrix_v4.csv`（2,000 行 × 23 列） |

### 特征矩阵列说明

| 列名 | 来源 | 说明 |
|------|------|------|
| `allele_id` | ClinVar | ClinVar Allele ID |
| `chrom`, `pos`, `ref`, `alt` | ClinVar | GRCh38 坐标 |
| `gene` | ClinVar | 基因符号 |
| `label` | ClinVar | pathogenic / benign |
| `sift_score`, `sift_score_inv` | Ensembl VEP | SIFT 分数（inv = 1 - score） |
| `polyphen_score` | Ensembl VEP | PolyPhen-2 HumVar 分数 |
| `am_pathogenicity` | Ensembl VEP | AlphaMissense 致病性分数 |
| `cadd_phred` | Ensembl VEP | CADD Phred 分数 |
| `evo2_llr` | NVIDIA NIM API（evo2-40b） | Evo2 对数似然比（REF vs ALT） |
| `genos_path` | Stomics Cloud API | Genos-10B 致病性分数 |
| `phylop` | Ensembl VEP | phyloP 100-way 保守性分数 |
| `gnomad_af`, `gnomad_log_af` | gnomAD v4 GraphQL | 人群等位基因频率（log10 变换） |

### 测试集

| 属性 | 值 |
|------|-----|
| Holdout Test | 300 个变异（从 2,000 中随机分层抽取 15%） |
| LOGO-CV | 16 个基因 × 20–21 个变异 = 320 个变异 |

### 数据来源与版本

| 数据源 | 版本 / 日期 | 访问方式 |
|--------|------------|---------|
| ClinVar | 2024-03 快照 | NCBI E-utilities |
| Ensembl VEP | GRCh38 release 111 | REST API（免费） |
| AlphaMissense | v1.0 | 通过 VEP 插件 |
| CADD | v1.7 | 通过 VEP 插件 |
| gnomAD | v4.0 | GraphQL API（免费） |
| phyloP | 100-way vertebrate | 通过 VEP |
| Evo2 | 40B 参数版本 | NVIDIA NIM API（需 Key） |
| Genos | 10B 参数版本 | Stomics Cloud API（需 Key）/ 本地 ngrok |

---

## 模型文件

所有模型文件保存在仓库根目录，按版本命名：

| 文件 | 说明 |
|------|------|
| `xgb_model_v5.pkl` | XGBoost 主分类器（v5，8 特征） |
| `platt_scaler_v5.pkl` | Isotonic Regression 校准器 |
| `shap_explainer.pkl` | SHAP TreeExplainer（基于 XGBoost） |
| `train_medians_v5.pkl` | 训练集各特征中位数（离线模式填充用） |
| `feature_cols_v5.pkl` | 特征列名列表（模型输入顺序） |
| `data/gnomad_af_cache.pkl` | gnomAD AF 查询缓存（加速重复查询） |
| `data/phylop_cache.pkl` | phyloP 查询缓存 |
| `data/vep_scores.pkl` | VEP 注释缓存 |

> **注意**：v2–v4 版本的模型文件（`xgb_model_v2.pkl` 等）保留用于版本对比，生产环境请使用 v5。

---

## API 依赖

### 必需（免费，无需 Key）

| API | 端点 | 用途 |
|-----|------|------|
| Ensembl VEP REST | `https://rest.ensembl.org/vep/homo_sapiens/region` | SIFT / PolyPhen-2 / AlphaMissense / CADD / phyloP |
| Ensembl variant_recoder | `https://rest.ensembl.org/variant_recoder/homo_sapiens` | 蛋白变异名称 → 基因组坐标 |
| gnomAD GraphQL | `https://gnomad.broadinstitute.org/api` | 人群等位基因频率 |
| NCBI efetch | `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi` | ClinVar VCV 记录（review_status / conditions） |

### 可选（需要 Key 或本地服务）

| API | 获取方式 | 用途 |
|-----|---------|------|
| NVIDIA NIM Evo2-40B | [build.nvidia.com](https://build.nvidia.com) | Evo2 LLR 实时计算（离线模式用中位数替代） |
| Stomics Cloud Genos | [cloud.stomics.tech](https://cloud.stomics.tech) | Genos-10B 致病性评分（离线模式用中位数替代） |
| Genos 本地 ngrok | 自行部署 Genos-10B 后用 ngrok 暴露 | 本地 k-mer Jaccard 评分（`--genos-url` 参数） |
| 任意 LLM（OpenAI-compatible） | DashScope / Moonshot / OpenAI / DeepSeek | 中文临床报告生成 |

**离线模式说明**：不提供 Evo2 / Genos API Key 时，系统自动使用训练集中位数填充（Evo2 LLR = -0.074，Genos Score = 0.676），AUROC 从 0.9985 降至 0.9892，影响可接受。

---

## 快速开始

### 安装

```bash
git clone https://github.com/zja2004/SNV-judge.git
cd SNV-judge
pip install -r requirements.txt
```

### 方式一：Python API（推荐）

```python
from skill.scripts.predict import load_model_artifacts, predict_variant, print_shap_summary

# 加载模型（只需一次）
artifacts = load_model_artifacts(model_dir=".")

# 单变异预测（GRCh38 坐标）
result = predict_variant("17", 7674220, "C", "T", artifacts=artifacts)
print(f"致病概率: {result['cal_prob']:.1%}")   # → 99.1%
print(f"ACMG 分级: {result['acmg_class']}")    # → Pathogenic (P)
print_shap_summary(result)

# 蛋白变异名称解析
from skill.scripts.predict import resolve_protein_variant
coords = resolve_protein_variant("BRCA1", "C61G")
result = predict_variant(coords["chrom"], coords["pos"],
                         coords["ref"], coords["alt"], artifacts=artifacts)

# VCF 批量预测
from skill.scripts.predict import predict_vcf
df = predict_vcf("variants.vcf", artifacts=artifacts, output_csv="results.csv")

# ClinVar 查询
from skill.scripts.predict import fetch_clinvar_classification
cv = fetch_clinvar_classification("17", 7674220, "C", "T")
print(cv["clinical_significance"])  # → Likely Pathogenic / Pathogenic
print(cv["review_status"])          # → criteria provided, multiple submitters, no conflicts
print(cv["conditions"][:3])         # → ['Li-Fraumeni syndrome 1', ...]

# LLM 临床报告
from skill.scripts.predict import generate_clinical_report
report = generate_clinical_report(
    result,
    api_key="sk-xxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    template="chinese"
)
print(report)
```

### 方式二：命令行

```bash
# 单变异预测
python skill/scripts/predict.py 17 7674220 C T .

# 启用 ClinVar 查询
python skill/scripts/predict.py --gene TP53 --protein R175H . --clinvar

# VCF 批量预测
python skill/scripts/predict.py --vcf variants.vcf . --output results.csv

# 启用本地 Genos 评分
python skill/scripts/predict.py 17 7674220 C T . \
    --genos-url https://xxx.ngrok-free.dev
```

### 方式三：Streamlit Web 界面

```bash
streamlit run app.py
```

访问 `http://localhost:8501`，支持：
- 单变异预测（含 SHAP 瀑布图）
- 批量 VCF 文件上传预测
- 蛋白变异名称解析
- ClinVar 实时查询
- LLM 临床报告生成（需配置 API Key）
- 模型校准曲线与 LOGO-CV 泛化性可视化

---

## 功能模块

### `skill/scripts/predict.py` — 核心预测模块

| 函数 | 说明 |
|------|------|
| `load_model_artifacts(model_dir)` | 加载所有模型文件（pkl），返回 artifacts 字典 |
| `predict_variant(chrom, pos, ref, alt, artifacts, ...)` | 单变异预测，返回完整结果字典 |
| `predict_vcf(vcf_path, artifacts, output_csv)` | VCF 文件批量预测，返回 DataFrame |
| `resolve_protein_variant(gene, protein_change)` | 蛋白变异名称 → GRCh38 坐标（Ensembl variant_recoder） |
| `fetch_clinvar_classification(chrom, pos, ref, alt)` | ClinVar 实时查询（VEP → VCV efetch XML） |
| `fetch_genos_embedding_score(chrom, pos, ref, alt, genos_url)` | 本地 Genos k-mer Jaccard 评分 |
| `generate_clinical_report(result, api_key, ...)` | LLM 中文临床报告生成 |
| `print_result(result)` | 打印预测结果摘要 |
| `print_shap_summary(result)` | 打印 SHAP 特征贡献摘要 |
| `plot_shap_waterfall(result, save_path)` | 生成 SHAP 瀑布图 |

**`predict_variant()` 返回字典结构：**

```python
{
    "chrom": "17",
    "pos": 7674220,
    "ref": "C",
    "alt": "T",
    "cal_prob": 0.9907,           # 校准概率
    "raw_prob": 0.9923,           # 未校准概率
    "acmg_class": "Pathogenic (P)",
    "acmg_confidence": "High confidence",
    "feature_vec": [...],          # 8 维特征向量
    "shap_values": [...],          # 8 维 SHAP 值
    "vep_scores": {...},           # VEP 注释详情
    "gnomad_log_af": -4.48,
    "clinvar": {...},              # ClinVar 查询结果（若启用）
}
```

### `kimi_report.py` — LLM 临床报告模块

支持任意 OpenAI-compatible API 端点：

| Provider | base_url |
|----------|---------|
| Alibaba DashScope | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| Moonshot (Kimi) | `https://api.moonshot.cn/v1` |
| OpenAI | `https://api.openai.com/v1` |
| DeepSeek | `https://api.deepseek.com/v1` |

报告模板：`"chinese"`（默认）/ `"english"` / `"summary"`

### `train.py` — 模型训练脚本

完整复现训练流程（需要 Evo2 + Genos API Key）：

```bash
export EVO2_API_KEY="nvapi-..."
export GENOS_API_KEY="sk-..."
python train.py
```

### `skill/scripts/generalization_eval.py` — LOGO-CV 评估

```python
from skill.scripts.generalization_eval import run_logo_cv
results = run_logo_cv(model_dir=".", n_splits=5)
```

---

## 文件结构

```
SNV-judge/
├── app.py                          # Streamlit Web 应用
├── train.py                        # 模型训练脚本（需 API Key）
├── kimi_report.py                  # LLM 临床报告模块
├── requirements.txt                # Python 依赖
├── opencode.json                   # OpenCode AI 编辑器配置
│
├── skill/                          # Agent Skill 包
│   ├── SKILL.md                    # Skill 指令文档（AI agent 接口）
│   ├── scripts/
│   │   ├── predict.py              # 核心预测模块（主要入口）
│   │   └── generalization_eval.py  # LOGO-CV 泛化性评估
│   └── references/
│       ├── acmg-criteria.md        # ACMG/AMP 2015 证据条目参考
│       └── troubleshooting.md      # 常见错误与解决方案
│
├── data/
│   ├── feature_matrix_v4.csv       # 训练集特征矩阵（2000 × 23）
│   ├── feature_matrix_v4.xlsx      # 同上（Excel 格式）
│   ├── clinvar_raw.csv             # ClinVar 原始数据
│   ├── model_metrics_v5.csv        # v5 模型性能指标
│   ├── calibration_metrics_v5.csv  # 校准方法对比
│   ├── gnomad_af_cache.pkl         # gnomAD 查询缓存
│   ├── phylop_cache.pkl            # phyloP 查询缓存
│   └── vep_scores.pkl              # VEP 注释缓存
│
├── figures/
│   ├── fig1_roc_comparison.svg     # ROC 曲线对比（各版本）
│   ├── fig2_ablation.svg           # 特征消融实验
│   ├── fig3_data_distribution.svg  # 数据集分布
│   ├── fig4_architecture.svg       # 系统架构图
│   ├── fig_logo_cv.svg             # LOGO-CV 泛化性结果
│   ├── fig_calibration_comparison.svg  # 校准方法对比
│   ├── fig_shap_summary_beeswarm.svg   # SHAP 蜂群图
│   ├── fig_shap_waterfall_tp53.svg     # TP53 R175H SHAP 瀑布图
│   └── fig_shap_waterfall_brca2.svg    # BRCA2 N372H SHAP 瀑布图
│
├── docs/
│   └── session_2026-04-17_bug_fixes.md  # 开发日志
│
├── test_screenshots/               # Skill 全量测试截图
│   ├── branch1_shap_comparison.png
│   ├── branch2_batch_prediction.png
│   ├── branch3_vcf_prediction.png
│   ├── branch4_resolve_protein.png
│   ├── branch5_clinvar.png
│   ├── branch6_clinical_report.png
│   └── skill_test_report.md        # 测试报告
│
├── xgb_model_v5.pkl                # XGBoost 模型（v5）
├── platt_scaler_v5.pkl             # Isotonic 校准器（v5）
├── shap_explainer.pkl              # SHAP TreeExplainer
├── train_medians_v5.pkl            # 训练集中位数（离线填充）
├── feature_cols_v5.pkl             # 特征列名
│
└── [v2–v4 历史版本文件]             # 保留用于版本对比
```

---

## 已知局限

### 1. MYH7 泛化性不足（AUROC = 0.79）

MYH7 致病变异以**增益功能（gain-of-function）机制**为主，而 SIFT、PolyPhen-2 等工具基于序列保守性和结构稳定性设计，对增益功能变异的区分能力有限。SIFT/PolyPhen-2 在 MYH7 致病与良性变异间的分布高度重叠。

**改进方向**：引入 AlphaFold2 预测的蛋白质结构特征（pLDDT、ΔΔG）；针对心肌病基因设计专项子模型。

### 2. Genos 本地评分分布偏移

本地 Genos-10B `/generate` 端点的 k-mer Jaccard 距离（范围 ~0.00–0.15）与训练集 genos_score（中位数 0.676）量纲不同，已加入线性重映射（`mapped = 0.30 + jaccard × 4.0`）作为工程近似。根本解决方案需要用本地方案重新计算训练集并重训模型。

### 3. ClinVar conditions 非 allele-specific

`fetch_clinvar_classification()` 返回的 conditions 来自 rsID 级别的 VCV 记录，对多等位基因位点（如 rs11540652 对应 C/A/G/T 四种等位基因）可能包含其他等位基因的疾病条目。

### 4. 小样本 LOGO-CV 置信区间宽

每个基因仅 20–21 个变异（10P/10B），AUROC=1.00 的基因（共 8 个）置信区间极宽，不代表完美泛化。

### 5. BRCA2 N372H 过度评分

BRCA2 N372H（ClinVar = Benign）被预测为 Likely Pathogenic（87.1%），原因是 gnomAD AF 较低（log-AF = -4.48），模型将其解读为致病信号。这是 gnomAD 作为第一重要特征的已知局限。

---

## 开发日志

| 版本 | 主要变更 |
|------|---------|
| v1 | 基础模型：SIFT + PolyPhen-2 + AlphaMissense + CADD，XGBoost，AUROC = 0.9503 |
| v2 | 集成 Evo2-40B + Genos-10B 基础模型，Stacking 集成，AUROC = 0.9664 |
| v3 | 新增 phyloP 进化保守性特征，扩展训练集至 2,000 变异 |
| v4 | 新增 gnomAD v4 人群频率特征，AUROC = 0.9664 |
| v5 | Isotonic Regression 校准（ECE ≈ 0），ACMG 5 级分类，SHAP 可解释性，LLM 临床报告，LOGO-CV 泛化性评估 |
| v5.1 | 批量 VEP 注释、VCF 支持、蛋白变异名称解析、ClinVar 查询 |
| v5.2 | Genos 本地 ngrok 支持（k-mer Jaccard 方案） |
| v5.3 | 修复 resolve_protein_variant（vcf_string=1 + 版本号去除）、fetch_clinvar_classification 重写（VEP→VCV efetch XML）、Genos 分布重映射、LLM ACMG 保守性偏差修复 |

---

## 给下一位开发者

### 环境配置

```bash
# 推荐 Python 3.11
conda create -n snv-judge python=3.11
conda activate snv-judge
pip install -r requirements.txt
```

### 最重要的文件

1. **`skill/scripts/predict.py`** — 所有核心功能的入口，包含最新的 bug 修复
2. **`kimi_report.py`** — LLM 报告模块，独立于 predict.py
3. **`skill/SKILL.md`** — AI agent 接口文档，包含完整的函数调用示例
4. **`skill/references/troubleshooting.md`** — 常见错误和解决方案

### 待完成工作

- [ ] MYH7 性能改进（引入 AlphaFold2 结构特征）
- [ ] Genos 评分系统性校准（重新计算训练集 Jaccard 值并重训模型）
- [ ] ClinVar conditions allele-specific 改进（通过 RCV 编号过滤）
- [ ] 扩展训练集（纳入 ClinVar 2024 更新数据，约 12 万条新记录）
- [ ] LOGO-CV Bootstrap 置信区间（1000 次重采样）
- [ ] app.py 标题 v4 → v5 修复

### 关键 API 端点

```python
# Ensembl VEP（免费，无需 Key）
VEP_URL = "https://rest.ensembl.org/vep/homo_sapiens/region"

# gnomAD GraphQL（免费，无需 Key）
GNOMAD_URL = "https://gnomad.broadinstitute.org/api"

# ClinVar efetch（免费，无需 Key）
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# NVIDIA NIM Evo2（需要 API Key）
EVO2_URL = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/generate"

# Genos 本地（需要本地部署 + ngrok）
GENOS_URL = "https://xxx.ngrok-free.dev/generate"  # 只有 /generate 和 /health 端点
```

### 注意事项

1. **Genos 服务器**：只有 `/generate`（DNA 续写）和 `/health` 端点，没有 `/extract`（embedding）端点。当前使用 k-mer Jaccard 距离方案，存在分布偏移问题（见已知局限 #2）。
2. **ClinVar esearch**：`esearch clinvar term=rsID[rs]` 在 NCBI 中不可靠，始终返回空列表。正确方案是通过 VEP `var_synonyms.ClinVar` 获取 VCV 编号，再用 efetch 查询。
3. **Ensembl variant_recoder**：调用时必须加 `?vcf_string=1` 参数，且 transcript ID 需去掉版本号（如 `ENST00000357654.9` → `ENST00000357654`）。
4. **LLM ACMG 分级**：必须通过 `acmg_class` 参数将模型输出的分级传给 `generate_report_stream()`，否则 LLM 会自行降级（如将 Pathogenic 降为 Likely Pathogenic）。

---

*最后更新：2026-04-21 | GitHub: [zja2004/SNV-judge](https://github.com/zja2004/SNV-judge)*
