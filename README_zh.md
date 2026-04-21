# SNV-judge v5.2

> 🌐 语言切换：[English](README.md) | **中文**

面向人类错义 SNV 致病性预测的可解释集成元模型。融合经典评分工具、基因组基础模型、进化保守性与人群等位基因频率，并内置通用 LLM 驱动的 AI 临床报告生成器。

在 **2,000 个 ClinVar 错义变异**（547 个基因，1:1 致病/良性，专家组审核，GRCh38）上训练。

![SHAP 分析](figures/shap_analysis_v4.png)

---

## 核心指标

| 指标 | 数值 |
|------|------|
| 5 折 CV AUROC | **0.9679** [0.959–0.974] |
| 5 折 CV AUPRC | **0.9643** [0.953–0.972] |
| 独立测试集 AUROC（n=300）| **0.9547** |
| 独立测试集 F1 | **0.8946** |
| Brier 分数 | **0.0685**（Isotonic 校准）|
| 训练变异数 | 2,000（547 个基因）|
| 特征数 | 8 |

---

## 版本更新

### v5.2（当前版本）
- **`skill/scripts/predict.py`** — 离线预测 CLI（1,163 行，18 个函数）：VEP 注释、gnomAD AF、ClinVar 查询、SHAP 瀑布图、批量 VCF 评分、`print_shap_summary()` 终端输出
- **通用 LLM 后端** — AI 临床报告支持任意 OpenAI 兼容服务商（阿里云百炼、Moonshot、OpenAI、DeepSeek 等）；应用内服务商选择器 + 自动拉取模型列表
- **Genos 本地 embedding 模式** — 无需 Stomics Cloud API Key，通过本地 ngrok 端点计算 REF/ALT 序列余弦距离
- **LOGO-CV 图表** — 模型信息面板中展示 16 个疾病基因家族的泛化性评估

### v5.0
- Isotonic Regression 校准（Brier 0.0685 vs Platt 0.0743，降低 7.8%）
- ACMG 5 级徽章（P / LP / VUS / LB / B）附置信度说明
- 会话级变异查询历史，支持并排 SHAP 对比
- 可靠性图（校准曲线）
- AI 报告模板选择：中文 / 英文 / 摘要

### 版本对比

| | v1 | v2 | v3 | v4 | v5 | **v5.2** |
|---|---|---|---|---|---|---|
| 训练变异数 | 842 | 1,800 | 2,000 | 2,000 | 2,000 | **2,000** |
| 特征数 | 4 | 6 | 7 | 8 | 8 | **8** |
| 校准方法 | — | Platt | Platt | Platt | Isotonic | **Isotonic** |
| AUROC（CV）| ~0.91 | 0.9373 | 0.9488 | 0.9664 | 0.9679 | **0.9679** |
| 离线 CLI | — | — | — | — | — | **✓** |
| 通用 LLM | — | — | — | — | — | **✓** |

---

## 功能特点

- **实时 VEP 注释** — 通过 Ensembl VEP REST API 获取 SIFT、PolyPhen-2、AlphaMissense、CADD、phyloP
- **Evo2-40B 评分** — 零样本对数似然比（9.3 万亿 token DNA 基础模型）[1]
- **Genos-10B 评分** — 以人类为中心的基因组基础模型致病性评分 [2]；v5.2 支持本地 embedding 模式
- **gnomAD v4 AF** — 人群等位基因频率（ACMG BA1/PM2 信号）
- **Stacking 集成** — XGBoost + LightGBM 基础学习器，逻辑回归元学习器
- **Isotonic 校准** — 比 Platt 缩放更好的概率校准
- **ACMG 5 级徽章** — 自动 P/LP/VUS/LB/B 分类
- **SHAP 可解释性** — 每个变异的特征贡献图 + `print_shap_summary()` 终端输出
- **🤖 AI 临床报告** — 任意 OpenAI 兼容 LLM 将所有工具输出综合为结构化 ACMG 风格报告（中文 / 英文 / 摘要）；应用内服务商/模型选择器，支持自动拉取模型列表
- **离线 CLI** — `skill/scripts/predict.py` 支持无 Streamlit UI 的批量 VCF 评分
- **Streamlit UI** — 致病性仪表盘、评分条形图、SHAP 图、变异历史、AI 报告标签页

---

## 模型性能

### 交叉验证（5 折，n=2,000）

| 模型 | AUROC [95% CI] | AUPRC [95% CI] | Brier |
|---|---|---|---|
| **v5：Isotonic 校准** | **0.9679 [0.959–0.974]** | **0.9643 [0.953–0.972]** | **0.0685** |
| v4：Platt 校准 | 0.9664 [0.958–0.972] | 0.9671 [0.956–0.973] | 0.0743 |
| v3：+ phyloP | 0.9488 [0.938–0.958] | 0.9447 [0.933–0.955] | — |
| v2：+ Evo2 + Genos | 0.9373 [0.927–0.947] | 0.9345 [0.921–0.946] | — |
| AlphaMissense 单独 | 0.9109 [0.898–0.923] | 0.9393 [0.927–0.948] | — |
| CADD 单独 | 0.9039 [0.886–0.916] | 0.9220 [0.903–0.938] | — |
| Genos 单独 | 0.6478 [0.619–0.674] | 0.7231 [0.683–0.749] | — |

### 独立测试集（n=300，150 致病 / 150 良性）

| 数据集 | AUROC | AUPRC | F1 | 灵敏度 | 特异度 |
|---|---|---|---|---|---|
| 5 折 CV OOF | 0.9679 | 0.9643 | 0.9022 | — | — |
| **独立测试集** | **0.9547** | **0.9527** | **0.8946** | **0.9333** | **0.8467** |

![校准对比](figures/fig_calibration_comparison.png)
![独立测试集验证](figures/fig_validation_holdout.png)

### ROC 与 PR 曲线

![模型曲线](figures/fig1_roc_comparison.png)

### 消融实验

![消融实验](figures/fig2_ablation.png)

### 泛化性 — 留一基因交叉验证（LOGO-CV）

对每个基因，在所有其他基因上训练新模型，在留出基因的变异上评估（每基因约 20 个，10P/10B）。

![LOGO-CV](figures/fig_logo_cv.png)

| 基因 | 疾病类别 | AUROC | F1 |
|------|---------|-------|----|
| BRCA1, BRCA2 | 遗传性乳腺癌/卵巢癌 | 1.000 | 0.952 / 0.909 |
| TP53, FBN1, LDLR, GAA, USH2A, RUNX1 | 抑癌基因 / 结缔组织 / 代谢 / 视网膜 / 血液 | 1.000 | 0.91–1.00 |
| SOS1, RAF1, MSH2, MSH6 | RAS 病 / Lynch MMR | 0.980–0.991 | 0.857–0.952 |
| RYR1, MECP2 | 肌肉骨骼 / 神经发育 | 0.930–0.955 | 0.750–0.857 |
| MLH1 | Lynch MMR | 0.873 | 0.870 |
| **MYH7** | **心肌病** | **0.790** | **0.800** |

**非 BRCA 基因平均 AUROC = 0.9642**（14 个基因）。MYH7 是最弱的基因——详见[局限性](#局限性)。

> 每个基因测试集仅约 20 个变异，AUROC 置信区间约 ±0.10。部分基因 AUROC = 1.00 反映小样本完美分离，不代表真实世界性能。

---

## AI 临床报告

支持**任意 OpenAI 兼容 LLM**，无需重启即可在应用内配置：

```
工具输出（VEP + Evo2 + Genos + gnomAD + SHAP）
        ↓
  kimi_report.py — 证据格式化
        ↓
  你选择的 LLM（Qwen / Kimi / GPT-4o / DeepSeek / ...）
        ↓
  结构化 ACMG 风格报告（流式 Markdown）
```

**内置服务商预设：**

| 服务商 | Base URL |
|--------|----------|
| Moonshot (Kimi) | `https://api.moonshot.cn/v1` |
| 阿里云百炼 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| OpenAI | `https://api.openai.com/v1` |
| DeepSeek | `https://api.deepseek.com/v1` |
| 自定义 | 任意 OpenAI 兼容端点 |

**应用内配置**：AI Report 标签页 → ⚙️ LLM 设置 → 选服务商 → 输入 API Key → 点「获取模型列表」→ 选模型 → 生成报告。

**报告模板**：中文临床报告 / 英文临床报告 / 简版摘要

---

## 未来愿景：Plan C — 多智能体系统

```
用户自然语言查询
        ↓
  规划智能体（LangGraph）
  ├── VEP 智能体       → SIFT · PP2 · AM · CADD · phyloP
  ├── Evo2 智能体      → NVIDIA NIM LLR 评分
  ├── Genos 智能体     → Cloud / 本地 embedding 评分
  ├── gnomAD 智能体    → GraphQL 人群频率
  ├── SNV-judge 智能体 → 集成预测 + SHAP
  └── 报告智能体       → ACMG 结构化临床报告
        ↓
  多轮对话 · 批量 VCF · 长期记忆
```

![Plan C 架构](figures/figB3_plan_c_vision.png)

---

## 仓库结构

```
SNV-judge/
├── app.py                          # Streamlit 网页应用（v5.2）
├── kimi_report.py                  # 通用 LLM 报告后端
├── train.py                        # 完整训练流水线
├── requirements.txt
│
├── xgb_model_v5.pkl                # Stacking 分类器（v5）
├── platt_scaler_v5.pkl             # Isotonic Regression 校准器（v5）
├── train_medians_v5.pkl            # 训练集特征中位数（缺失值填充）
├── feature_cols_v5.pkl             # 特征列名
│
├── data/
│   ├── feature_matrix_v4.csv/xlsx  # 2,000 变异特征矩阵
│   ├── calibration_metrics_v5.csv  # 校准对比数据
│   ├── model_metrics_v5.csv        # CV + 独立测试集指标
│   ├── scoring_ckpt.pkl            # 预计算 Evo2 LLR + Genos 评分
│   ├── vep_scores.pkl              # 预计算 VEP 评分
│   ├── phylop_cache.pkl            # 预计算 phyloP 评分
│   └── gnomad_af_cache.pkl         # 预计算 gnomAD v4 AF
│
├── skill/                          # 离线预测包
│   ├── SKILL.md                    # API 文档与快速开始
│   ├── scripts/
│   │   ├── predict.py              # 离线 CLI（v5.2，1,163 行，18 个函数）
│   │   └── generalization_eval.py  # LOGO-CV 评估脚本
│   └── references/
│       ├── acmg-criteria.md
│       └── troubleshooting.md
│
└── figures/
    ├── fig_logo_cv.png/svg         # LOGO-CV 泛化性图（16 基因）
    ├── fig_calibration_comparison.png/svg
    ├── fig_validation_holdout.png/svg
    ├── fig1_roc_comparison.png/svg
    ├── fig2_ablation.png/svg
    ├── figB3_plan_c_vision.png/svg
    └── shap_analysis_v4.png/svg
```

---

## 快速开始

### 1. 安装

```bash
git clone https://github.com/zja2004/SNV-judge.git
cd SNV-judge
pip install -r requirements.txt
```

### 2. API Key（全部可选）

```bash
# 评分模型
export EVO2_API_KEY="nvapi-..."      # NVIDIA NIM — 启用 Evo2 评分
export GENOS_API_KEY="sk-..."        # Stomics Cloud — 启用 Genos 评分

# AI 临床报告 LLM — 支持任意 OpenAI 兼容服务商
export LLM_API_KEY="sk-..."          # 你选择的服务商 API Key
export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export LLM_MODEL="qwen-plus"
```

> **不设置任何 Key 也能运行**：Evo2/Genos 用训练集中位数填充；LLM 报告标签页隐藏；其他功能（VEP、gnomAD、SHAP、ACMG 分级）全部正常。  
> 所有 LLM 设置均可**直接在应用内配置**，无需重启。

### 3. 启动应用

```bash
streamlit run app.py
# → http://localhost:8501
```

### 4. 离线 CLI

```bash
# 单变异预测
python skill/scripts/predict.py 17 7675088 C T /path/to/SNV-judge

# 启用本地 Genos embedding（无需 API Key）
python skill/scripts/predict.py 17 7675088 C T /path/to/SNV-judge \
    --genos-url https://xxx.ngrok-free.dev

# 批量 VCF 预测
python skill/scripts/predict.py --vcf variants.vcf /path/to/SNV-judge \
    --output results.csv
```

### 5. Python API

```python
from skill.scripts.predict import load_model_artifacts, predict_variant, print_shap_summary

artifacts = load_model_artifacts("/path/to/SNV-judge")
result = predict_variant("17", 7675088, "C", "T", artifacts=artifacts)

print(f"致病概率: {result['prob_pathogenic']:.1%}")
print(f"ACMG 分级: {result['acmg_tier']}")
print_shap_summary(result)
```

### 6. 示例变异

| 变异 | 基因 | 预期结果 |
|------|------|---------|
| chr17:7675088 C>T | TP53 R175H | 致病 |
| chr17:43063931 G>A | BRCA1 R1699W | 致病 |
| chr13:32332592 A>C | BRCA2 N372H | 良性 |

---

## 预计算评分（无需 API Key 即可重训练）

`data/` 目录包含训练 v5 所用的所有中间评分：

| 文件 | 内容 | 覆盖率 |
|------|------|--------|
| `scoring_ckpt.pkl` | Evo2-40B LLR + Genos-10B 评分 | 1,677/2,000（83.9%）|
| `vep_scores.pkl` | SIFT · PolyPhen-2 · AlphaMissense · CADD | 95–100% |
| `phylop_cache.pkl` | phyloP 保守性评分 | 1,922/2,000（96.1%）|
| `gnomad_af_cache.pkl` | gnomAD v4 等位基因频率 | 2,000 条 |
| `feature_matrix_v4.xlsx` | 完整特征矩阵 | 2,000 × 22 |

```bash
python train.py --use-cache   # 无需任何 API Key 即可重训练
```

**数据来源**（所有评分于 2026 年 3 月生成）：

| 评分 | 来源 | 版本 |
|------|------|------|
| Evo2 LLR | NVIDIA NIM `evo2-40b` | `health.api.nvidia.com` |
| Genos Score | Stomics Cloud `variant_predict` | `cloud.stomics.tech` |
| SIFT / PolyPhen-2 / AlphaMissense / CADD / phyloP | Ensembl VEP REST | GRCh38 / e113 |
| gnomAD AF | gnomAD GraphQL | gnomAD r4 |
| ClinVar 变异 | ClinVar FTP | 2026 年 3 月 |

---

## 方法

### 数据
- **来源**：ClinVar（2026 年 3 月），≥2 星审核状态的错义 SNV
- **训练集**：2,000 个变异（1,000P + 1,000B），547 个基因，每基因最多 10 个
- **基因组版本**：GRCh38

### 特征

| 特征 | 工具 | 方向 | 覆盖率 |
|------|------|------|--------|
| SIFT（取反）| SIFT4G | ↑ 越有害 | 94% |
| PolyPhen-2 | PolyPhen-2 | ↑ 越有害 | 88% |
| AlphaMissense | Google DeepMind | ↑ 越致病 | 87% |
| CADD Phred | CADD v1.7 | ↑ 越有害 | 100% |
| Evo2 LLR | Arc Institute / NVIDIA [1] | ↓ 越致病 | 84% |
| Genos Score | 浙江实验室 [2] | ↑ 越致病 | 84% |
| phyloP | Ensembl VEP / UCSC | ↑ 越保守 | 96% |
| gnomAD log-AF | gnomAD v4 | ↓ 越罕见越致病 | 37% |

缺失值用训练集中位数填充。

### 模型
- **算法**：XGBoost + LightGBM Stacking，逻辑回归元学习器
- **校准**：Isotonic Regression，在折外预测上拟合
- **评估**：5 折分层交叉验证
- **超参数**：XGBoost/LightGBM（n_estimators=300, max_depth=4, lr=0.05）

### Evo2 对数似然比
Evo2 是一个 400 亿参数的 DNA 语言模型，在 9.3 万亿核苷酸 token 上训练 [1]。LLR = log P(alt 上下文) − log P(ref 上下文)，使用 101 bp 窗口。负 LLR 表示替代等位基因在进化先验下可能性更低，提示功能破坏。

### Genos 致病性评分
Genos 是一个 1.2B–10B 参数的以人类为中心的基因组基础模型 [2]。通过 `variant_predict` 端点输入 GRCh38 坐标获取致病性概率。v5.2 新增本地 embedding 模式：通过 `/extract` 端点计算 REF/ALT 序列 embedding 的余弦距离。

---

## 局限性

- 训练集是 ClinVar 的 2,000 变异样本；在罕见/新型变异上的性能可能有所不同
- Evo2/Genos API 调用每个变异增加约 2–5 秒；大型 VCF 建议批量评分
- Genos 单独 AUROC 较低（0.65）；主要通过与经典特征的交互发挥作用
- gnomAD AF 在训练集中覆盖率为 37%；XGBoost 学会将缺失模式本身作为信号
- **MYH7（心肌病）是已知的弱基因** — LOGO-CV AUROC = 0.79，16 个测试基因中最低。三个原因：
  1. **增益功能机制**：MYH7 致病变异通过显性增益功能（肌球蛋白 ATPase 动力学改变）发挥作用，而非功能丧失。SIFT/PolyPhen-2 针对功能丧失型变异校准，系统性低估 MYH7 致病性。
  2. **中间序列评分**：MYH7 致病变异（肌球蛋白头部结构域，残基 167–931）的 SIFT（0.01–0.1）和 PolyPhen-2（0.5–0.85）评分往往与良性变异重叠。
  3. **测试集较小**：每基因仅 20 个变异，AUROC 置信区间约 ±0.10。
  - **计划改进（v6）**：添加蛋白结构特征（到 ATP 结合位点的距离、AlphaFold2 结构域间接触能）。
- **未经临床验证，不可用于临床诊断**

---

## 引用

- **[1] Evo2**：Brixi et al., *bioRxiv* 2025. https://arcinstitute.org/manuscripts/Evo2
- **[2] Genos**：浙江实验室, 2024. https://github.com/zhejianglab/Genos
- **AlphaMissense**：Cheng et al., *Science* 2023
- **CADD**：Kircher et al., *Nature Genetics* 2014；Rentzsch et al., *NAR* 2019
- **SIFT**：Ng & Henikoff, *Genome Research* 2001
- **PolyPhen-2**：Adzhubei et al., *Nature Methods* 2010
- **Ensembl VEP**：McLaren et al., *Genome Biology* 2016
- **gnomAD v4**：Karczewski et al., *Nature* 2020；Chen et al., *bioRxiv* 2023
- **ClinVar**：Landrum et al., *NAR* 2016

---

## 许可证

MIT。各评分工具有其自身许可证：AlphaMissense（CC-BY 4.0）、CADD（免费用于非商业用途）。

## 作者

周嘉诺（Junow Chow）
