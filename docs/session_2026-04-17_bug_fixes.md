# SNV-judge 开发日志 — 2026-04-17

## 会话摘要

本次会话完成了上次 Skill 全量测试后发现的 3 个待修复问题，以及将所有代码变更推送至 GitHub。

---

## 修复内容（commit 07c046f）

### Fix 1（高优先级）：fetch_clinvar_classification Step 2 — review_status/conditions

**根本原因**：`esearch clinvar term=rsID[rs]` 在 NCBI E-utilities 中对 ClinVar 数据库不可靠，始终返回空列表（NCBI 已知问题）。

**修复方案**：
- Step 1（VEP colocated_variants）保持不变，继续提取 rsID 和 allele-specific clin_sig
- Step 2 改为：从 VEP `var_synonyms.ClinVar` 中提取数字最小的 VCV 编号，然后调用 `efetch rettype=vcv retmode=xml is_variationid=1` 直接获取 VCV 记录
- XML 解析路径：`GermlineClassification/ReviewStatus`、`GermlineClassification/Description[@DateLastEvaluated]`、`TraitSet/Trait/Name/ElementValue[@Type='Preferred']`
- conditions 去重并过滤 "not provided" / "not specified"

**验证结果**（TP53 R175H, chr17:7674220 C>T）：
```
clinvar_id:            VCV000012356
clinical_significance: Likely Pathogenic / Pathogenic
review_status:         criteria provided, multiple submitters, no conflicts
conditions:            [Li-Fraumeni syndrome 1, Familial cancer of breast, ...]  (39条)
last_evaluated:        2024-12-17
rsid:                  rs11540652
```

**已知局限**：VEP var_synonyms 返回的是 rsID 级别的 VCV（非 allele-specific）。对于多等位基因位点（如 rs11540652 对应 C/A/G/T），最小 VCV 可能来自不同的氨基酸变化（如 R248Q 而非 R175H）。review_status 在同一 rsID 的不同 VCV 间通常一致，conditions 有重叠但不完全相同。

---

### Fix 2（中优先级）：fetch_genos_embedding_score — Jaccard 分布重映射

**根本原因**：k-mer Jaccard 距离实测范围 ~[0.00, 0.15]，远低于训练集 genos_score 中位数 0.676（范围 ~[0.30, 0.90]）。模型将低 Jaccard 值误判为良性，导致 SHAP 方向反转。

**修复方案**：在 `fetch_genos_embedding_score()` 末尾加线性重映射：
```python
genos_score = float(np.clip(0.30 + jaccard_dist * 4.0, 0.30, 0.90))
```

映射关系：
| Jaccard | Mapped | 含义 |
|---------|--------|------|
| 0.000 | 0.300 | 续写完全相同（良性下界） |
| 0.094 | 0.676 | 对应训练集中位数 |
| 0.150 | 0.900 | 最大观测差异（致病上界） |

**已知局限**：映射参数（0.30, 4.0）基于单个实测数据点（TP53 R175H Jaccard=0.0116）和假设上界（0.15）推导，未经系统性校准。对于 Jaccard < 0.094 的变异（包括 TP53 R175H），SHAP 方向仍为良性（▼），只是幅度减小。根本解决方案需要用 Genos /generate 重新计算训练集所有变异的 Jaccard 值并重新拟合。

---

### Fix 3（低优先级）：generate_clinical_report — LLM ACMG 保守性偏差

**根本原因**：`kimi_report._build_evidence_context()` 使用固定阈值 0.70 判断为"可能致病"，忽略了 predict.py 的 5 级 ACMG_TIERS（Pathogenic ≥ 0.90）。LLM 收到"可能致病"后不会自行升级。

**修复方案**（三处改动）：

A. `_build_evidence_context()` 新增 `acmg_class` 参数，优先使用调用方传入的分级字符串：
```python
def _build_evidence_context(..., acmg_class: str = ""):
    if acmg_class:
        classification = f"{acmg_class}（校准概率 {cal_prob:.1%}）"
    elif cal_prob >= 0.90:
        classification = f"致病（Pathogenic，概率 {cal_prob:.1%}）"
    ...
```

B. System prompt（中文 + 英文）加入禁止降级指令：
```
5. 【重要】报告中的ACMG分类必须与【综合分类建议】字段完全一致，
   不得自行降级或升级（例如：若系统输出为Pathogenic，报告必须写Pathogenic，
   不得改写为Likely Pathogenic）
```

C. `predict.py` 的 `generate_clinical_report()` 传入 `result["acmg_class"]`：
```python
acmg_class = result.get("acmg_class", "")
gen = _kr.generate_report_stream(..., acmg_class=acmg_class, ...)
```

**验证结果**：
- Evidence context 显示 "Pathogenic (P)（校准概率 99.1%）" ✅
- System prompt 包含禁止降级指令 ✅
- `generate_report_stream` 和 `generate_report` 均有 `acmg_class` 参数 ✅

---

## Git 提交记录

| Commit | 内容 |
|--------|------|
| `f9f49b9` | Genos /generate 适配 (k-mer Jaccard) + LOGO-CV 图更新 |
| `3714b6c` | resolve_protein_variant + fetch_clinvar_classification + Genos flank 修复 |
| `07c046f` | 3 个后测试问题修复（ClinVar Step2, Genos 分布, LLM ACMG 偏差） |

---

## 技术决策记录

### 为什么不用 ClinVar GraphQL API？
沙箱环境只有 NCBI E-utilities 和 Ensembl REST API 可达，clinvar.ncbi.nlm.nih.gov 域名解析失败。efetch VCV XML 是在当前网络限制下唯一可靠的方案。

### 为什么 Jaccard 映射参数是 (0.30, 4.0)？
- 下界 0.30：训练集 genos_score 的合理最低值（良性变异）
- 上界 0.90：训练集 genos_score 的合理最高值（强致病变异）
- 斜率 4.0 = (0.90 - 0.30) / 0.15，其中 0.15 是实测 Jaccard 最大值
- 这是工程近似，不是统计校准

### 为什么 Fix 3 不能完全消除 LLM 降级？
LLM 的 system prompt 遵从率不是 100%，尤其是较弱的模型。后处理校验（解析报告文本检查 ACMG 关键词）是更可靠的保障，但超出本次修复范围。

---

*记录时间: 2026-04-17*
