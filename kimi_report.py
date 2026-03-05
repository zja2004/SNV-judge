"""
kimi_report.py — AI-powered clinical variant interpretation report
===================================================================
Uses Kimi (Moonshot AI) to generate structured ACMG-style clinical
interpretation reports from SNV-judge v4 prediction results.

The module acts as the "reasoning layer" of the SNV-judge agent:
  Tool outputs (VEP, Evo2, Genos, gnomAD, SHAP) → Kimi LLM → Clinical report

Usage:
    from kimi_report import generate_report_stream, generate_report

    # Streaming (for Streamlit st.write_stream)
    for chunk in generate_report_stream(variant_info, scores, shap_vals, cal_prob):
        print(chunk, end="", flush=True)

    # Non-streaming (returns full string)
    report = generate_report(variant_info, scores, shap_vals, cal_prob)
"""

import os
import math
import numpy as np
from typing import Generator

# Kimi API — OpenAI-compatible
KIMI_API_KEY  = os.environ.get("KIMI_API_KEY", "")
KIMI_BASE_URL = "https://api.moonshot.cn/v1"
KIMI_MODEL    = "moonshot-v1-32k"

# ACMG classification thresholds (matching app.py)
PATHOGENIC_THRESH  = 0.70
BENIGN_THRESH      = 0.40

# Feature display names and their ACMG relevance
FEATURE_ACMG = {
    "SIFT (inv)":      ("PP3/BP4", "sequence conservation (SIFT)"),
    "PolyPhen-2":      ("PP3/BP4", "structural impact (PolyPhen-2)"),
    "AlphaMissense":   ("PP3/BP4", "structure-based pathogenicity (AlphaMissense)"),
    "CADD Phred":      ("PP3/BP4", "combined annotation depletion (CADD)"),
    "Evo2 LLR":        ("PP3/BP4", "evolutionary language model log-likelihood (Evo2-40B)"),
    "Genos Score":     ("PP3/BP4", "human-centric genomic foundation model (Genos-10B)"),
    "phyloP":          ("PP3/BP4", "vertebrate phylogenetic conservation (phyloP)"),
    "gnomAD log-AF":   ("BA1/PM2", "population allele frequency (gnomAD v4)"),
}

FEATURE_LABELS = [
    "SIFT (inv)", "PolyPhen-2", "AlphaMissense", "CADD Phred",
    "Evo2 LLR", "Genos Score", "phyloP", "gnomAD log-AF",
]


def _get_kimi_client():
    """Return an OpenAI client pointed at Kimi API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required: pip install openai>=1.0")
    key = KIMI_API_KEY
    if not key:
        raise ValueError(
            "KIMI_API_KEY not set. "
            "Set environment variable: export KIMI_API_KEY='sk-...'"
        )
    return OpenAI(api_key=key, base_url=KIMI_BASE_URL)


def _format_gnomad_af(log_af: float | None) -> str:
    """Convert log10 AF back to human-readable frequency string."""
    if log_af is None or (isinstance(log_af, float) and math.isnan(log_af)):
        return "未在gnomAD v4中检索到（可能为新发变异，PM2证据）"
    af = 10 ** log_af
    if af < 1e-7:
        return f"极罕见 / 未见（AF < 1×10⁻⁷，PM2强证据）"
    elif af < 0.001:
        return f"罕见（AF ≈ {af:.2e}，PM2支持证据）"
    elif af < 0.01:
        return f"低频（AF ≈ {af:.3f}）"
    elif af < 0.05:
        return f"中等频率（AF ≈ {af:.3f}）"
    else:
        return f"常见变异（AF ≈ {af:.3f}，BA1强良性证据）"


def _format_phylop(phylop: float | None) -> str:
    if phylop is None or (isinstance(phylop, float) and math.isnan(phylop)):
        return "无数据"
    if phylop > 2.0:
        return f"{phylop:.2f}（高度保守，进化约束强）"
    elif phylop > 0.5:
        return f"{phylop:.2f}（中度保守）"
    elif phylop > -0.5:
        return f"{phylop:.2f}（中性进化）"
    else:
        return f"{phylop:.2f}（进化加速，约束弱）"


def _format_evo2(evo2_llr: float | None) -> str:
    if evo2_llr is None or (isinstance(evo2_llr, float) and math.isnan(evo2_llr)):
        return "未获取（需要Evo2 API密钥）"
    if evo2_llr < -1.0:
        return f"{evo2_llr:.3f}（显著负值，替代等位基因进化上不利，支持致病性）"
    elif evo2_llr < -0.3:
        return f"{evo2_llr:.3f}（轻度负值，替代等位基因略不利）"
    elif evo2_llr < 0.3:
        return f"{evo2_llr:.3f}（接近中性）"
    else:
        return f"{evo2_llr:.3f}（正值，替代等位基因进化上可接受）"


def _build_evidence_context(
    variant_info: dict,
    scores: dict,
    shap_vals: np.ndarray | list,
    cal_prob: float,
    evo2_llr: float = float("nan"),
    genos_path: float = float("nan"),
    gnomad_log_af: float = float("nan"),
    model_ver: str = "v4",
) -> str:
    """Build a structured evidence summary string to pass to Kimi."""

    # Classification
    if cal_prob >= PATHOGENIC_THRESH:
        classification = f"可能致病（Likely Pathogenic，概率 {cal_prob:.1%}）"
    elif cal_prob >= BENIGN_THRESH:
        classification = f"意义不明确（VUS，概率 {cal_prob:.1%}）"
    else:
        classification = f"可能良性（Likely Benign，概率 {cal_prob:.1%}）"

    # SHAP top contributors
    labels = FEATURE_LABELS[:len(shap_vals)]
    shap_pairs = sorted(zip(labels, shap_vals), key=lambda x: abs(x[1]), reverse=True)
    shap_lines = []
    for feat, sv in shap_pairs:
        direction = "→致病" if sv > 0 else "→良性"
        shap_lines.append(f"  • {feat}: SHAP={sv:+.4f} {direction}")

    # gnomAD
    gnomad_str = _format_gnomad_af(gnomad_log_af)

    # phyloP
    phylop_raw = scores.get("phylop")
    phylop_val = float(phylop_raw) if phylop_raw is not None else (
        gnomad_log_af if False else float("nan")  # keep nan if missing
    )
    phylop_str = _format_phylop(phylop_val if phylop_raw is not None else None)

    # Evo2
    evo2_str = _format_evo2(evo2_llr)

    # Genos
    if genos_path is None or (isinstance(genos_path, float) and math.isnan(genos_path)):
        genos_str = "未获取（需要Genos API密钥）"
    else:
        genos_str = f"{genos_path:.4f}（{'高' if genos_path > 0.7 else '中' if genos_path > 0.4 else '低'}致病性评分）"

    context = f"""
=== 变异基本信息 ===
变异位置: chr{variant_info.get('chrom', '?')}:{variant_info.get('pos', '?')} {variant_info.get('ref', '?')}>{variant_info.get('alt', '?')} (GRCh38)
基因: {scores.get('gene', '未知')}
转录本: {scores.get('transcript', '未知')}
蛋白变化: {scores.get('hgvsp', '未知')}
变异类型: {scores.get('consequence', '未知')}

=== SNV-judge v4 集成预测结果 ===
校准概率: {cal_prob:.4f} ({cal_prob:.1%})
综合分类建议: {classification}
使用模型版本: {model_ver}（8特征集成：4个经典工具 + Evo2 + Genos + phyloP + gnomAD AF）

=== 各工具评分详情 ===
1. SIFT (inv): {f"{1 - scores['sift_score']:.4f} ({scores.get('sift_pred', '')})" if scores.get('sift_score') is not None else "无数据"}
   解读: SIFT > 0.5 提示氨基酸替换有害（PP3证据）

2. PolyPhen-2: {f"{scores['polyphen_score']:.4f} ({scores.get('polyphen_pred', '')})" if scores.get('polyphen_score') is not None else "无数据"}
   解读: > 0.85 = probably_damaging（PP3证据）

3. AlphaMissense: {f"{scores['am_pathogenicity']:.4f} ({scores.get('am_class', '')})" if scores.get('am_pathogenicity') is not None else "无数据"}
   解读: > 0.564 = likely_pathogenic（PP3证据）

4. CADD Phred: {f"{scores['cadd_phred']:.2f}" if scores.get('cadd_phred') is not None else "无数据"}
   解读: > 20 = 前1%最有害变异（PP3证据）；> 30 = 前0.1%

5. Evo2-40B LLR（基因组语言模型）: {evo2_str}
   解读: 负值表示替代等位基因在进化语言模型中概率更低，支持功能损害

6. Genos-10B（人类基因组基础模型）: {genos_str}
   解读: 基于人类基因组训练的致病性专项评分

7. phyloP保守性评分: {phylop_str}
   解读: 高保守位点突变更可能有害（PP3证据）

8. gnomAD v4等位基因频率: {gnomad_str}
   解读: 遵循ACMG BA1（AF>5%=良性）/ PM2（极罕见=致病支持）规则

=== SHAP特征贡献分析（按重要性排序）===
{chr(10).join(shap_lines)}

=== AlphaMissense分类 ===
{scores.get('am_class', '未知')}（AlphaMissense官方分类）
""".strip()

    return context


def _build_system_prompt() -> str:
    return """你是一位专业的临床遗传学家和基因组变异解读专家，具备以下专业背景：
- 熟悉ACMG/AMP 2015变异分类指南（Richards et al., Genetics in Medicine 2015）
- 精通基因组基础模型（Evo2、Genos）和经典变异注释工具（SIFT、PolyPhen-2、AlphaMissense、CADD）
- 了解gnomAD人群频率数据库和phyloP进化保守性评分的临床意义
- 能够整合多维度证据，给出结构化的变异解读报告

你的任务是基于SNV-judge v4智能体系统提供的多源证据，生成一份专业的临床变异解读报告。

报告要求：
1. 使用中文撰写，专业术语保留英文
2. 结构清晰，分节呈现
3. 明确引用ACMG证据条目（如PP3、PM2、BA1等）
4. 对每个工具的结果给出简洁解读
5. 最终给出综合分类建议和临床意义说明
6. 语气专业但易于理解，适合临床遗传咨询场景
7. 在报告末尾注明：本报告由AI辅助生成，仅供参考，不构成临床诊断依据

报告格式（严格按照以下结构）：
## 变异解读报告

### 一、变异基本信息
[变异坐标、基因、蛋白变化等]

### 二、集成预测结果
[SNV-judge v4的综合评分和分类]

### 三、多维度证据分析
#### 3.1 经典注释工具证据（PP3/BP4）
[SIFT、PolyPhen-2、AlphaMissense、CADD的结果和ACMG证据强度]

#### 3.2 基因组基础模型证据
[Evo2-40B和Genos-10B的结果解读]

#### 3.3 进化保守性证据（PP3/BP4）
[phyloP评分解读]

#### 3.4 人群频率证据（BA1/PM2）
[gnomAD AF解读和ACMG证据条目]

#### 3.5 SHAP特征贡献分析
[哪些特征对本次预测贡献最大，及其生物学意义]

### 四、综合分类建议
[基于ACMG框架的综合分类：致病/可能致病/VUS/可能良性/良性]
[支持该分类的主要证据汇总]

### 五、临床意义与建议
[该变异的临床意义、建议的后续验证步骤]

### 六、局限性说明
[本次分析的局限性，包括数据缺失、模型局限等]

---
*本报告由SNV-judge v4智能体系统（Kimi AI辅助）自动生成，仅供科研参考，不构成临床诊断依据。*"""


def generate_report_stream(
    variant_info: dict,
    scores: dict,
    shap_vals,
    cal_prob: float,
    evo2_llr: float = float("nan"),
    genos_path: float = float("nan"),
    gnomad_log_af: float = float("nan"),
    model_ver: str = "v4",
) -> Generator[str, None, None]:
    """
    Stream Kimi-generated clinical report as text chunks.
    Suitable for Streamlit st.write_stream().

    Yields:
        str: text chunks from the LLM stream
    """
    client = _get_kimi_client()
    evidence = _build_evidence_context(
        variant_info, scores, shap_vals, cal_prob,
        evo2_llr, genos_path, gnomad_log_af, model_ver,
    )

    user_msg = f"""请基于以下SNV-judge v4智能体系统的多源证据，生成一份专业的临床变异解读报告：

{evidence}

请严格按照系统提示中的报告格式生成报告，确保每个证据条目都有明确的ACMG证据强度标注。"""

    stream = client.chat.completions.create(
        model=KIMI_MODEL,
        messages=[
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.3,   # lower temp for clinical consistency
        max_completion_tokens=2048,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


def generate_report(
    variant_info: dict,
    scores: dict,
    shap_vals,
    cal_prob: float,
    evo2_llr: float = float("nan"),
    genos_path: float = float("nan"),
    gnomad_log_af: float = float("nan"),
    model_ver: str = "v4",
) -> str:
    """
    Non-streaming version: returns the full report as a string.
    Useful for batch processing or testing.
    """
    return "".join(generate_report_stream(
        variant_info, scores, shap_vals, cal_prob,
        evo2_llr, genos_path, gnomad_log_af, model_ver,
    ))


def check_kimi_available() -> tuple[bool, str]:
    """
    Check if Kimi API is available and the key is valid.
    Returns (is_available: bool, message: str)
    """
    key = KIMI_API_KEY
    if not key:
        return False, "未设置 KIMI_API_KEY 环境变量"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=KIMI_BASE_URL)
        # Lightweight test: list models
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        if KIMI_MODEL in model_ids or any("moonshot" in m for m in model_ids):
            return True, f"Kimi API 连接正常（模型: {KIMI_MODEL}）"
        else:
            return True, f"Kimi API 连接正常（可用模型: {', '.join(model_ids[:3])}）"
    except Exception as e:
        return False, f"Kimi API 连接失败: {str(e)[:100]}"
