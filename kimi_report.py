"""
kimi_report.py — Universal LLM Clinical Variant Interpretation Report
======================================================================
Supports any OpenAI-compatible API endpoint (Moonshot/Kimi, Alibaba DashScope,
OpenAI, DeepSeek, Qwen, etc.).

The module acts as the "reasoning layer" of the SNV-judge agent:
  Tool outputs (VEP, Evo2, Genos, gnomAD, SHAP) → LLM → Clinical report

Configuration (in priority order):
  1. Runtime arguments passed to generate_report_stream() / generate_report()
  2. Environment variables: LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
  3. Legacy env vars: KIMI_API_KEY (treated as LLM_API_KEY with Moonshot base URL)

Usage:
    from kimi_report import generate_report_stream, generate_report, list_models

    # List available models for a given endpoint
    models = list_models("https://dashscope.aliyuncs.com/compatible-mode/v1", "sk-xxx")

    # Streaming (for Streamlit st.write_stream)
    for chunk in generate_report_stream(
        variant_info, scores, shap_vals, cal_prob,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-xxx",
        model="qwen-plus",
    ):
        print(chunk, end="", flush=True)
"""

import os
import math
import numpy as np
from typing import Generator

# ── Default configuration (env vars or legacy Kimi fallback) ──────────────
_ENV_API_KEY  = os.environ.get("LLM_API_KEY") or os.environ.get("KIMI_API_KEY", "")
_ENV_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.moonshot.cn/v1")
_ENV_MODEL    = os.environ.get("LLM_MODEL", "moonshot-v1-32k")

# Legacy aliases kept for backward compatibility
KIMI_API_KEY  = _ENV_API_KEY
KIMI_BASE_URL = _ENV_BASE_URL
KIMI_MODEL    = _ENV_MODEL

# ACMG classification thresholds (matching app.py)
PATHOGENIC_THRESH = 0.70
BENIGN_THRESH     = 0.40

FEATURE_LABELS = [
    "SIFT (inv)", "PolyPhen-2", "AlphaMissense", "CADD Phred",
    "Evo2 LLR", "Genos Score", "phyloP", "gnomAD log-AF",
]

FEATURE_ACMG = {
    "SIFT (inv)":    ("PP3/BP4", "sequence conservation (SIFT)"),
    "PolyPhen-2":    ("PP3/BP4", "structural impact (PolyPhen-2)"),
    "AlphaMissense": ("PP3/BP4", "structure-based pathogenicity (AlphaMissense)"),
    "CADD Phred":    ("PP3/BP4", "combined annotation depletion (CADD)"),
    "Evo2 LLR":      ("PP3/BP4", "evolutionary language model log-likelihood (Evo2-40B)"),
    "Genos Score":   ("PP3/BP4", "human-centric genomic foundation model (Genos-10B)"),
    "phyloP":        ("PP3/BP4", "vertebrate phylogenetic conservation (phyloP)"),
    "gnomAD log-AF": ("BA1/PM2", "population allele frequency (gnomAD v4)"),
}

# ── Known provider presets ─────────────────────────────────────────────────
PROVIDER_PRESETS = {
    "Moonshot (Kimi)":   "https://api.moonshot.cn/v1",
    "Alibaba DashScope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "OpenAI":            "https://api.openai.com/v1",
    "DeepSeek":          "https://api.deepseek.com/v1",
    "自定义 / Custom":   "",
}


# ── Model listing ──────────────────────────────────────────────────────────

def list_models(base_url: str, api_key: str) -> list[str]:
    """
    Fetch available model IDs from any OpenAI-compatible endpoint.

    Returns a sorted list of model ID strings, or raises on failure.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required: pip install openai>=1.0")

    client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"))
    models = client.models.list()
    ids = sorted({m.id for m in models.data})
    return ids


def check_available(
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> tuple[bool, str]:
    """
    Check if the LLM endpoint is reachable and the key is valid.
    Falls back to env-var defaults when arguments are None.

    Returns:
        (is_available: bool, message: str)
    """
    url = base_url or _ENV_BASE_URL
    key = api_key or _ENV_API_KEY
    mdl = model or _ENV_MODEL

    if not key:
        return False, "未设置 API Key（LLM_API_KEY 环境变量或应用内输入）"
    try:
        ids = list_models(url, key)
        if mdl in ids:
            return True, f"连接正常（模型: {mdl}，共 {len(ids)} 个可用模型）"
        else:
            return True, f"连接正常（共 {len(ids)} 个可用模型，当前选择: {mdl}）"
    except Exception as e:
        return False, f"连接失败: {str(e)[:120]}"


# Legacy alias
def check_kimi_available() -> tuple[bool, str]:
    return check_available()


# ── Evidence formatting helpers ────────────────────────────────────────────

def _format_gnomad_af(log_af) -> str:
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


def _format_phylop(phylop) -> str:
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


def _format_evo2(evo2_llr) -> str:
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
    shap_vals,
    cal_prob: float,
    evo2_llr: float = float("nan"),
    genos_path: float = float("nan"),
    gnomad_log_af: float = float("nan"),
    model_ver: str = "v5",
) -> str:
    """Build structured evidence summary string to pass to the LLM."""
    if cal_prob >= PATHOGENIC_THRESH:
        classification = f"可能致病（Likely Pathogenic，概率 {cal_prob:.1%}）"
    elif cal_prob >= BENIGN_THRESH:
        classification = f"意义不明确（VUS，概率 {cal_prob:.1%}）"
    else:
        classification = f"可能良性（Likely Benign，概率 {cal_prob:.1%}）"

    labels = FEATURE_LABELS[:len(shap_vals)]
    shap_pairs = sorted(zip(labels, shap_vals), key=lambda x: abs(x[1]), reverse=True)
    shap_lines = [
        f"  • {feat}: SHAP={sv:+.4f} {'→致病' if sv > 0 else '→良性'}"
        for feat, sv in shap_pairs
    ]

    phylop_raw = scores.get("phylop")
    phylop_str = _format_phylop(float(phylop_raw) if phylop_raw is not None else None)

    genos_str = (
        "未获取（需要Genos API密钥）"
        if genos_path is None or (isinstance(genos_path, float) and math.isnan(genos_path))
        else f"{genos_path:.4f}（{'高' if genos_path > 0.7 else '中' if genos_path > 0.4 else '低'}致病性评分）"
    )

    return f"""
=== 变异基本信息 ===
变异位置: chr{variant_info.get('chrom','?')}:{variant_info.get('pos','?')} {variant_info.get('ref','?')}>{variant_info.get('alt','?')} (GRCh38)
基因: {scores.get('gene','未知')}
转录本: {scores.get('transcript','未知')}
蛋白变化: {scores.get('hgvsp','未知')}
变异类型: {scores.get('consequence','未知')}

=== SNV-judge {model_ver} 集成预测结果 ===
校准概率: {cal_prob:.4f} ({cal_prob:.1%})
综合分类建议: {classification}
使用模型版本: {model_ver}（8特征集成：4个经典工具 + Evo2 + Genos + phyloP + gnomAD AF）

=== 各工具评分详情 ===
1. SIFT (inv): {f"{1 - scores['sift_score']:.4f} ({scores.get('sift_pred','')})" if scores.get('sift_score') is not None else "无数据"}
2. PolyPhen-2: {f"{scores['polyphen_score']:.4f} ({scores.get('polyphen_pred','')})" if scores.get('polyphen_score') is not None else "无数据"}
3. AlphaMissense: {f"{scores['am_pathogenicity']:.4f} ({scores.get('am_class','')})" if scores.get('am_pathogenicity') is not None else "无数据"}
4. CADD Phred: {f"{scores['cadd_phred']:.2f}" if scores.get('cadd_phred') is not None else "无数据"}
5. Evo2-40B LLR: {_format_evo2(evo2_llr)}
6. Genos-10B: {genos_str}
7. phyloP: {phylop_str}
8. gnomAD v4 AF: {_format_gnomad_af(gnomad_log_af)}

=== SHAP特征贡献分析（按重要性排序）===
{chr(10).join(shap_lines)}

=== AlphaMissense分类 ===
{scores.get('am_class','未知')}
""".strip()


def _build_system_prompt(template: str = "chinese") -> str:
    if template == "english":
        return (
            "You are an expert clinical geneticist. Generate a professional clinical variant "
            "interpretation report in English based on multi-source evidence from SNV-judge v5.\n\n"
            "Requirements:\n"
            "1. Structured sections with clear headings\n"
            "2. Explicitly cite ACMG evidence criteria (PP3, PM2, BA1, etc.)\n"
            "3. Professional tone suitable for clinical genetics consultation\n"
            "4. End with: *AI-assisted report, for research reference only.*\n\n"
            "Report format:\n"
            "## Variant Interpretation Report\n"
            "### I. Variant Information\n"
            "### II. Integrated Prediction\n"
            "### III. Multi-dimensional Evidence Analysis\n"
            "#### 3.1 Classical Tools (PP3/BP4)\n"
            "#### 3.2 Genomic Foundation Models\n"
            "#### 3.3 Evolutionary Conservation (PP3/BP4)\n"
            "#### 3.4 Population Frequency (BA1/PM2)\n"
            "#### 3.5 SHAP Feature Contributions\n"
            "### IV. ACMG Classification Summary\n"
            "### V. Clinical Significance and Recommendations\n"
            "### VI. Limitations\n"
            "---\n*AI-assisted report. Not for clinical diagnosis.*"
        )
    elif template == "summary":
        return (
            "你是一位临床遗传学专家。基于SNV-judge v5系统的多源证据，生成一份简洁的变异解读摘要（3-5句话）。\n"
            "第一句：变异信息和预测结果；第二句：最关键证据；第三句：人群频率；第四句：综合ACMG建议。\n"
            "直接输出摘要段落，无需标题。末尾注明「AI辅助生成，仅供参考」。"
        )
    else:  # chinese (default)
        return (
            "你是一位专业的临床遗传学家和基因组变异解读专家，熟悉ACMG/AMP 2015变异分类指南。\n"
            "基于SNV-judge v5智能体系统提供的多源证据，生成一份专业的临床变异解读报告。\n\n"
            "要求：\n"
            "1. 使用中文，专业术语保留英文\n"
            "2. 明确引用ACMG证据条目（PP3、PM2、BA1等）\n"
            "3. 语气专业，适合临床遗传咨询场景\n"
            "4. 末尾注明：本报告由AI辅助生成，仅供参考，不构成临床诊断依据\n\n"
            "报告格式：\n"
            "## 变异解读报告\n"
            "### 一、变异基本信息\n"
            "### 二、集成预测结果\n"
            "### 三、多维度证据分析\n"
            "#### 3.1 经典注释工具证据（PP3/BP4）\n"
            "#### 3.2 基因组基础模型证据\n"
            "#### 3.3 进化保守性证据（PP3/BP4）\n"
            "#### 3.4 人群频率证据（BA1/PM2）\n"
            "#### 3.5 SHAP特征贡献分析\n"
            "### 四、综合分类建议\n"
            "### 五、临床意义与建议\n"
            "### 六、局限性说明\n"
            "---\n*本报告由SNV-judge v5智能体系统自动生成，仅供科研参考，不构成临床诊断依据。*"
        )


# ── Core generation functions ──────────────────────────────────────────────

def generate_report_stream(
    variant_info: dict,
    scores: dict,
    shap_vals,
    cal_prob: float,
    evo2_llr: float = float("nan"),
    genos_path: float = float("nan"),
    gnomad_log_af: float = float("nan"),
    model_ver: str = "v5",
    template: str = "chinese",
    # LLM backend — if None, falls back to env-var defaults
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> Generator[str, None, None]:
    """
    Stream LLM-generated clinical report as text chunks.
    Compatible with Streamlit st.write_stream().

    Args:
        base_url:  OpenAI-compatible API base URL (e.g. DashScope, Moonshot, OpenAI)
        api_key:   API key for the chosen provider
        model:     Model ID to use (e.g. "qwen-plus", "moonshot-v1-32k", "gpt-4o")
        template:  "chinese" | "english" | "summary"

    Yields:
        str: text chunks from the LLM stream
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required: pip install openai>=1.0")

    _url = (base_url or _ENV_BASE_URL).rstrip("/")
    _key = api_key or _ENV_API_KEY
    _mdl = model or _ENV_MODEL

    if not _key:
        raise ValueError(
            "未设置 API Key。请在应用设置面板中输入，或设置环境变量 LLM_API_KEY。"
        )

    client = OpenAI(api_key=_key, base_url=_url)
    evidence = _build_evidence_context(
        variant_info, scores, shap_vals, cal_prob,
        evo2_llr, genos_path, gnomad_log_af, model_ver,
    )

    if template == "english":
        user_msg = (
            "Please generate a professional clinical variant interpretation report "
            "based on the following multi-source evidence from the SNV-judge v5 system:\n\n"
            f"{evidence}\n\n"
            "Follow the report format in the system prompt strictly."
        )
    elif template == "summary":
        user_msg = (
            f"请基于以下SNV-judge v5系统的多源证据，生成一份简洁的变异解读摘要（3-5句话）：\n\n{evidence}"
        )
    else:
        user_msg = (
            "请基于以下SNV-judge v5智能体系统的多源证据，生成一份专业的临床变异解读报告：\n\n"
            f"{evidence}\n\n"
            "请严格按照系统提示中的报告格式生成报告，确保每个证据条目都有明确的ACMG证据强度标注。"
        )

    max_tokens = 512 if template == "summary" else 2048

    stream = client.chat.completions.create(
        model=_mdl,
        messages=[
            {"role": "system", "content": _build_system_prompt(template)},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=max_tokens,
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
    model_ver: str = "v5",
    template: str = "chinese",
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> str:
    """Non-streaming version. Returns the full report as a string."""
    return "".join(generate_report_stream(
        variant_info, scores, shap_vals, cal_prob,
        evo2_llr, genos_path, gnomad_log_af, model_ver,
        template=template,
        base_url=base_url, api_key=api_key, model=model,
    ))
