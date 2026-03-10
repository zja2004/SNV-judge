"""
agent/chat_ui.py — SNV-judge Plan C: Streamlit Agent Chat Tab
==============================================================
Renders the "🤖 Agent Chat" tab inside the main app.py.
Import and call render_agent_chat_tab() from app.py.

Features:
  - Chat bubble UI (st.chat_message)
  - Tool call trace (collapsible expander per turn)
  - Patient profile sidebar panel
  - VCF file upload → batch analysis trigger
  - Session memory management (clear / switch session)
  - Quick-action buttons for common queries
"""

import json
import os
import streamlit as st
from pathlib import Path

# ── Lazy-load agent (cached per Streamlit session) ─────────────────────────
@st.cache_resource(show_spinner="Loading SNV-judge Agent…")
def _get_executor():
    """Load and cache the AgentExecutor (shared across reruns)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agent.agent import create_agent
    return create_agent()


def _run(user_input: str, session_id: str) -> dict:
    """Run agent with the given input and session."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agent.agent import run_agent
    executor = _get_executor()
    return run_agent(executor, user_input, session_id=session_id)


def _load_patient_profiles() -> dict:
    """Load patient profiles from JSON file."""
    db_path = Path(__file__).parent / "patient_profiles.json"
    if db_path.exists():
        with open(db_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ── ACMG badge HTML ────────────────────────────────────────────────────────
_ACMG_COLORS = {
    "Pathogenic (P)":        ("#c0392b", "#fff"),
    "Likely Pathogenic (LP)":("#e74c3c", "#fff"),
    "VUS":                   ("#e67e22", "#fff"),
    "Likely Benign (LB)":    ("#27ae60", "#fff"),
    "Benign (B)":            ("#1a7a4a", "#fff"),
}

def _acmg_badge(label: str) -> str:
    bg, fg = _ACMG_COLORS.get(label, ("#888", "#fff"))
    return (f'<span style="background:{bg};color:{fg};padding:2px 8px;'
            f'border-radius:8px;font-size:12px;font-weight:bold;">{label}</span>')


# ── Tool call display ──────────────────────────────────────────────────────
_TOOL_ICONS = {
    "vep_annotate":        "🔬",
    "gnomad_frequency":    "🌍",
    "evo2_score":          "🧬",
    "genos_score":         "🤖",
    "snv_judge_predict":   "⚡",
    "search_clinvar":      "📚",
    "save_patient_variant":"💾",
    "get_patient_history": "📋",
    "batch_vcf_analyze":   "📂",
}

def _render_tool_calls(tool_calls: list):
    """Render tool call trace in a collapsible expander."""
    if not tool_calls:
        return
    with st.expander(f"🔧 工具调用记录 ({len(tool_calls)} 次)", expanded=False):
        for i, tc in enumerate(tool_calls, 1):
            icon = _TOOL_ICONS.get(tc["tool"], "🔧")
            st.markdown(f"**{i}. {icon} `{tc['tool']}`**")
            # Input
            inp = tc.get("input", "")
            if isinstance(inp, dict):
                inp_str = json.dumps(inp, ensure_ascii=False, indent=2)
            else:
                inp_str = str(inp)
            st.code(inp_str[:300] + ("…" if len(inp_str) > 300 else ""),
                    language="json")
            # Output preview
            out = tc.get("output", "")
            try:
                out_parsed = json.loads(out) if isinstance(out, str) else out
                out_str = json.dumps(out_parsed, ensure_ascii=False, indent=2)
            except Exception:
                out_str = str(out)
            st.code(out_str[:400] + ("…" if len(out_str) > 400 else ""),
                    language="json")
            if i < len(tool_calls):
                st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════
# MAIN RENDER FUNCTION
# ══════════════════════════════════════════════════════════════════════════
def render_agent_chat_tab():
    """Render the full Agent Chat tab. Call this from app.py."""

    # ── Session state init ─────────────────────────────────────────────
    if "agent_messages" not in st.session_state:
        st.session_state["agent_messages"] = []   # list of {role, content, tool_calls}
    if "agent_session_id" not in st.session_state:
        st.session_state["agent_session_id"] = "default"

    session_id = st.session_state["agent_session_id"]

    # ── Sidebar: Patient Profiles ──────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🗂️ 患者档案")
        profiles = _load_patient_profiles()
        if profiles:
            for pid, profile in profiles.items():
                variants = profile.get("variants", [])
                n_p  = sum(1 for v in variants if "Pathogenic" in v.get("acmg","") and "Likely" not in v.get("acmg",""))
                n_lp = sum(1 for v in variants if "Likely Pathogenic" in v.get("acmg",""))
                n_vus= sum(1 for v in variants if v.get("acmg","") == "VUS")
                with st.expander(f"👤 {pid} ({len(variants)} 变异)", expanded=False):
                    if n_p:  st.markdown(f"- P: {n_p}")
                    if n_lp: st.markdown(f"- LP: {n_lp}")
                    if n_vus:st.markdown(f"- VUS: {n_vus}")
                    for v in variants[-3:]:  # show last 3
                        st.markdown(
                            f"  `{v['variant']}` {v.get('gene','')}  "
                            + _acmg_badge(v.get("acmg","")) ,
                            unsafe_allow_html=True,
                        )
                    if st.button(f"查询 {pid} 历史", key=f"hist_{pid}",
                                 use_container_width=True):
                        st.session_state["agent_prefill"] = f"查询患者 {pid} 的所有历史变异记录"
                        st.rerun()
        else:
            st.caption("暂无患者档案。分析变异后可保存到档案。")

        st.markdown("---")
        st.markdown("### ⚙️ 会话管理")
        new_sid = st.text_input("会话 ID", value=session_id, key="sid_input",
                                help="不同会话有独立的对话记忆")
        if new_sid != session_id:
            st.session_state["agent_session_id"] = new_sid
            st.rerun()

        if st.button("🗑️ 清空对话记忆", use_container_width=True):
            from agent.agent import clear_memory
            clear_memory(session_id)
            st.session_state["agent_messages"] = []
            st.success("对话记忆已清空")
            st.rerun()

    # ── Header ─────────────────────────────────────────────────────────
    st.subheader("🤖 SNV-judge Agent — 自然语言对话分析")
    st.markdown(
        "直接用**中文或英文**描述你的分析需求，Agent 会自动调用合适的工具。  \n"
        "支持：单变异分析 · 追问 · 批量 VCF · 患者档案管理 · ClinVar 查询"
    )

    # ── Quick action buttons ───────────────────────────────────────────
    st.markdown("**快速操作：**")
    qcols = st.columns(4)
    quick_actions = [
        ("🔬 分析 TP53 R175H",    "请分析变异 17:7674220:C:T（TP53 R175H），给出完整的致病性评估"),
        ("🧬 分析 BRCA1 R1699W",  "请分析变异 17:43057062:C:T（BRCA1 R1699W），给出 ACMG 分类"),
        ("✅ 分析 BRCA2 N372H",   "请分析变异 13:32906729:C:A（BRCA2 N372H），这是一个已知良性变异"),
        ("📋 查看患者档案",        "列出所有已保存的患者档案"),
    ]
    for col, (label, query) in zip(qcols, quick_actions):
        if col.button(label, use_container_width=True, key=f"qa_{label}"):
            st.session_state["agent_prefill"] = query
            st.rerun()

    st.divider()

    # ── VCF file upload ────────────────────────────────────────────────
    with st.expander("📂 上传 VCF 文件进行批量分析", expanded=False):
        vcf_file = st.file_uploader(
            "上传 VCF 文件（.vcf 或 .vcf.gz）",
            type=["vcf", "gz"],
            key="agent_vcf_upload",
        )
        if vcf_file:
            import gzip
            raw = vcf_file.read()
            try:
                text = gzip.decompress(raw).decode("utf-8", errors="replace") \
                       if raw[:2] == b"\x1f\x8b" else raw.decode("utf-8", errors="replace")
            except Exception:
                text = ""
            if text:
                n_snvs = sum(1 for l in text.splitlines()
                             if l and not l.startswith("#"))
                st.info(f"检测到 {n_snvs} 行变异数据。点击下方按钮开始批量分析。")
                if st.button("🚀 批量分析此 VCF", type="primary", key="run_batch_vcf"):
                    # Truncate to first 200 lines for the prompt
                    vcf_lines = [l for l in text.splitlines()
                                 if l and not l.startswith("#")][:20]
                    vcf_snippet = "\n".join(vcf_lines)
                    query = (f"请批量分析以下 VCF 变异，给出排序后的致病性评估和高风险变异摘要：\n\n"
                             f"```\n{vcf_snippet}\n```")
                    st.session_state["agent_prefill"] = query
                    st.rerun()

    # ── Chat history display ───────────────────────────────────────────
    for msg in st.session_state["agent_messages"]:
        with st.chat_message(msg["role"],
                             avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg.get("tool_calls"):
                _render_tool_calls(msg["tool_calls"])

    # ── Handle prefill (from quick buttons / sidebar) ──────────────────
    prefill = st.session_state.pop("agent_prefill", None)

    # ── Chat input ─────────────────────────────────────────────────────
    user_input = st.chat_input(
        "输入变异坐标（如 17:7674220:C:T）或自然语言问题…",
        key="agent_chat_input",
    )

    # Use prefill if no direct input
    if prefill and not user_input:
        user_input = prefill

    if user_input:
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        st.session_state["agent_messages"].append({
            "role": "user", "content": user_input, "tool_calls": []
        })

        # Run agent
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Agent 正在分析，请稍候…"):
                try:
                    result = _run(user_input, session_id)
                    response = result["output"]
                    tool_calls = result["tool_calls"]
                except Exception as e:
                    response = f"Agent 执行出错：{str(e)}\n\n请检查 API Key 或网络连接。"
                    tool_calls = []

            st.markdown(response)
            if tool_calls:
                _render_tool_calls(tool_calls)

        st.session_state["agent_messages"].append({
            "role": "assistant",
            "content": response,
            "tool_calls": tool_calls,
        })
        st.rerun()

    # ── Empty state ────────────────────────────────────────────────────
    if not st.session_state["agent_messages"]:
        st.markdown("""
<div style="text-align:center;padding:40px;color:#888;">
<h3>👋 欢迎使用 SNV-judge Agent</h3>
<p>你可以这样开始：</p>
<ul style="text-align:left;display:inline-block;">
<li>输入变异坐标：<code>17:7674220:C:T</code></li>
<li>自然语言提问：<em>"分析 TP53 R175H 的致病性"</em></li>
<li>追问：<em>"这个变异在 ClinVar 里有记录吗？"</em></li>
<li>患者管理：<em>"把这个变异保存到患者 P001 的档案"</em></li>
<li>批量分析：上传 VCF 文件或粘贴多行变异</li>
</ul>
</div>
""", unsafe_allow_html=True)
