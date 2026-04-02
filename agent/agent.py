"""
agent/agent.py — SNV-judge Plan C: LangChain Agent Core
=========================================================
Builds a Kimi-powered clinical genetics agent with:
  - Tool-calling (9 tools: VEP, gnomAD, Evo2, Genos, SNV-judge, ClinVar,
                           patient save/load, batch VCF)
  - Short-term memory: ConversationBufferWindowMemory (last 10 turns)
  - Long-term memory: JSON patient profiles (persistent across sessions)
  - Bilingual (Chinese/English) natural language understanding

Usage:
    from agent.agent import create_agent, run_agent

    agent_executor = create_agent()
    response = run_agent(agent_executor, "分析变异 17:7674220:C:T", session_id="demo")
"""

import os
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

from agent.tools import ALL_TOOLS

# ── Kimi LLM config ────────────────────────────────────────────────────────
KIMI_API_KEY  = os.environ.get("KIMI_API_KEY",
                               "sk-OZUGp5zERGbsqndojmH2k2YBshSKfvltXtBFTzN4uLNF1idd")
KIMI_BASE_URL = "https://api.moonshot.cn/v1"
KIMI_MODEL    = "moonshot-v1-32k"

# ── System prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是 SNV-judge Agent，一位专业的临床遗传学 AI 助手，由 SNV-judge v5 系统驱动。

## 你的能力
你可以调用以下工具来分析人类基因组错义变异（missense SNV）：

1. **vep_annotate** — 获取 SIFT、PolyPhen-2、AlphaMissense、CADD、phyloP 注释
2. **gnomad_frequency** — 查询 gnomAD v4 人群等位基因频率（ACMG BA1/PM2 证据）
3. **evo2_score** — Evo2-40B 基因组语言模型零样本评分
4. **genos_score** — Genos-10B 人类基因组基础模型致病性评分
5. **snv_judge_predict** — 完整 v5 集成预测（自动调用上述工具 + XGBoost/LightGBM 集成 + Isotonic 校准）
6. **search_clinvar** — 搜索 ClinVar 数据库中的已有分类记录
7. **save_patient_variant** — 将分析结果保存到患者长期档案
8. **get_patient_history** — 查询患者历史变异记录
9. **batch_vcf_analyze** — 批量分析 VCF 格式的多个变异

## 工作原则

### 变异输入格式
- 标准格式：`CHROM:POS:REF:ALT`（GRCh38），例如 `17:7674220:C:T`
- 如果用户提供基因名+蛋白变化（如"TP53 R175H"），请告知需要提供基因组坐标，
  或使用 search_clinvar 查找对应坐标

### 分析流程
- **单变异分析**：优先使用 `snv_judge_predict`（它会自动调用所有工具）
- **需要详细某项证据**：单独调用对应工具（如只需 gnomAD 频率）
- **批量分析**：使用 `batch_vcf_analyze`
- **患者管理**：分析完成后，如用户提到患者信息，主动询问是否保存到档案

### ACMG 分类标准（v5 阈值）
- ≥ 90%  → Pathogenic (P)
- 70–90% → Likely Pathogenic (LP)
- 40–70% → VUS（意义不明确）
- 20–40% → Likely Benign (LB)
- < 20%  → Benign (B)

### 回答风格
- 使用中文回答（除非用户用英文提问）
- 专业术语保留英文（ACMG、SIFT、gnomAD 等）
- 结构清晰：先给结论，再解释证据
- 对 VUS 变异，明确说明不确定性和建议的后续步骤
- 引用 ACMG 证据条目（PP3、PM2、BA1 等）

### 重要提示
- 本系统仅供科研参考，不构成临床诊断依据
- 对于高风险变异（P/LP），建议结合家系分析和功能实验验证
"""

# ── Per-session memory store ───────────────────────────────────────────────
_memory_store: dict[str, ConversationBufferWindowMemory] = {}

def get_memory(session_id: str = "default") -> ConversationBufferWindowMemory:
    """Get or create a ConversationBufferWindowMemory for a session."""
    if session_id not in _memory_store:
        _memory_store[session_id] = ConversationBufferWindowMemory(
            k=10,                          # keep last 10 turns
            memory_key="chat_history",
            return_messages=True,
        )
    return _memory_store[session_id]

def clear_memory(session_id: str = "default") -> None:
    """Clear conversation memory for a session."""
    if session_id in _memory_store:
        _memory_store[session_id].clear()

def get_all_sessions() -> list[str]:
    """Return all active session IDs."""
    return list(_memory_store.keys())


# ── Agent factory ──────────────────────────────────────────────────────────
def create_agent(temperature: float = 0.3) -> AgentExecutor:
    """Create and return a configured SNV-judge AgentExecutor.

    Args:
        temperature: LLM temperature (0.3 = consistent clinical reasoning)

    Returns:
        Configured AgentExecutor with all tools and memory.
    """
    llm = ChatOpenAI(
        api_key=KIMI_API_KEY,
        base_url=KIMI_BASE_URL,
        model=KIMI_MODEL,
        temperature=temperature,
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm=llm, tools=ALL_TOOLS, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
    return executor


def run_agent(
    executor: AgentExecutor,
    user_input: str,
    session_id: str = "default",
) -> dict:
    """Run the agent with memory for a given session.

    Args:
        executor:   AgentExecutor from create_agent()
        user_input: User's natural language query
        session_id: Session identifier for memory isolation

    Returns:
        dict with keys:
          - output: str — agent's final response
          - intermediate_steps: list — tool calls made
          - tool_calls: list[dict] — simplified tool call log
    """
    memory = get_memory(session_id)
    chat_history = memory.chat_memory.messages

    result = executor.invoke({
        "input": user_input,
        "chat_history": chat_history,
    })

    # Update memory
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(result["output"])

    # Parse tool calls for UI display
    tool_calls = []
    for action, observation in result.get("intermediate_steps", []):
        tool_calls.append({
            "tool":   action.tool,
            "input":  action.tool_input,
            "output": observation[:500] + "..." if len(str(observation)) > 500 else observation,
        })

    return {
        "output":             result["output"],
        "intermediate_steps": result.get("intermediate_steps", []),
        "tool_calls":         tool_calls,
    }


# ── Convenience: single-shot query (no memory) ────────────────────────────
def quick_query(user_input: str) -> str:
    """Run a single query without memory. Useful for testing."""
    executor = create_agent()
    result = run_agent(executor, user_input, session_id="_quick")
    clear_memory("_quick")
    return result["output"]
