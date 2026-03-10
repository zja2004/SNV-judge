# SNV-judge Plan C — LangChain Multi-Agent System

> **Branch**: `plan-c`  
> **Status**: Active development  
> **Requires**: `langchain>=0.3`, `langchain-openai>=0.2`, `KIMI_API_KEY`

## Overview

Plan C upgrades the static Plan B pipeline (single LLM call) into a **LangChain-powered autonomous agent** that can:

- Understand natural language queries in Chinese or English
- Autonomously decide which tools to call and in what order
- Maintain multi-turn conversation memory (short-term, per-session)
- Persist patient variant profiles across sessions (long-term memory)
- Analyze batches of variants from VCF files

```
User natural language query
        ↓
  Kimi LLM (moonshot-v1-32k) — Tool-calling agent
  ├── vep_annotate        → SIFT · PP2 · AlphaMissense · CADD · phyloP
  ├── gnomad_frequency    → gnomAD v4 AF + ACMG BA1/PM2
  ├── evo2_score          → Evo2-40B LLR (NVIDIA NIM)
  ├── genos_score         → Genos-10B pathogenicity (Stomics)
  ├── snv_judge_predict   → v5 ensemble + Isotonic calibration + SHAP
  ├── search_clinvar      → NCBI ClinVar records
  ├── save_patient_variant→ JSON patient profile (long-term memory)
  ├── get_patient_history → Retrieve patient's historical variants
  └── batch_vcf_analyze   → Multi-variant VCF scoring + ranking
        ↓
  Structured clinical response (ACMG-cited, bilingual)
```

## Architecture

| Component | File | Description |
|---|---|---|
| Tools | `agent/tools.py` | 9 LangChain `@tool` functions |
| Agent | `agent/agent.py` | AgentExecutor + memory management |
| UI | `agent/chat_ui.py` | Streamlit chat tab renderer |
| Patient DB | `agent/patient_profiles.json` | Persistent patient profiles (auto-created) |

## Tools

| Tool | Purpose | External API |
|---|---|---|
| `vep_annotate` | SIFT, PolyPhen-2, AlphaMissense, CADD, phyloP | Ensembl VEP REST |
| `gnomad_frequency` | Population AF + ACMG BA1/PM2 | gnomAD GraphQL |
| `evo2_score` | Genomic LLR (zero-shot) | NVIDIA NIM (optional) |
| `genos_score` | Human-centric pathogenicity | Stomics Cloud (optional) |
| `snv_judge_predict` | Full v5 ensemble prediction | Local model |
| `search_clinvar` | ClinVar records lookup | NCBI E-utilities |
| `save_patient_variant` | Persist to patient profile | Local JSON |
| `get_patient_history` | Retrieve patient history | Local JSON |
| `batch_vcf_analyze` | Multi-variant VCF scoring | All of the above |

## Memory Architecture

```
Short-term (per-session):
  ConversationBufferWindowMemory(k=10)
  → Remembers last 10 turns
  → Isolated per session_id
  → Cleared on demand

Long-term (persistent):
  agent/patient_profiles.json
  → Survives app restarts
  → Indexed by patient_id
  → Stores variant, ACMG, probability, notes, timestamp
```

## Quick Start

```bash
# Set API keys
export KIMI_API_KEY="sk-..."          # Required
export EVO2_API_KEY="nvapi-..."       # Optional
export GENOS_API_KEY="sk-..."         # Optional

# Run the app (Agent Chat is the 5th tab)
streamlit run app.py
```

### Python API

```python
from agent.agent import create_agent, run_agent

executor = create_agent()

# Single query
result = run_agent(executor, "分析变异 17:7674220:C:T", session_id="demo")
print(result["output"])

# Multi-turn (memory maintained within same session_id)
r1 = run_agent(executor, "分析 TP53 R175H: 17:7674220:C:T", session_id="s1")
r2 = run_agent(executor, "这个变异的 gnomAD 频率是多少？", session_id="s1")  # remembers context
r3 = run_agent(executor, "保存到患者 P001 档案", session_id="s1")
```

## Example Conversations

**Single variant analysis:**
```
User: 请分析变异 17:7674220:C:T，这是 TP53 的 R175H 突变
Agent: [calls snv_judge_predict]
       TP53 R175H 被分类为 Pathogenic (P)，校准概率 96.9%。
       主要证据：AlphaMissense=0.9963 (PP3)，gnomAD AF=1.97e-5 (PM2)...
```

**Follow-up question (memory):**
```
User: 这个变异在 ClinVar 里有记录吗？
Agent: [calls search_clinvar with "TP53 R175H" — remembers context]
       ClinVar ID 12374: NM_000546.6(TP53):c.524G>A (p.Arg175His)...
```

**Patient profile:**
```
User: 保存到患者 P001，备注：Li-Fraumeni 综合征先证者
Agent: [calls save_patient_variant]
       已保存到患者 P001 档案。
```

**Batch VCF:**
```
User: 批量分析以下变异：
      17 7674220 . C T
      13 32906729 . C A
Agent: [calls batch_vcf_analyze]
       分析了 2 个变异：
       1. chr17:7674220 C>T (TP53) — Pathogenic (P), 96.9%
       2. chr13:32906729 C>A (BRCA2) — Likely Benign (LB), 18.2%
```

## Comparison: Plan B vs Plan C

| Dimension | Plan B (v5) | Plan C (this branch) |
|---|---|---|
| Architecture | Static pipeline | LangChain autonomous agent |
| Reasoning | Single LLM call | Multi-step tool orchestration |
| Tool calling | Fixed order | Dynamic, query-dependent |
| Dialogue | None | Multi-turn with memory |
| Memory | Session-only | Short-term + long-term (JSON) |
| Patient records | None | Persistent profiles |
| Batch analysis | Fixed pipeline | Agent-orchestrated with auto-prioritization |
| Language | Chinese only | Chinese + English |

## Limitations

- Evo2 and Genos scoring require separate API keys (optional; training medians used as fallback)
- Patient profiles stored as local JSON (not encrypted; do not store real patient data in production)
- Agent may occasionally call unnecessary tools for simple queries (LLM reasoning overhead)
- Not validated for clinical use — research only
