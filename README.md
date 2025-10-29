
# **KaelumAI 🧠**

### *The Missing Reasoning Layer for LLMs*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io)

---

## 🎯 The Problem

Large Language Models *sound* intelligent but often **reason poorly**.
They hallucinate, contradict themselves, and produce logical or mathematical errors — all with convincing fluency.

“Chain-of-thought” and “self-reflection” improve *style*, not *truth*.
There has never been a **standard reasoning layer** that:

* verifies intermediate logic,
* cross-checks factual claims, and
* exposes *why* a model reached its conclusion.

---

## 💡 The Solution — **The Reasoning Layer**

**KaelumAI** is a **Modular Cognitive Processor (MCP)** — a reasoning middleware that can be plugged into any LLM runtime or agent framework.

It functions as a **logic co-processor**, verifying reasoning traces, refining them through multi-LLM reflection, and returning *auditable conclusions* with quantitative confidence.

> 🧠 *Think of it as a GPU for reasoning* — a plug-in layer that validates and accelerates thought inside any AI system.

---

## ✨ Key Features

| Feature                                | Description                                                                      |
| -------------------------------------- | -------------------------------------------------------------------------------- |
| 🧠 **Reasoning MCP Core**              | Composable pipeline: generation → verification → reflection → scoring            |
| 🔍 **Symbolic + Factual Verification** | Math via SymPy and factual retrieval through FAISS / Chroma RAG                  |
| 🤖 **Verifier & Reflector LLMs**       | Independent LLMs review and repair logic to prevent self-confirmation bias       |
| 🧾 **Confidence Scoring Engine**       | Quantifies reliability of each reasoning trace and aggregates confidence         |
| 🔄 **Self-Correction Loop**            | Automatically repairs inconsistent or invalid reasoning chains                   |
| ⚙️ **Adaptive Policy Controller**      | RL and heuristic scheduling minimize latency and cost while maintaining accuracy |
| 🧩 **Tool-Based Integration**          | Register as `reasoning_mcp` in `models.tools([...])` — works with any LLM stack  |
| 📜 **Trace Logging & Analytics**       | Stores verified reasoning, errors, and metrics for transparency and fine-tuning  |
| 🌐 **Cloud Deployment Ready**          | Stateless MCP microservices, distributed verifier networks, real-time telemetry  |

---

## 🧠 Architecture Overview

```
┌───────────────────────────────┐
│        User / Agent Query     │
└─────────────┬─────────────────┘
              │
         [ModelRuntime]
              │
  ┌───────────┴──────────────────────┐
  │   Registered Tools (Composable)  │
  │ reasoning_mcp • retriever • api  │
  └────────────┬─────────────────────┘
               ▼
     ┌───────────────────────────────┐
     │        KaelumAI MCP Layer     │
     │ ├─ Generation (Base LLM)      │
     │ ├─ Verification (Symbolic/RAG)│
     │ ├─ Verifier LLM + Reflector LLM│
     │ ├─ Confidence & Policy Engine │
     │ ├─ Trace Logger + Telemetry   │
     └────────────┬──────────────────┘
                  ▼
     ┌──────────────────────────────┐
     │   Verified Reasoning Output  │
     │   + Confidence + Citations   │
     └──────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/ashworks1706/KaelumAI.git
cd KaelumAI
pip install -r requirements.txt
```

### Minimal Example

```python
from kaelum import ReasoningMCPTool, ModelRuntime, LLMClient, LLMConfig, MCPConfig

llm = LLMClient(LLMConfig(model="gpt-4o-mini"))
mcp_tool = ReasoningMCPTool(MCPConfig())
runtime = ModelRuntime(llm).attach(mcp_tool)

print(runtime.generate_content(
    "Explain how reinforcement learning can optimize a RAG retriever."
))
```

### Run as API

```bash
uvicorn app.main:app --reload
```

---

## ⚙️ Implementation Patterns

### 1️⃣ **Standalone Reasoning**

```python
from kaelum import MCP, MCPConfig
mcp = MCP(MCPConfig())
result = mcp.infer("If 3x + 5 = 11, what is x?")
print(result.final)
```

### 2️⃣ **LangChain / LangGraph Tool**

```python
from langchain.agents import initialize_agent, Tool
from kaelum import ReasoningMCPTool, MCPConfig

reasoning_tool = Tool(
    name="kaelum_reasoning",
    func=lambda q: ReasoningMCPTool(MCPConfig()).run([{"role":"user","content":q}]),
    description="Verifies and corrects reasoning traces"
)
agent = initialize_agent([reasoning_tool], llm=base_llm,
                         agent_type="zero-shot-react-description")
```

### 3️⃣ **Micro-MCP Service**

Expose a verified reasoning endpoint:

```
POST /verify_reasoning
```

Compatible with **MCP Manifest v0.1** for multi-model integration.

---

## 🔁 Request–Response Lifecycle

```json
// Request
{
  "query": "Explain how RL improves retrieval in RAG.",
  "reasoning_trace": [
    "RL adjusts retriever weights based on answer quality.",
    "Reward = similarity between predicted and gold answer."
  ]
}

// KaelumAI performs:
// 1. Symbolic/RAG verification
// 2. Verifier LLM critique
// 3. Reflector LLM repair
// 4. Confidence scoring and trace logging

// Response
{
  "verified": true,
  "confidence": 0.94,
  "final_answer": "RL fine-tunes document retrieval using reward signals of answer relevance.",
  "trace": ["Verified logical consistency between retrieval weights and reward signal."]
}
```

---

## 📂 Project Structure

```
KaelumAI/
├── core/        # Reasoning pipeline (generation / verification / reflection)
├── tools/       # ReasoningMCPTool adapter + protocol
├── runtime/     # ModelRuntime orchestration
├── app/         # FastAPI microservice
├── mcp/         # MCP manifest + spec adapter
├── tests/       # Unit / integration tests
└── README.md
```

---

## ⚙️ LangChain + Guardrails Integration

```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from guardrails import Guard
from kaelum import ReasoningMCPTool, MCPConfig, LLMConfig

base_llm = ChatOpenAI(model="gpt-4o")

reasoning_mcp = ReasoningMCPTool(MCPConfig(
    llm=LLMConfig(model="gpt-4o"),
    verifier_llm=LLMConfig(model="gpt-3.5-turbo"),
    reflector_llm=LLMConfig(model="claude-3-haiku"),
    use_symbolic=True
))

guard = Guard.from_rail("""
<rail version="0.1">
  <output>
    <string name="final_answer" description="Verified reasoning answer"/>
  </output>
</rail>
""")

reasoning_tool = Tool(
    name="kaelum_reasoning",
    func=lambda q: reasoning_mcp.run([{"role":"user","content":q}]),
    description="Verifies reasoning before output"
)

agent = initialize_agent([reasoning_tool], base_llm,
                         agent_type="zero-shot-react-description")

response = agent.run("Explain how RL improves retrieval in RAG.")
verified = reasoning_mcp.run([{"role":"user","content":response}])
safe_output = guard.parse(verified["final"])

print("✅ Verified:", safe_output)
print("Confidence:", verified["diagnostics"]["confidence"])
```

---

## 🧱 Scalability & Deployment

| Layer                    | Function                            | Scale Strategy                           |
| ------------------------ | ----------------------------------- | ---------------------------------------- |
| **Reasoning Kernel**     | Core reasoning microservice         | Stateless, horizontally scalable         |
| **Verifier Network**     | Parallel LLMs reviewing logic       | Distributed model routing                |
| **Symbolic/RAG Modules** | Deterministic fact & math checks    | Plug-and-play backends                   |
| **Policy Learner**       | RL scheduler for verification depth | Adaptive latency–accuracy trade-off      |
| **Telemetry & Storage**  | Reasoning logs + metrics            | Redis / Postgres with Grafana dashboards |

KaelumAI runs as a **cloud-native Reasoning Platform** that scales across both agents (horizontal) and reasoning complexity (vertical).

---

## 🗺️ Release Summary (2025)

| Module                        | Status     | Highlights                                       |
| ----------------------------- | ---------- | ------------------------------------------------ |
| **Reasoning Kernel**          | ✅ Complete | Generation → Verification → Reflection → Scoring |
| **Symbolic & RAG Verifiers**  | ✅          | Multi-backend verification (SymPy, FAISS)        |
| **Verifier Network**          | ✅          | Cross-model logic validation                     |
| **RL Policy Controller**      | ✅          | Adaptive reasoning depth                         |
| **LangChain / LangGraph SDK** | ✅          | One-line integration                             |
| **Dashboard & Metrics**       | ✅          | Live reasoning telemetry & analytics             |

---

## 🎯 Use Cases

* 🎓 **Education** — verify AI tutor logic in real time
* 💼 **Finance / Healthcare** — audit critical AI decisions
* 🔬 **Research** — benchmark reasoning reliability
* 🤖 **Agentic Systems** — verify logic before execution

---

## 📊 Impact Benchmarks

| Metric                      | Target               |
| --------------------------- | -------------------- |
| **Reasoning Accuracy**      | +35 % vs vanilla CoT |
| **Hallucination Detection** | > 90 %               |
| **Latency Overhead**        | < 20 % of base model |
| **Trace Transparency**      | 100 % auditable      |
| **Integration Time**        | < 30 min             |

---

## 🤝 Contributing

1. Fork the repo
2. Create branch `feature/x`
3. Add or improve modules (verifier, retriever, policy)
4. Submit PR 🚀

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## 📬 Contact

**Email:** [ashworks1706@gmail.com](mailto:ashworks1706@gmail.com)
**GitHub:** [https://github.com/ashworks1706/KaelumAI](https://github.com/ashworks1706/KaelumAI)


This is now the **final production-ready README** for launch — KaelumAI v2 presented as a fully realized, distributed reasoning framework with verifier networks, adaptive RL policies, and end-to-end integration support.
