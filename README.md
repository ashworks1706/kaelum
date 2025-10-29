# KaelumAI 🧠

### *Making AI Reasoning Verifiable*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io)

---

## 🎯 The Problem

Large Language Models are incredibly fluent but fundamentally unreliable.

They produce **plausible but incorrect** reasoning, hallucinate facts, and make mathematical errors — all while sounding confident. Current agentic systems rely on "self-reflection" and "chain-of-thought," but these are **not objectively verified**. The AI simply becomes more confident, not more correct.

This makes LLMs **unusable in safety-critical domains** like education, finance, healthcare, and autonomous systems.

**The core issue:** There's no standard way to verify *why* an AI reached its conclusion.

---

## 💡 Our Solution

**KaelumAI** is a Model Context Protocol (MCP) tool that acts as a **logic co-processor** for any LLM or agentic system.

We intercept reasoning traces, validate them step-by-step using symbolic logic and factual verification, then return **verified reasoning chains** with confidence scores and corrections.

> **Think of it as a reasoning safety net** — a verification layer that makes AI thought processes transparent and trustworthy.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Symbolic Verification** | Validates mathematical and logical reasoning using `SymPy` and formal logic engines |
| 📚 **Factual Verification** | Cross-checks claims against knowledge bases (ChromaDB, FAISS) to detect hallucinations |
| 📊 **Confidence Scoring** | Assigns reliability metrics to each reasoning step and overall trace |
| 🔄 **Feedback Loop** | Returns corrected reasoning to the model via standardized MCP schema |
| 📝 **Trace Logging** | Captures reasoning patterns for debugging, evaluation, and fine-tuning |
| 🔌 **MCP Standard** | Plug-and-play integration with any MCP-compatible system |

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Agent / LLM    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│     Reasoning MCP (KaelumAI)    │
├─────────────────────────────────┤
│  ├─ Parsing Layer               │
│  ├─ Symbolic Verifier (SymPy)   │
│  ├─ Factual Verifier (RAG)      │
│  ├─ Confidence Scorer            │
│  ├─ Feedback Generator           │
│  └─ Trace Logger                 │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  External Knowledge & Tools     │
│  • RAG Databases                │
│  • Symbolic Solvers             │
│  • APIs & Knowledge Graphs      │
└─────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/ashworks1706/KaelumAI.git
cd KaelumAI
pip install -r requirements.txt
```

### Run the Server

```bash
uvicorn app.main:app --reload
```

### Test the API

```bash
curl -X POST http://localhost:8000/verify_reasoning \
  -H "Content-Type: application/json" \
  -d '{
    "query": "If 3x + 5 = 11, what is x?",
    "reasoning": [
      {"step_text": "Subtract 5 from both sides → 3x = 6"},
      {"step_text": "Divide by 3 → x = 2"}
    ]
  }'
```

**Response:**

```json
{
  "verified": true,
  "confidence": 0.97,
  "final_answer": "2",
  "feedback": "All reasoning steps verified through symbolic validation.",
  "corrections": []
}
```

---

## 🧠 Example Usage

### Python SDK

```python
from kaelum import ReasoningMCP

# Initialize the verifier
verifier = ReasoningMCP()

# Verify reasoning trace
result = verifier.verify(
    query="If 3x + 5 = 11, what is x?",
    steps=[
        "Subtract 5 from both sides → 3x = 6",
        "Divide by 3 → x = 2"
    ]
)

print(f"Verified: {result.verified}")
print(f"Confidence: {result.confidence}")
print(f"Answer: {result.final_answer}")
```

### Integration with LangGraph

```python
from langgraph.prebuilt import create_react_agent
from kaelum import reasoning_mcp_tool

# Add KaelumAI as a verification tool
tools = [reasoning_mcp_tool, ...other_tools]
agent = create_react_agent(model, tools)

# The agent can now verify its reasoning before acting
```

---

## 📂 Project Structure

```
KaelumAI/
├── app/
│   ├── main.py              # FastAPI server
│   ├── models.py            # Pydantic schemas
│   ├── verifier/
│   │   ├── symbolic.py      # Math & logic verification
│   │   ├── factual.py       # RAG-based fact checking
│   │   └── scorer.py        # Confidence calculation
│   └── utils/
│       ├── parser.py        # Reasoning trace extraction
│       └── logger.py        # Trace storage
├── mcp/
│   ├── manifest.json        # MCP specification
│   └── adapter.py           # MCP protocol interface
├── tests/                   # Unit & integration tests
├── data/                    # Knowledge bases & logs
├── requirements.txt
└── README.md
```

---

## 🔌 MCP Specification

```json
{
  "name": "kaelum_reasoning",
  "version": "0.1.0",
  "description": "MCP tool for verifying AI reasoning traces",
  "type": "tool",
  "capabilities": ["verify_reasoning", "score_confidence"],
  "api": {
    "verify_reasoning": {
      "method": "POST",
      "path": "/verify_reasoning",
      "input_schema": "ReasoningRequest",
      "output_schema": "ReasoningResponse"
    }
  }
}
```

---

## 🎯 Use Cases

### 🎓 Education
Verify AI tutors' explanations in real-time, ensuring students receive mathematically and factually correct guidance.

### 💼 Enterprise AI
Add an audit layer to agentic workflows, making AI decision-making explainable and compliant.

### 🔬 Research
Generate datasets of verified reasoning traces for training more reliable models and benchmarking LLM reasoning capabilities.

### 🤖 Autonomous Agents
Prevent agents from taking actions based on faulty logic by verifying reasoning before execution.

---

## 🗺️ Roadmap

| Phase | Timeline | Deliverables |
|-------|----------|--------------|
| **MVP** | Q4 2025 | ✅ MCP protocol implementation<br>✅ Symbolic verification (SymPy)<br>✅ Basic confidence scoring |
| **V1.0** | Q1 2026 | 🚧 RAG factual verifier<br>🚧 Advanced scoring algorithms<br>🚧 Dashboard UI |
| **V1.5** | Q2 2026 | 📅 Multi-modal reasoning support<br>📅 Custom verification rules<br>📅 Enterprise features |
| **V2.0** | Q3 2026 | 🔮 RL feedback integration<br>🔮 Reasoning-optimized fine-tuning<br>🔮 API marketplace |

---

## 🤝 Integration Partners

KaelumAI seamlessly integrates with:

- **LangChain / LangGraph** — Add as verification tool in agent chains
- **AutoGen** — Plug into multi-agent conversations
- **CrewAI** — Verify reasoning before task delegation
- **Custom Agentic Systems** — Any MCP-compatible framework

---


## 📊 Impact & Metrics

Our goal is to improve AI reliability across the industry:

| Metric | Target |
|--------|---------|
| **Reasoning Accuracy** | +20-40% improvement |
| **Hallucination Detection** | 85%+ accuracy |
| **Transparency** | 100% traceable reasoning |
| **Integration Time** | <30 minutes |

---

## 🤝 Contributing

We welcome contributions! Whether you're adding verification modules, expanding knowledge bases, or improving documentation.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🌟 Vision

We're building the **standard for verifiable reasoning in AI systems**.

Just as GPUs became essential for AI training, reasoning verification will become essential for AI deployment.

Every autonomous agent should validate its logic before acting. Every LLM should show its work. Every AI system should be transparent about *why* it reached a conclusion.

**KaelumAI makes this possible.**

> "Turning language models from guessing machines into trustworthy reasoning systems."

---

## 📬 Contact

- **Email:** ashworks1706@gmail.com
---

<div align="center">

**Made with 🧠 by the KaelumAI Team**

[⭐ Star us on GitHub](https://github.com/ashworks1706/KaelumAI) 

</div>
