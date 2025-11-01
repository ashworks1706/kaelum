
# 🧠 KaelumAI

**Reasoning Acceleration Layer for Lightweight LLMs**

> 🧪 **Testing:** One comprehensive notebook in `test_notebooks/testing.ipynb` - covers LLM selection, benchmarks, verification, reflection, performance, and integration testing

---

## 🌍 Overview

**KaelumAI** is a *modular reasoning verification layer* designed to make small and mid-sized LLMs **think better, faster, and more reliably** — without costly finetuning.

It acts as a **middleware MCP (Model-Context Protocol)** between your application and the base model, enabling contextual reasoning, symbolic math verification, factual guardrails, and adaptive reflection loops — all with minimal latency and cost.

---

## 💡 The Problem

Smaller and cheaper LLMs (e.g. **Llama 3 3B**, **Mistral 7B**, **Gemini Flash**) are fast but unreliable:

* ❌ Poor reasoning and logical consistency
* ❌ Frequent hallucinations
* ❌ Wrong tool selection
* ❌ Weak math performance
* ❌ Inefficient agent orchestration

Traditional fixes like **RLHF**, **fine-tuning**, or **distillation** are expensive and slow.

---

## ⚙️ Our Solution — *Inference-Time Reasoning Enhancement*

KaelumAI adds a verification and reflection layer *at inference*, not training:

| Layer                 | Description                                                               |
| --------------------- | ------------------------------------------------------------------------- |
| 🧩 **Contextualizer** | Builds structured reasoning context from history, RAG, and tools          |
| 🔍 **Verifier**       | Performs symbolic, factual, and numeric checks using deterministic rules  |
| 🔄 **Reflexor**       | Runs lightweight self-reflection passes to correct low-confidence outputs |
| 🧠 **Orchestrator**   | Routes to the right tool/agent dynamically using cost-aware policies      |
| 📊 **Tracer**         | Produces transparent reasoning traces for debugging and interpretability  |

**Goals:**

* ⚡ **Speed:** < 500 ms overhead
* 💸 **Efficiency:** Single LLM, smart caching
* 🎯 **Accuracy:** > 90 % gain on reasoning benchmarks

---

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/ashworks1706/KaelumAI.git
cd KaelumAI
pip install -r requirements.txt

# Pull a local model via Ollama
ollama pull qwen2.5:7b
```

```python
from kaelum import enhance

# Simple usage
result = enhance("What is 25% of 80?")

# Customize parameters
result = enhance(
    "Explain quantum entanglement",
    model="qwen2.5:7b",
    temperature=0.7,
    max_tokens=2048,
    max_iterations=2,
)
```

**🧪 For Testing & Experiments:**

Open `test_notebooks/testing.ipynb` in Jupyter:
- All-in-one testing suite with 8 organized sections
- Pre-configured test cells for different scenarios
- Speed vs Quality mode comparisons
- Model benchmarking (llama3.2:3b vs qwen2.5:7b)
- Document findings inline with markdown
- Sequential testing workflow for fast iteration

**⚡ Quick Demo:**
```bash
python example.py  # Simple one-shot demo
```

---

**⚡ Speed vs Quality Trade-offs:**

| Mode | temperature | max_tokens | max_iterations | Speed | Quality |
|------|-------------|------------|----------------|-------|---------|
| **Speed** | 0.3 | 512 | 1 | ⚡⚡⚡ Fast (2-3s) | ⭐⭐ Good |
| **Balanced** | 0.5 | 1024 | 1 | ⚡⚡ Medium (4-6s) | ⭐⭐⭐ Better |
| **Quality** | 0.7 | 2048 | 2 | ⚡ Slow (8-12s) | ⭐⭐⭐⭐ Best |

**🎯 Quick Start:**
```bash
# Default (llama3.2:3b, speed mode)
python example.py

# Specify model
python example.py qwen2.5:7b

# Specify model + mode
python example.py llama3.2:3b balanced
python example.py qwen2.5:7b quality

# Or edit presets directly in example.py
```

---

## 📁 Project Structure

```
kaelum/
├── __init__.py             # Public API: enhance() function
├── core/
│   ├── config.py          # Settings & environment config
│   ├── reasoning.py       # LLM client & trace generation
│   ├── verification.py    # SymPy symbolic math verifier
│   ├── reflection.py      # Self-reflection loop
│   └── rag_adapter.py     # RAG connectors (ChromaDB, Qdrant)
└── runtime/
    └── orchestrator.py    # MCP pipeline coordinator

test_notebooks/              # 🧪 Complete testing suite
├── 01_llm_selection.ipynb         # Choose best LLM
├── 02_benchmark_testing.ipynb     # GSM8K, TruthfulQA, ToolBench
├── 03_verification_testing.ipynb  # SymPy + RAG testing
├── 04_reflection_testing.ipynb    # Self-improvement testing
├── 05_performance_optimization.ipynb  # Speed optimization
└── 06_integration_edge_cases.ipynb    # Real-world scenarios
```

**Key Files:**
- `kaelum/__init__.py` → Main API entry point
- `reasoning.py` → Handles LLM calls & reasoning trace generation
- `verification.py` → Verifies math/logic using SymPy
- `orchestrator.py` → Runs verification → reflection loop
- `test_notebooks/` → **Start here for testing and development**
- `example.py` → Quick demo (single query)

---

## 🧩 Architecture

```
┌────────────────────────────┐
│        User Query          │
└────────────┬───────────────┘
             │
      Context Builder
             ↓
┌────────────────────────────┐
│     Base LLM (e.g. 7B)     │
└────────────┬───────────────┘
             │
   Reasoning Trace & Output
             ↓
┌────────────────────────────┐
│  Verification Layer (Kaelum)│
│ - Symbolic check (SymPy)   │
│ - Factual check (RAG)      │
│ - Self-reflection loop     │
└────────────┬───────────────┘
             ↓
      Enhanced & Verified Response
```

---

## 🧱 Development Roadmap

### 🏗 Sprint 1 — Core MVP

* [ ] LLM client (Ollama, OpenAI, vLLM)
* [ ] Reasoning trace generation
* [ ] Symbolic verification (SymPy)
* [ ] Confidence scoring
* [ ] One-line API

### 🔍 Sprint 2 — Verification Layer

* [ ] RAG adapters (ChromaDB, Qdrant)
* [ ] Factual verification layer
* [ ] Self-reflection loop
* [ ] Adaptive stopping

### ⚡ Sprint 3 — Optimization

* [ ] LRU + Redis caching
* [ ] Tool selection guardrails
* [ ] Agent orchestration
* [ ] Prompt optimization

### 🧠 Sprint 4 — Benchmarks & Testing

* [ ] Speed benchmarks
* [ ] Hallucination detection tests
* [ ] Tool selection accuracy
* [ ] Math reasoning tests
* [ ] Agent orchestration tests

---

## 📊 Benchmark Decision Matrix

| Use Case                    | Recommended Benchmarks                   | Why                                   |
| --------------------------- | ---------------------------------------- | ------------------------------------- |
| **Production API**          | Speed + Cost Analysis                    | Evaluate scalability and latency      |
| **Customer Support Bot**    | Hallucination Detection + Tool Selection | Needs reliable factual grounding      |
| **Math Tutor / STEM Agent** | GSM8K / MATH + HalluEval                 | Symbolic & numerical correctness      |
| **Research Assistant**      | TruthfulQA + ToolBench                   | Factual precision + API correctness   |
| **Code Assistant**          | ToolBench + Math Reasoning               | Logic + algorithmic reliability       |
| **Multi-Agent System**      | Agent Orchestration Benchmark            | Proper task routing and collaboration |
| **General Purpose**         | All five                                 | Comprehensive evaluation              |

**Available Benchmarks**

* ⚡ Speed — Latency / token throughput
* 🧠 Math Reasoning — GSM8K / MATH subset
* 🔍 Hallucination — TruthfulQA / HalluEval
* 🧰 Tool Selection — ToolBench subset
* 🤝 Agent Orchestration — Custom workflow tests

---

## 🧮 LLM Decision Matrix (2025 Update)

| Constraint                | Recommended Models                                        | Why                                           |
| ------------------------- | --------------------------------------------------------- | --------------------------------------------- |
| **Local / Privacy-First** | **Qwen 2.5 7B**, **Llama 3.2 3B**, **Mistral 7B**         | Fully local via Ollama / vLLM                 |
| **Best Overall Quality**  | **Gemma 2 9B**, **Qwen 2.5 14B**, **Llama 3.1 8B**        | High reasoning + factual scores               |
| **Fastest (Edge / CPU)**  | **Phi-3 Mini (3.8B)**, **Llama 3.2 3B**                   | Sub-second inference on 8 GB GPU              |
| **Math-Heavy Reasoning**  | **Qwen 2.5 7B**, **DeepSeek Math 7B**, **DeepSeek R1 8B** | Specialized math training datasets            |
| **Long Context / Memory** | **Llama 3.2 3B (128K)**, **Gemma 2 9B (32K)**             | Extended context for multi-agent coordination |
| **Low VRAM Deployment**   | **Phi-3 Mini**, **Llama 3.2 3B**                          | Fits on laptops / 8 GB GPUs                   |
| **Balanced All-Rounder**  | **Mistral 7B**, **Qwen 2.5 7B**, **Llama 3.1 8B**         | Great mix of cost / latency / accuracy        |

**Model Specs (4-bit Quantized Reference)**

| Model            | Size | Context | VRAM (4-bit) | Speed | Notes                         |
| ---------------- | ---- | ------- | ------------ | ----- | ----------------------------- |
| Llama 3.2 3B     | 3B   | 128K    | ≈ 2.5 GB     | ⚡⚡⚡   | Fastest baseline for Kaelum   |
| Llama 3.1 8B     | 8B   | 128K    | ≈ 5 GB       | ⚡⚡    | Balanced quality/speed        |
| Qwen 2.5 7B      | 7B   | 32K     | ≈ 4.5 GB     | ⚡⚡    | Strong in math & code         |
| DeepSeek R1 8B   | 8B   | 16K     | ≈ 5 GB       | ⚡⚡    | Reasoning-optimized           |
| DeepSeek Math 7B | 7B   | 16K     | ≈ 4.5 GB     | ⚡⚡    | Symbolic math expert          |
| Mistral 7B       | 7B   | 32K     | ≈ 4.5 GB     | ⚡⚡    | General purpose               |
| Phi-3 Mini 3.8B  | 3.8B | 128K    | ≈ 2.8 GB     | ⚡⚡⚡   | Ultra-fast edge model         |
| Gemma 2 9B       | 9B   | 32K     | ≈ 5.5 GB     | ⚡     | High quality from Google      |
| Qwen 2.5 14B     | 14B  | 32K     | ≈ 8 GB       | ⚡     | Top open reasoning model 2025 |

⚡ = speed rating (fewer ⚡ → slower but smarter)

---

## 🎯 Target Metrics

| Priority         | Metric                       | Target   |
| ---------------- | ---------------------------- | -------- |
| ⚡ Speed          | Latency Overhead             | < 500 ms |
| 🧠 Reasoning     | Math Correctness             | > 95 %   |
| 🔍 Factuality    | Hallucination Detection Rate | > 90 %   |
| 🧰 Tool Use      | Correct Tool Selection       | > 85 %   |
| 🤖 Orchestration | Agent Accuracy               | > 80 %   |
| 💸 Cost          | $/1K queries                 | < $0.10  |

---

## 📚 Suggested Research & References

* [Anthropic – *Tracing the Thoughts of a Language Model* (2025)](https://www.anthropic.com/research/tracing-thoughts-language-model)
* [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
* [Self-RAG (2024): Retrieval-Augmented Generation with Self-Verification](https://arxiv.org/abs/2310.06112)
* [DeepSeek R1 Technical Report (2025)](https://medium.com/data-science-in-your-pocket/deepseek-r1-best-open-source-reasoning-llm-outperforms-openai-o1-b79869392945)
* [ToolBench: Benchmarking LLM Tool Use and APIs](https://github.com/openbmb/toolbench)
* [TruthfulQA / HalluEval for Hallucination Testing](https://github.com/sylinrl/hallueval)
* [GSM8K / MATH Datasets for Reasoning](https://github.com/openai/grade-school-math)
