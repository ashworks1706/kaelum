# 🧠 KaelumAI

**Fast, verifiable reasoning for small local language models**

---

## 🧩 Overview

KaelumAI is a **reasoning-verification runtime** that makes small, cheap LLMs (3-8 B) **think reliably** without retraining.
It sits between your app and any base model, performing real-time reasoning, symbolic + factual verification, and self-correction.

> **Core Idea:** Verification as a first-class inference step.
> Kaelum turns “dumb” models into fast, trustworthy reasoners.

---

## ⚙️ How It Works

```
User Query
   ↓
Reasoning  →  Verification  →  Reflection  →  Confidence-Scored Answer
```

At runtime Kaelum:

1. Builds minimal reasoning traces.
2. Verifies them symbolically (math) and/or factually (RAG).
3. Runs a lightweight reflection loop when confidence < threshold.
4. Returns a verified answer + trace + confidence score.

---

## 🚀 Usage

```python
from kaelum import enhance, set_reasoning_model

# Configure your reasoning model
set_reasoning_model(
    provider="ollama",
    model="llama3.2:3b",
    temperature=0.7,
    max_tokens=2048,
    max_reflection_iterations=2,
    use_symbolic_verification=True,
    use_factual_verification=False,
)

result = enhance("What is 15% of 200?")
print(result.answer, result.confidence)
```

---

## 🧰 Install

```bash
git clone https://github.com/ashworks1706/KaelumAI.git
cd KaelumAI
pip install -e .

# Example local model
ollama pull llama3.2:3b && ollama serve
```

---

## 🔧 Tweakable Parameters

### Model Settings

| Parameter  | Description                                       |
| ---------- | ------------------------------------------------- |
| `provider` | `"ollama"`, `"vllm"`, or `"custom"`               |
| `model`    | Model name (e.g. `"llama3.2:3b"`, `"qwen2.5:7b"`) |
| `base_url` | Custom endpoint (optional)                        |

### Generation Settings

* `temperature` 0.0-2.0 (controls randomness)
* `max_tokens` 1-128 000 (limit response length)

### Reasoning Settings

* `max_reflection_iterations` 0-5  → depth of self-correction
* `use_symbolic_verification` bool → math checking (SymPy)
* `use_factual_verification` bool → RAG fact checking
* `rag_adapter` → plug-in retriever if factual verification enabled

---

## 🧠 Core Modules

| Module                | Purpose                                    | Tech                             |
| --------------------- | ------------------------------------------ | -------------------------------- |
| **Reasoner**          | Runs base LLM and produces reasoning trace | Any 3-8 B model via Ollama/vLLM  |
| **Verifier**          | Symbolic (math) + factual checking         | SymPy, RAG retrievers            |
| **Reflexor**          | Self-reflection loop on low confidence     | LLM prompt re-evaluation         |
| **Confidence Engine** | Entropy + consistency scoring              | Custom logit/trace metrics       |
| **Router (Planned)**  | Policy model deciding tools/agents         | Heuristic → trainable controller |

---

## 🧮 Benchmark Focus

| Metric               | Target                 | Meaning                |
| -------------------- | ---------------------- | ---------------------- |
| Reasoning Accuracy   | +30 → 50 % vs baseline | GSM8K / ToolBench      |
| Hallucination Rate   | <10 %                  | TruthfulQA subset      |
| Latency Overhead     | < 500 ms               | Verifier loop cost     |
| Cost per 1 k queries | < $0.10                | Run cheap local models |

Benchmarks under development: GSM8K (math), ToolBench (tool routing), TruthfulQA (factual).

---

## 🧩 Differentiators

| What Kaelum Does                | Why It Matters                              |
| ------------------------------- | ------------------------------------------- |
| **Inference-time verification** | Improves answers without training           |
| **Adaptive reflection**         | Self-correction within fixed latency budget |
| **Model-agnostic runtime**      | Wraps any local LLM or API                  |
| **Small-model optimization**    | Runs on laptop-grade GPUs (3-8 B)           |
| **Verifiable reasoning traces** | Audit and visualize each decision           |

**Tagline:** *“Fast, verified reasoning for small models.”*

---

## 🧱 Architecture Snapshot

```
User Query
   ↓
Context Builder (RAG optional)
   ↓
Reasoner (Base LLM)
   ↓
Verifier (Symbolic + Factual)
   ↓
Reflexor (Self-Correction)
   ↓
Confidence Engine → Output
```

*(Planned)* Micro-controller (“Kaelum Brain”) will learn routing and reasoning depth from collected traces.

---

## 🧪 Testing

```bash
python example.py          # default config
python test_settings.py    # different parameter sets
```

---

## 🔮 Roadmap (Internal)

1️⃣ MVP Reasoning + Verification loop
2️⃣ Latency & Cost benchmark suite
3️⃣ Router LLM (Kaelum Brain) prototype
4️⃣ Domain agents (Math, Logic, Fact)
5️⃣ Training on reasoning traces → controller model

---
