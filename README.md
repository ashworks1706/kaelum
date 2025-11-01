# KaelumAI 🧠

**Reasoning acceleration layer for lightweight LLMs**

---

## The Problem

Cheap LLMs (Llama 8B, Mistral 7B, Gemini Flash) struggle with:

- Poor reasoning & logical errors
- Hallucinations
- Wrong tool selection
- Math errors
- Bad agent orchestration

Fine-tuning/RLHF is too expensive and slow.

---

## Our Solution

Add reasoning verification at inference time:

- **Symbolic verification** - Math checking with SymPy
- **Factual verification** - RAG-based fact checking
- **Self-correction** - Adaptive reflection
- **Confidence scoring** - Reliability metrics

**Goals:**

- 🚀 Fast (<500ms overhead)
- 💰 Cheap (single LLM, smart caching)
- 🎯 Accurate (>90% improvement on benchmarks)

---

## Quick Start

```bash
# Setup
git clone https://github.com/ashworks1706/KaelumAI.git
cd KaelumAI
pip install -r requirements.txt

# Run Ollama locally
ollama pull qwen2.5:7b
```

```python
from kaelum import enhance

# Simple usage
result = enhance("What is 25% of 80?")

# Math mode
result = enhance("Solve: 3x + 5 = 20", mode="math")
```

---

## What We're Building

### Sprint 1: Core MVP

- [ ] LLM client (Ollama, OpenAI, vLLM)
- [ ] Reasoning trace generation
- [ ] Symbolic verification (SymPy)
- [ ] Basic confidence scoring
- [ ] One-line API

### Sprint 2: Verification

- [ ] RAG adapters (ChromaDB, Qdrant)
- [ ] Factual verification layer
- [ ] Self-reflection loop
- [ ] Adaptive stopping

### Sprint 3: Optimization

- [ ] LRU + Redis caching
- [ ] Tool selection guardrails
- [ ] Agent orchestration
- [ ] Prompt optimization

### Sprint 4: Benchmarks

- [ ] Speed benchmarks
- [ ] Hallucination detection tests
- [ ] Tool selection accuracy
- [ ] Math reasoning tests
- [ ] Agent orchestration tests

---

## Benchmark Decision Matrix

**Which benchmark to run for your use case:**

| Use Case | Recommended Benchmarks | Why |
|----------|----------------------|-----|
| **Production API** | Speed, Cost Analysis | Need fast responses at scale |
| **Customer Support Bot** | Hallucination Detection, Tool Selection | Must avoid false info & pick right tools |
| **Math Tutoring** | Math Reasoning (GSM8K), Hallucination Detection | Correct answers & no made-up facts |
| **Research Assistant** | Hallucination Detection (TruthfulQA), Tool Selection | Factual accuracy & proper tool usage |
| **Code Assistant** | Tool Selection (ToolBench), Math Reasoning | Correct logic & algorithm reasoning |
| **Multi-Agent System** | Agent Orchestration, Tool Selection | Agent coordination & task routing |
| **General Purpose** | All 5 Benchmarks | Comprehensive evaluation |

**Available Benchmarks:**
- **Speed** - Latency overhead measurement (25 queries)
- **Hallucination Detection** - TruthfulQA-style (20 cases)
- **Tool Selection** - ToolBench-inspired (25 scenarios)
- **Math Reasoning** - GSM8K-style (20 problems)
- **Agent Orchestration** - Custom workflow tests (10 scenarios)

---

## LLMs Decision Matrix

**Open-source models to test based on your constraints:**

| Constraint | Recommended Models | Notes |
|------------|-------------------|-------|
| **Local/Privacy** | Qwen 2.5 7B, Llama 3.2 3B, Mistral 7B | Run on Ollama, fully local |
| **Best Quality** | Qwen 2.5 7B, Llama 3.1 8B | Top open-source performers |
| **Fastest** | Llama 3.2 3B, Phi-3 Mini | Sub-second inference |
| **Math-Heavy** | Qwen 2.5 7B, DeepSeek Math | Better symbolic reasoning |
| **Long Context** | Llama 3.2 3B (128K), Qwen 2.5 7B | Extended context windows |
| **Low VRAM** | Llama 3.2 3B, Phi-3 Mini | Run on 8GB GPU |
| **Balanced** | Qwen 2.5 7B, Llama 3.2 3B | Good quality/speed tradeoff |

**Model Specs Reference:**

| Model | Size | Context | VRAM (4-bit) | Speed | Notes |
|-------|------|---------|--------------|-------|-------|
| Llama 3.2 3B | 3B | 128K | ~2.5GB | ⚡⚡⚡ | Best for speed |
| Llama 3.1 8B | 8B | 128K | ~5GB | ⚡⚡ | Balanced quality |
| Qwen 2.5 7B | 7B | 32K | ~4.5GB | ⚡⚡ | Best for math |
| Mistral 7B | 7B | 32K | ~4.5GB | ⚡⚡ | General purpose |
| Phi-3 Mini | 3.8B | 128K | ~2.8GB | ⚡⚡⚡ | Microsoft, fast |
| DeepSeek Math | 7B | 16K | ~4.5GB | ⚡⚡ | Math specialist |
| Gemma 2 9B | 9B | 8K | ~5.5GB | ⚡ | Google, high quality |

*Note: VRAM for quantized (4-bit). ⚡⚡⚡ = fastest*

---

## Target Metrics

| Priority | Metric | Target |
|----------|--------|--------|
| **Speed** | Latency overhead | <500ms |
| **Hallucination** | Detection rate | >90% |
| **Tool Selection** | Accuracy | >85% |
| **Math** | Correctness | >95% |
| **Orchestration** | Agent accuracy | >80% |
| **Cost** | $/1K queries | <$0.10 |

---

## Research to Implement
