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

## Research to Implement

**Week 1-2:** Pick 3 techniques each and implement:

- Chain-of-Verification (CoVe) - Meta
- Self-Consistency - Google
- ReAct - Princeton/Google
- Tree-of-Thoughts (ToT) - Princeton
- Program-Aided Language Models (PAL) - CMU
- Verify-and-Edit - OpenAI

**Week 3:** Run benchmarks and compare
**Week 4:** Combine best techniques

---

## Target Metrics

| Priority                 | Metric                | Target |
| ------------------------ | --------------------- | ------ |
| **Speed**          | Latency overhead      | <500ms |
| **Hallucination**  | Detection rate        | >90%   |
| **Tool Selection** | Accuracy              | >85%   |
| **Math**           | Correctness           | >95%   |
| **Orchestration**  | Agent accuracy        | >80%   |
| **Cost**           | $/1K queries | <$0.10 |        |

---

## Current Status

**Version:** 0.1.0-alpha

**What works:**

- Basic LLM integration
- Symbolic verification
- Simple reflection

**In progress:**

- RAG verification
- Benchmarking suite
- Documentation
