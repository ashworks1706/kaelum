
# KaelumAI 🧠

**Reasoning Acceleration Layer for Lightweight LLMs**

> *Make cheap models reason better through verification, not training.*

---

## 🎯 The Problem

Companies using lightweight LLMs (Llama 8B, Mistral 7B, Gemini Flash, Claude Haiku) face:

- ❌ Poor reasoning quality (logical errors, inconsistencies)
- ❌ Hallucinations (making up facts)
- ❌ Wrong tool selection
- ❌ Math errors (multi-step calculations)
- ❌ Bad agent orchestration

**Traditional solutions** (fine-tuning, RLHF) are expensive and not plug-and-play.

---

## 💡 The Solution

Lightweight reasoning layer with:
- ✅ Symbolic verification (SymPy for math)
- ✅ Factual verification (pluggable RAG)
- ✅ Adaptive reflection (self-correction)
- ✅ Confidence scoring
- ✅ Cost optimization (single-LLM design)

**Goals:**
- 🚀 Fast (<500ms overhead)
- 💰 Cheap (single LLM, no multi-model overhead)
- 🔌 Simple (one line to enable)
- 🎯 Precise (no fallbacks, fails loudly)

| Priority | Metric | Target |
|----------|--------|--------|
| **Speed** | Latency overhead | <500ms |
| **Hallucination** | Detection rate | >90% |
| **Tool Selection** | Accuracy | >85% |
| **Math** | Answer correctness | >95% |
| **Orchestration** | Agent accuracy | >80% |
| **Cost** | $/1K queries | <$0.10 |


## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│          User Query                 │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│        enhance() - Public API       │
│  • Mode detection (auto/math/code)  │
│  • Config setup                     │
│  • Cache check                      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│     MCP Orchestrator (Single LLM)   │
├─────────────────────────────────────┤
│  1. ReasoningGenerator              │
│     └─> Generate reasoning trace    │
│                                     │
│  2. VerificationEngine              │
│     ├─> SymbolicVerifier (SymPy)   │
│     └─> FactualVerifier (RAG)      │
│                                     │
│  3. ReflectionEngine                │
│     └─> Self-correction loop        │
│                                     │
│  4. ConfidenceScorer                │
│     └─> Calculate reliability       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      Enhanced Response + Metadata   │
│  • Final answer                     │
│  • Confidence score                 │
│  • Reasoning trace                  │
│  • Verification results             │
└─────────────────────────────────────┘
```

**Key Design Principles:**
- **Single LLM** - Same model for all tasks (cost-efficient)
- **No fallbacks** - Fails loudly for proper debugging
- **Stateless** - Easy to scale horizontally
- **Modular** - Swap verification backends easily

---



### Mode Templates

KaelumAI automatically adjusts prompts based on mode:

| Mode | Best For | Template |
|------|----------|----------|
| `auto` | General queries | Adaptive reasoning |
| `math` | Calculations, equations | Step-by-step math solving |
| `code` | Programming logic | Code-focused reasoning |
| `logic` | Formal reasoning | Logical deduction |
| `creative` | Open-ended tasks | Creative problem-solving |


## 🚀 Quick Start

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:7b

# Clone & Install
git clone https://github.com/ashworks1706/KaelumAI.git
cd KaelumAI
pip install -r requirements.txt
```

### Usage

```python
from kaelum import enhance

# Simple
result = enhance("What is 25% of 80?")

# With mode
result = enhance("Solve: 3x + 5 = 20", mode="math")

# With RAG verification
from kaelum.core.rag_adapter import ChromaAdapter
adapter = ChromaAdapter(collection=my_db)
result = enhance(
    "What is the speed of light?",
    rag_adapter=adapter,
    use_factual_verification=True
)

# CLI
kaelum "What is 15% of 240?"
```

---

## 📊 Benchmarks

```bash
# Run all benchmarks
kaelum-benchmark all

# Run specific
kaelum-benchmark speed
kaelum-benchmark hallucination
kaelum-benchmark math

# Compare models
kaelum-benchmark all --models llama-3.2-3b qwen2.5:7b
```

**5 Priority Areas:**
1. Speed (latency overhead)
2. Hallucination detection
3. Tool selection accuracy
4. Math reasoning
5. Agent orchestration

See `benchmarks/BENCHMARK_GUIDE.md` for details.

---


---

## ⚙️ Configuration

### Environment Variables

```bash
# Optional: Set default API base
export OPENAI_API_BASE=http://localhost:11434/v1

# Optional: Set API key (not needed for Ollama)
export OPENAI_API_KEY=your-key-here
```

### Performance Tuning

```python
config = MCPConfig(
    llm=LLMConfig(
        temperature=0.3,      # Lower = more deterministic
        max_tokens=2048,      # Increase for longer reasoning
    ),
    max_reflection_iterations=3,  # More iterations = better quality
    confidence_threshold=0.80,    # Higher = more strict
)
```

---


### Adding New Features

1. **New Verification Backend**
   - Extend `VerificationEngine` in `kaelum/core/verification.py`
   - Add tests in `tests/test_verification.py`

2. **New RAG Adapter**
   - Implement `RAGAdapter` interface in `kaelum/core/rag_adapter.py`
   - Add example in `examples/`

3. **New Benchmark**
   - Add test cases in `benchmarks/test_cases.py`
   - Extend `BenchmarkSuite` in `benchmarks/benchmark_suite.py`

---

