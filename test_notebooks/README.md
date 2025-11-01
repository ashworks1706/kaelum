# 🧪 Test Notebooks

**Comprehensive testing suite for KaelumAI development**

## 📚 Notebook Index

### 01_llm_selection.ipynb
**Goal:** Choose the best open-source LLM for the project
- Speed benchmarks across models
- Reasoning quality comparison
- Math accuracy tests
- Decision matrix with recommendations

### 02_benchmark_testing.ipynb
**Goal:** Test against standard benchmarks (GSM8K, TruthfulQA, ToolBench)
- Speed: < 500ms overhead
- Math accuracy: > 90%
- Hallucination reduction: > 90%
- Logic reasoning tests

### 03_verification_testing.ipynb
**Goal:** Test symbolic verification and RAG adapters
- SymPy integration testing
- Error detection accuracy
- False positive/negative analysis
- RAG adapter setup (ChromaDB, Qdrant)

### 04_reflection_testing.ipynb
**Goal:** Test self-reflection and iterative improvement
- Reflection trigger conditions
- Quality improvement measurement
- Optimal iteration count
- Confidence score calibration

### 05_performance_optimization.ipynb
**Goal:** Identify bottlenecks and optimize speed
- Component latency breakdown
- Token usage efficiency
- Temperature vs speed analysis
- Caching opportunities

### 06_integration_edge_cases.ipynb
**Goal:** Test real-world scenarios and edge cases
- Error handling and graceful failures
- Unusual inputs and paradoxes
- Mode switching (math/code/logic)
- API integration patterns

## 🚀 Getting Started

1. **Pull required models:**
```bash
ollama pull llama3.2:3b
ollama pull qwen2.5:7b
```

2. **Start Jupyter:**
```bash
jupyter notebook test_notebooks/
```

3. **Run notebooks in order:**
   - Start with `01_llm_selection.ipynb` to choose your model
   - Then run benchmarks and component tests
   - Document findings in each notebook's markdown cells

## 📝 Documentation Workflow

Each notebook has:
- **Test cells** - Pre-configured experiments
- **Markdown cells** - Document your findings
- **Summary sections** - Track pass/fail status

Fill in the markdown sections as you test to create a complete testing report.

## 🎯 Sprint Alignment

These notebooks align with TODO.md sprints:
- **Sprint 1 (Core MVP):** Notebooks 01, 02
- **Sprint 2 (Verification):** Notebook 03
- **Sprint 3 (Optimization):** Notebooks 04, 05
- **Sprint 4 (Testing):** Notebook 06

## 🤝 Team Workflow

**Suggested approach:**
1. Ash: LLM selection + benchmarks (01, 02)
2. r3tr0: Verification + reflection (03, 04)
3. wsb: Performance + integration (05, 06)

Share findings in Discord after each notebook!
