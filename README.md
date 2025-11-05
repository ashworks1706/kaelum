# Kaelum

A production-ready reasoning framework combining neural routing, Monte Carlo Tree Search, domain-specific verification, and self-reflection for robust multi-step problem solving.

**What is this?** Kaelum is an AI reasoning system that combines multiple AI techniques to solve complex problems step-by-step. It's like having multiple expert assistants (math, code, logic, etc.) working together, where each assistant explores different solution paths and the system verifies answers before returning them.

Core concepts:

- Query → Neural Router → Expert Worker (LATS) → Verification → Reflection → Result
- Six specialized workers: Math, Logic, Code, Factual, Creative, Analysis
- **MCTS** (Monte Carlo Tree Search): A search algorithm that explores multiple solution paths by building a tree of possibilities, commonly used in game AI like AlphaGo
- **Global semantic tree cache**: Stores previously solved problems using AI embeddings (numerical representations of meaning) for instant retrieval of similar queries
- Continuous learning: router trains on outcomes; thresholds are F1-optimized

---

## 🎯 Features

- **Neural Router**: A deep learning model using embeddings (vector representations of text meaning) and structural features to intelligently select which expert worker should handle each query and predict optimal search parameters.
- **Expert Workers**: Six LLM-based (Large Language Model) domain specialists that run LATS to explore multiple reasoning paths in parallel.
- **LATS (Language Agent Tree Search)**: An adaptation of MCTS for language reasoning - explores different solution paths, scores them using domain-specific metrics, and selects the best one.
- **Verification Engine**: Domain-specific correctness checks - uses SymPy (symbolic mathematics library) for math, AST (Abstract Syntax Tree - code structure representation) for Python, and semantic similarity checks for logic/factual content.
- **Reflection Engine**: When verification fails, analyzes errors and generates improved reasoning steps, then retries (up to configurable iterations) - essentially "learning from mistakes."
- **Tree Cache**: Stores successful reasoning trees with embeddings; uses cosine similarity (measures how similar two vectors are, 0-1 scale) for fast lookup (default threshold 0.85).
- **Adaptive Threshold Calibration**: Automatically finds optimal decision thresholds by maximizing F1 score (harmonic mean of precision and recall - a measure of classification accuracy).
- **Active Learning & Fine-tuning**: Intelligently selects valuable examples for training and generates batches for model fine-tuning.
- **Metrics & Analytics**: Comprehensive tracking of queries, tokens (text units processed), cache hit rate, verification rate, etc.

---

## Quick Start

Install runtime dependencies and clone repo:

```bash
# System requirements: Python 3.8+, Ollama or vLLM (recommended)
curl -fsSL https://ollama.com/install.sh | sh    # Ollama (example)
git clone https://github.com/ashworks1706/KaelumAI.git
cd KaelumAI
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure `.env`:

```bash
cp .env.example .env
# Edit .env to point to LLM backend and tune settings:
# LLM_BASE_URL=http://localhost:11434/v1
# LLM_MODEL=qwen2.5:3b
# MAX_REFLECTION_ITERATIONS=2
# USE_SYMBOLIC_VERIFICATION=true
```

Run:

```bash
python run.py
# or with Docker
./docker-run.sh
```

---

## Example: Python API

```python
from kaelum import enhance, set_reasoning_model, get_metrics

# Optional: configure model / router settings
set_reasoning_model(
  base_url="http://localhost:11434/v1",
  model="qwen2.5:3b",
  temperature=0.7,
  max_tokens=2048,
  enable_routing=True,
  use_symbolic_verification=True,
  max_reflection_iterations=2,
)

# Solve a query
result = enhance("What is the derivative of x^2 + 3x?")
print(result)

# Inspect metrics
metrics = get_metrics()
print(metrics["analytics"])
```

---

## Architecture Overview

Top-level components:

- core/router.py — PolicyNetwork: routes queries and predicts LATS depth and simulations.
- core/lats.py — LATS implementation (MCTS).
- core/workers.py & specialized workers — domain logic + prompts.
- core/verification.py — domain validators (SymPy, AST, embedding checks).
- core/tree_cache.py — semantic cache with cosine similarity lookup.
- core/reflection.py — error analysis and self-correction loop.
- runtime/orchestrator.py — pipeline orchestration and training data export.

Data flow:

1. Router embeds query and selects a worker + parameters.
2. Check global tree cache: if match (cosine > 0.85) return cached tree.
3. Run LATS (default 10 simulations) building multiple paths.
4. Verify candidate path with domain-specific rules.
5. If verification fails, reflection produces improved steps and retries (up to configured iterations).
6. Record outcomes for router training and threshold calibration.

---

## Verification & Reflection

Verification samples:

- Math: SymPy symbolic checks (equivalence, derivatives, integrals).
- Code: AST parse + language-specific checks (Python, JS, TS supported).
- Logic / Factual / Creative / Analysis: semantic checks, conclusion detection, specificity.

Reflection loop:

- Identify verification issues
- Generate revised reasoning steps
- Retry until pass or max iterations

---

## Adaptive Threshold Calibration

**What are thresholds?** In classification tasks (e.g., "Is this a math query?"), models output a confidence score (0-1). The threshold determines the cutoff - scores above it predict "yes", below predict "no".

How Kaelum optimizes thresholds:

- Records (confidence score, threshold used, whether prediction was correct) for every decision
- After sufficient samples (default 20), runs **grid search**: tests many threshold values (0.20, 0.25, ..., 0.85)
- Calculates **F1 score** for each threshold: `F1 = 2 * (precision * recall) / (precision + recall)`
  - **Precision**: Of predictions we made, how many were correct?
  - **Recall**: Of actual positives, how many did we find?
  - **F1 score**: Balances both metrics (1.0 = perfect, 0.0 = useless)
- Selects threshold that maximizes F1 score
- Persists optimal thresholds to `.kaelum/calibration/optimal_thresholds.json`
- Graceful fallback to default thresholds when data is insufficient

---

## LATS & UCT

**What is UCT?** UCT (Upper Confidence Bound applied to Trees) is the selection algorithm that decides which path to explore next in the search tree. It balances exploitation (following promising paths) with exploration (trying untested options).

UCT formula:
```
UCT(node) = Q(node) / N(node) + c * sqrt(ln N(parent) / N(node))
```

- **Q(node)**: Cumulative reward from all simulations through this node (how good this path has been)
- **N(node)**: Visit count (how many times we've explored this node)
- **c**: Exploration constant (default √2) - higher values encourage more exploration
- **First term** (Q/N): Exploitation - prefer nodes with high average reward
- **Second term**: Exploration - prefer less-visited nodes to discover new paths

Default behavior:

- Simulations: 10 per query (router can increase for complex problems)
- Expand: LLM generates next reasoning steps from current node
- Simulate: Score the reasoning path using domain-specific reward functions
- Backpropagate: Update all ancestor nodes with the reward, helping future selection

---

## Tree Cache

**How it works:** The cache stores successful reasoning trees using semantic embeddings (vector representations that capture meaning, not just words). When a new query arrives, it's converted to an embedding and compared against cached queries.

- **Embeddings**: Generated via sentence-transformers (a neural network that converts text to fixed-length vectors)
- **Cosine similarity**: Measures how "close" two embeddings are in vector space (1.0 = identical, 0.0 = completely different)
- **Lookup threshold**: 0.85 (queries with similarity ≥ 0.85 retrieve cached solution)
- Successful trees stored with embeddings, metadata (worker type, confidence), and full reasoning trace
- **Cache hit**: Returns complete LATS tree instantly (~0.001s instead of 2-5s for new search)
- **Cross-domain caching**: A math solution can accelerate similar logic or analysis queries if semantically close

---

## Active Learning & Fine-Tuning

**What is active learning?** Instead of training on random data, intelligently select the most valuable examples (queries where the model struggled, diverse examples, complex reasoning, etc.) to maximize learning efficiency.

How Kaelum collects training data:

- Automatically captures (query, reasoning steps, answer) triples during operation
- **Selection strategies** for generating training batches:
  - **Uncertainty**: Queries where model had low confidence - helps improve weak areas
  - **Diversity**: Semantically diverse queries via max-min distance sampling - ensures broad coverage
  - **Error**: Failed verification attempts with reflection improvements - learns from mistakes
  - **Complexity**: High tree depth, many simulations, multi-step reasoning - trains on hard problems
  - **Mixed**: Balanced combination of all strategies (recommended)
- Export formatted datasets for fine-tuning with Hugging Face Transformers, OpenAI, etc.
- Fine-tuned models show improved performance on domain-specific reasoning tasks

---

## Project Layout (abridged)

```
Kaelum/
├── core/
│   ├── router.py
│   ├── lats.py
│   ├── workers.py
│   ├── tree_cache.py
│   ├── verification.py
│   ├── reflection.py
│   └── threshold_calibrator.py
├── runtime/
│   └── orchestrator.py
├── requirements.txt
└── README.md
```

---

## Testing & Development

```bash
pip install pytest pytest-cov
python -m pytest -v
python -m pytest --cov=core --cov=runtime
```

Contributing:

1. Fork repo
2. Create branch: git checkout -b feature-name
3. Run tests and open a PR

---

## Performance & Limits

- Default LATS simulations: 10 (router can increase for complex queries)
- Typical query latency: 2–5s (uncached); cached queries ~0.001s (1000x faster)
- Verification: High accuracy for math (SymPy symbolic validation) and Python AST parsing
- Language support: Python, JavaScript, TypeScript for code verification
- **Not yet implemented**: Parallel LATS (multi-GPU tree search), multi-turn conversation state tracking, ensemble voting across multiple models, web UI dashboard

---

## Research & References

Kaelum builds upon several key research areas in AI and reasoning:

- [Browne et al. (2012): "A Survey of Monte Carlo Tree Search Methods"](https://ieeexplore.ieee.org/document/6145622)
- [Silver et al. (2016): "Mastering the game of Go with deep neural networks and tree search" (AlphaGo)](https://www.nature.com/articles/nature16961)
- [Wei et al. (2022): "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"](https://arxiv.org/abs/2201.11903)
- [Yao et al. (2023): "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"](https://arxiv.org/abs/2305.10601)
- [Shinn et al. (2023): "Reflexion: Language Agents with Verbal Reinforcement Learning"](https://arxiv.org/abs/2303.11366)
- [Madaan et al. (2023): "Self-Refine: Iterative Refinement with Self-Feedback"](https://arxiv.org/abs/2303.17651)
- [Shazeer et al. (2017): "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"](https://arxiv.org/abs/1701.06538)
- [Fedus et al. (2021): "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"](https://arxiv.org/abs/2101.03961)
- [Welleck et al. (2022): "Symbolic Knowledge Distillation: from General Language Models to Commonsense Models"](https://arxiv.org/abs/2110.07178)
- [Settles (2009): "Active Learning Literature Survey"](https://minds.wisconsin.edu/handle/1793/60660)
- [Reimers & Gurevych (2019): "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"](https://arxiv.org/abs/1908.10084)
