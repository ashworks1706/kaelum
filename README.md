# Kaelum

A production-ready reasoning framework combining neural routing, Monte Carlo Tree Search (LATS), domain-specific verification, and self-reflection for robust multi-step problem solving.

Core concepts:

- Query → Neural Router → Expert Worker (LATS) → Verification → Reflection → Result
- Six specialized workers: Math, Logic, Code, Factual, Creative, Analysis
- True MCTS (select / expand / simulate / backprop) per query
- Global semantic tree cache for fast retrieval
- Continuous learning: router trains on outcomes; thresholds are F1-optimized

---

## 🎯 Features

- Neural Router: embedding + structural features predict worker and LATS parameters.
- Expert Workers: LLM-based domain specialists that run LATS to explore reasoning paths.
- LATS (Language Agent Tree Search): MCTS with UCT selection and domain scoring.
- Verification Engine: domain-specific checks (SymPy for math, AST for Python, semantic checks for factual/logic).
- Reflection Engine: error analysis and up to configurable retries to self-correct.
- Tree Cache: stores successful reasoning trees with embeddings; cosine similarity lookup (default threshold 0.85).
- Adaptive Threshold Calibration: grid-search F1 optimization per task type, persisted.
- Active Learning & Fine-tuning: collect traces for training and automatic batch generation.
- Metrics & Analytics: track queries, tokens, cache hit rate, verification rate, etc.

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

- Records (score, threshold, was_correct) per decision.
- After sufficient samples (default 20), runs grid search over thresholds to maximize F1.
- Persists optimal thresholds to `.kaelum/calibration/optimal_thresholds.json`.
- Graceful fallback to base thresholds when samples are insufficient.

---

## LATS & UCT

UCT formula:
UCT(node) = Q(node) / N(node) + c * sqrt(ln N(parent) / N(node))

- Q(node): cumulative reward
- N(node): visit count
- c: exploration constant (default √2)

Default behavior:

- Simulations: 10 (configurable by router)
- Expand with LLM-generated reasoning steps
- Simulate using domain scoring functions
- Backpropagate rewards and extract highest-value path

---

## Tree Cache

- Embeddings via sentence-transformers
- Cosine similarity lookup threshold: 0.85 (configurable)
- Successful trees stored with embeddings and metadata (worker, confidence)
- Cache hit returns full LATS tree, enabling sub-second responses

---

## Active Learning & Fine-Tuning

- collect (query, reasoning, answer) triples for fine-tuning
- Strategies for batch generation: uncertainty, diversity, error, complexity, mixed
- Fine-tune with collected high-quality traces; framework supports export and training pipeline

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
- Typical query latency: 2–5s (uncached); cached queries ~0.001s
- Verification: high accuracy for math (SymPy) and Python AST; honest language support (Python/JS/TS)
- Not yet implemented: parallel LATS (multi-GPU), multi-turn conversation state, ensemble voting, web UI
