# Kaelum

Kaelum is a research framework for building reasoning systems with learned routing policies. The core idea is to treat the meta-control of the reasoning pipeline (which strategy to use, how much verification, when to reflect) as a learnable problem rather than a fixed heuristic. This repo provides a minimal, developer-focused scaffold for experimentation.

---

## Highlights

- Learned neural router that selects reasoning strategies per query.
- Parallel verification layers: symbolic (SymPy), factual (RAG), consistency checks.
- Localized reflection that fixes only failing steps (bounded iterations).
- LATS: lightweight, domain-agnostic tree search (caller provides simulator & expander).
- Outcome logging for offline supervised / bandit training of the router.

---

## Quick concepts

- Generate → Verify → Reflect:

  1. Reasoner produces a step-tagged trace (LLM).
  2. Verifiers evaluate each step (symbolic / factual / consistency).
  3. Confidence engine aggregates scores; if low, Reflexor attempts localized corrections.
  4. Router (neural or baseline) decides strategy before execution.
- LATS (Local Agent Tree Search): MCTS-style search without built-in simulation — the developer must provide `simulator(node) -> float` and `expand_fn(parent_node) -> child_state`.

---

## Key files (concise)

- `kaelum/__init__.py` — public API and helpers.
- `kaelum/core/neural_router.py` — feature extraction, PyTorch PolicyNetwork, NeuralRouter, outcome recording.
- `kaelum/core/router_policy.py` — abstract `RouterPolicy` interface.
- `kaelum/core/router.py` — enums & data structures (QueryType, ReasoningStrategy, RoutingDecision).
- `kaelum/runtime/orchestrator.py` — orchestration and environment wiring.
- `kaelum/core/reasoning.py` — LLM interface and structured reasoning traces.
- `kaelum/core/verification.py` — symbolic, factual, consistency checks.
- `kaelum/core/reflection.py` — localized self-correction logic.
- `kaelum/core/lats.py` — LATS tree search implementation (requires caller-provided simulator/expander).
- `kaelum/core/neural_router_trainer.py` — training utilities for the policy network.

---

## Requirements & setup

Minimum:

- Python 3.8+
- PyTorch (required for neural router)
- SymPy (symbolic verifier)
- sentence-transformers (optional, falls back to zero embeddings)

Quick install (example):

```bash
python -m venv .venv
# fish:
source .venv/bin/activate.fish
# or bash:
source .venv/bin/activate
pip install -r requirements.txt
```

Sanity import:

```bash
python -c "import kaelum; print('import OK')"
```

Note: the code assumes PyTorch is present. If you need importability without PyTorch, add a guard in `neural_router.py`.

---

## Neural Router (overview)

- Input: 398-dim feature vector (384-dim embedding + ~14 handcrafted features).
- Model: lightweight MLP (residuals, multi-head outputs).
- Output: routing decision (strategy choice, reflection depth, verification flags, confidence threshold).
- Training: outcomes appended to `data_dir/outcomes.jsonl` for offline supervised/bandit training.

Record outcomes after each run:

```python
router = NeuralRouter(model_path=None, data_dir='.kaelum/neural_routing', device='cpu')
decision = router.route('Calculate the derivative of x^2 + 3x', context={})
# exec decision via orchestrator...
result = {'success': True, 'latency_ms': 245, 'confidence': 0.92}
router.record_outcome(decision, result)
```

---

## Reasoning, Verification & Reflection (concise)

Reasoner

- Produces structured step-tagged traces, e.g.:

```json
{
  "query": "Solve: 2x + 6 = 10",
  "steps": [{"id": "s1", "text": "Subtract 6 → 2x = 4"}, {"id": "s2", "text": "Divide by 2 → x = 2"}],
  "draft_answer": "x = 2"
}
```

Verifier

- Runs three parallel checks per step:
  - Symbolic (SymPy)
  - Factual (RAG + similarity)
  - Consistency (internal coherence)
- Each verifier returns a confidence score; the Confidence Engine aggregates them (weighted fusion).

Reflexor

- If aggregate confidence < threshold, it pinpoints failing steps and re-prompts the LLM to correct only those steps (bounded iterations, typically ≤2).

Output example (final result):

```json
{
  "reasoning_steps": [...],
  "verification": {"passed": true, "checks": {"symbolic": true, "factual": true, "consistency": true}},
  "reflection": null,
  "final_answer": "The total cost is $42.09"
}
```

---

## LATS (Local Agent Tree Search)

- Purpose: per-agent exploration (MCTS-like).
- Important: LATS is domain-agnostic and requires:
  - `simulator(node) -> float` (evaluate node, return reward)
  - `expand_fn(parent_node) -> child_state` (generate child states)
- Minimal usage pattern:

```python
from core.lats import LATS

def simulator(node): return evaluate(node)
def expand_fn(parent): return next_state

tree = LATS(root_state={'step':0}, simulator=simulator, expand_fn=expand_fn)
node = tree.select()
child = tree.expand(node, expand_fn(node))
reward = tree.simulate(child)
tree.backpropagate(child, reward)
best = tree.best_child()
```

LATS intentionally avoids embedding simulation logic so the search model is explicit and testable.

---

## Design decisions (brief)

- Learned routing instead of static heuristics: adapts to task distributions, costs, and model capabilities.
- Localized reflection: correct only failing steps to save compute and preserve valid work.
- Offline training of router: avoids inference-time RL complexity; logs outcomes for safe, batched learning.
- Small-model first: target 3–8B models, quantization and caching for low cost.

---

## Development notes & next steps

Suggested roadmap:

1. Implement a training harness to read `outcomes.jsonl` and train the PolicyNetwork.
2. Add benchmarks (GSM8K, ToolBench) for routing accuracy.
3. Improve feature engineering and experiment with contextual bandits before full RL.
4. Add deterministic LATS tests and fixture-based router tests.

If desired, a small contextual-bandit baseline and a replay-buffer logging scaffold can be added next.

---

## Quick dev steps

1. Create venv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run tests (if available):

```bash
python -m pytest -q
```

3. Start developing:

- Implement a policy by subclassing `core.router_policy.RouterPolicy`.
- Use `kaelum/runtime/orchestrator.py` as the environment for evaluation.
