

# 🧠 KaelumAI

**Fast, verifiable reasoning for small local language models**

---

## 🧩 Overview

KaelumAI is a reasoning-verification runtime that enables small, inexpensive language models (3–8B parameters) to reason accurately and reliably in real time.
Instead of fine-tuning or retraining models, Kaelum adds a lightweight inference-time verification layer that checks, corrects, and stabilizes reasoning as the model runs.

At its core, Kaelum transforms how reasoning works during inference:

It forces LLMs to externalize their thought process through structured reasoning traces.

It verifies these traces using symbolic computation (for math/logic) and factual retrieval (for claims).

It reflects and self-corrects dynamically if confidence is low — all within strict latency budgets.


---

## ⚙️ How It Works

```
User Query
   ↓
Reasoning  →  Verification  →  Reflection  →  Confidence-Scored Answer
```

At runtime Kaelum:

1. Produces a compact **reasoning trace** from the base LLM (step-tagged).
2. **Verifies** the trace:

   * **Symbolic** (e.g., SymPy) for math/logic,
   * **Factual** (RAG) for knowledge claims,
   * **Consistency** across steps/variants.
3. If confidence < threshold, triggers a **bounded self-reflection** pass to correct the answer (0–2 iters).
4. Returns **answer + confidence + provenance**, and logs the structured trace (for debugging and future controller training).

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
    use_symbolic_verification=True,   # math/logic checks
    use_factual_verification=False,   # enable when RAG adapter set
    confidence_threshold=0.7,
)

result = enhance("What is 15% of 200?")
print(result.answer, result.confidence)
print(result.verified_by)     # ['sympy', 'self_reflection'] (example)
print(result.trace_id)        # for logs/telemetry
```

**Return schema (Pythonic):**

```python
class KaelumResult(TypedDict):
    answer: str
    confidence: float           # 0.0 - 1.0
    verified_by: list[str]      # e.g. ["sympy","rag","self_reflection"]
    trace_id: str               # correlates logs
    trace: dict                 # optional, can be toggled off for prod
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

| Parameter  | Description                                      |
| ---------- | ------------------------------------------------ |
| `provider` | `"ollama"`, `"vllm"`, or `"custom"`              |
| `model`    | Model id (e.g., `"llama3.2:3b"`, `"qwen2.5:7b"`) |
| `base_url` | Custom endpoint (optional)                       |

### Generation Settings

* `temperature` 0.0–2.0 (lower → deterministic)
* `max_tokens` 1–128_000 (response length cap)

### Reasoning/Verification Settings

* `max_reflection_iterations`: 0–5 (bounded self-correction depth; default 1–2)
* `confidence_threshold`: 0.0–1.0 (gate for reflection/correction)
* `use_symbolic_verification`: bool (SymPy/logic checks)
* `use_factual_verification`: bool (RAG claims checking)
* `rag_adapter`: pluggable retriever (Chroma/Qdrant/your DB)
* `emit_trace`: bool (include full trace in result for debugging)
* `latency_budget_ms`: soft cap to trim steps/tools under load

---

## 🧠 Core Modules (Runtime Kernel)

| Module                | Purpose                                                                | Notes/Tech                                |
| --------------------- | ---------------------------------------------------------------------- | ----------------------------------------- |
| **Reasoner**          | Wraps the base LLM and yields a **step-tagged reasoning trace**        | Any 3–8B via Ollama/vLLM/API              |
| **Verifier**          | **Symbolic** (SymPy), **Factual** (RAG), **Consistency** checks        | Parallelizable; returns per-step scores   |
| **Reflexor**          | Self-reflection prompts driven by low confidence or failed checks      | Bounded iterations; localized corrections |
| **Confidence Engine** | Aggregates verifier scores + entropy/variance into a single confidence | Pluggable scoring policy                  |
| **Router (Planned)**  | Heuristic → **trainable micro-controller** for tools/agents/depth      | Future “Kaelum Brain” (1–2B policy)       |

---

## 🧱 Architecture Snapshot

```
User Query
   ↓
Context Builder (RAG optional, cacheable)
   ↓
Reasoner (Base LLM; emits step-tagged trace)
   ↓
Verifier (Symbolic math/logic + Factual RAG + Consistency)
   ↓
Reflexor (bounded self-correction if confidence < τ)
   ↓
Confidence Engine → {answer, confidence, verified_by, trace_id}
```

**Design principles:** small-model first, parallel verifiers, bounded loops, observable traces, sub-500ms overhead target.

---


### 🧩 **1. Reasoner — Step-Tagged Thought Generation**

**Purpose:**
The Reasoner is the heart of Kaelum’s inference pipeline. It wraps the base LLM (e.g., Llama 3.2 3B, Qwen 2.5 7B, or Mistral 7B) and standardizes how it produces reasoning traces. Instead of asking the model to “just give an answer,” Kaelum prompts it to *think out loud* in a structured way.

**How it works:**
When a query comes in, the Reasoner builds a prompt that instructs the model to reason step-by-step and label each step with a unique ID (`[Step 1]`, `[Step 2]`, etc.). These are parsed into a JSON-like trace:

```json
{
  "query": "Solve: 2x + 6 = 10",
  "steps": [
    {"id": "s1", "text": "Subtract 6 from both sides → 2x = 4"},
    {"id": "s2", "text": "Divide both sides by 2 → x = 2"}
  ],
  "draft_answer": "x = 2"
}
```

This structured reasoning trace becomes the **observable cognitive state** for Kaelum — a deterministic representation of the model’s internal chain-of-thought.

**Design Rationale:**
By externalizing reasoning, Kaelum gains control over what happens *between* steps — something black-box LLMs hide. The Reasoner never assumes the model is right; it simply creates an interpretable plan for the next modules to evaluate.

**Tech:**

* Any 3–8B LLM via **Ollama**, **vLLM**, or **OpenAI-compatible API**
* Deterministic decoding options (low temperature, top-p sampling)
* Optional JSON-mode parsing for structured reasoning traces
* Streaming support for real-time inspection

---

### 🔍 **2. Verifier — The Cognitive Filter Layer**

**Purpose:**
The Verifier is the first module that *judges* reasoning instead of generating it.
It validates every step in the trace through **symbolic**, **factual**, and **consistency** checks — producing granular confidence scores.

**How it works:**
After receiving a reasoning trace, the Verifier spawns independent processes or async tasks:

1. **Symbolic Verification:**

   * Parses mathematical or logical expressions (e.g., equations, inequalities, variable definitions).
   * Uses **SymPy** or equivalent CAS to check correctness.
   * Detects computational or algebraic inconsistencies.
   * Deterministic (no model required).
2. **Factual Verification:**

   * Extracts factual statements or claims from reasoning steps.
   * Retrieves relevant evidence via a **RAG adapter** (ChromaDB, Qdrant, local documents).
   * Computes semantic similarity (via cross-encoders or embedding cosine similarity).
   * Returns a factual confidence score and optional citations.
3. **Consistency Verification:**

   * Checks whether intermediate steps logically align with previous ones.
   * Compares variable values, assumptions, and final conclusions.
   * Optionally runs short self-consistency tests (e.g., generate 2–3 reasoning variants and compare).

**Design Rationale:**
Verification is the core of Kaelum’s philosophy — the LLM’s output must be *proven* correct, not assumed correct.
By splitting verification across symbolic, factual, and consistency axes, Kaelum decomposes “truth” into testable parts.
This design allows lightweight verifiers to run **in parallel**, yielding near-constant inference latency.

**Tech:**

* Symbolic: **SymPy**, **NumPy**, or custom logic parser
* Factual: **SentenceTransformer**, **Qdrant/Chroma**, cross-encoder reranking
* Consistency: heuristic invariance checks or mini self-consistency runs
* Fully async, parallelized with a thread pool or asyncio

**Output Example:**

```json
{
  "symbolic": {"score": 0.98, "ok": true},
  "factual": {"score": 0.85, "ok": true},
  "consistency": {"score": 0.91, "ok": true}
}
```

---

### 🔁 **3. Reflexor — Self-Reflection and Correction**

**Purpose:**
The Reflexor acts as Kaelum’s *self-awareness mechanism*. It handles cases where verification confidence is too low — meaning the reasoning might be wrong or incomplete. Instead of blindly retrying, the Reflexor *analyzes its own trace*, identifies weak steps, and attempts localized correction.

**How it works:**

1. The Confidence Engine signals low confidence (e.g., <0.7).

2. The Reflexor pinpoints failing steps (from Verifier output).

3. It prompts the LLM again — but not with the original question. Instead, it asks:

   > “You reasoned X, but verification failed at Step 2 (symbolic mismatch). Re-examine that step and correct it.”

4. The LLM reprocesses *only that step* (and dependent ones).

5. The Verifier rechecks the corrected trace, updating confidence.

This process repeats for a bounded number of iterations (usually ≤2) to prevent runaway loops.

**Design Rationale:**
Most self-reflection systems (e.g., Reflexion, CRITIC) perform *entire query restarts*, wasting compute and time.
Kaelum’s Reflexor focuses correction at the *point of failure*, making it precise, interpretable, and latency-safe.
It functions as an **error-correction lens** rather than a rethinking mechanism.

**Tech:**

* Uses lightweight meta-prompts with context injection (original trace + failure reason).
* Can run on the same model or a smaller, faster auxiliary verifier model.
* Configurable iteration limit (`max_reflection_iterations`).

---

### 📈 **4. Confidence Engine — Scoring and Termination Policy**

**Purpose:**
The Confidence Engine aggregates all verification results into a single scalar score between 0 and 1.
This score represents Kaelum’s **belief** in the correctness of the reasoning.
It determines whether to stop (accept answer) or continue (trigger Reflexor).

**How it works:**

* Inputs: symbolic, factual, and consistency scores; optional model entropy and token-level variance.
* Applies a weighted aggregation policy:

  ```python
  confidence = (
      0.6 * symbolic_score +
      0.3 * factual_score +
      0.1 * consistency_score
  )
  ```
* Optionally adjusts based on model’s logit entropy (uncertainty) or prior calibration from historical traces.
* Logs every confidence vector and final decision for training the future controller model.

**Design Rationale:**
Confidence here is *not a heuristic guess* — it’s a measurable fusion of verifier outputs.
This provides interpretability (“why Kaelum trusted this answer”) and creates a **ground truth dataset** for later training the **Kaelum Brain** (controller model).

**Tech:**

* Pluggable scoring policy (YAML/JSON config or learned model).
* Logs scores and confidence decisions to SQLite or Redis.
* Lightweight, O(1) time aggregation.

---

### 🧠 **5. Router (Planned) — The Kaelum Brain**

**Purpose:**
The Router (or Kaelum Brain) is the planned **meta-controller** that learns *how Kaelum should think*.
Instead of fixed heuristics (e.g., “if confidence < 0.7 → reflect once”), it dynamically decides which tools, agents, or reasoning depths to use per query.

**How it will work:**

* Takes as input: query metadata, verifier scores, model latency, and past trace stats.
* Predicts optimal inference strategy (e.g., “use symbolic only,” “call MathAgent,” “increase reflection depth”).
* Learns from logged data over time — effectively **a policy model** controlling the runtime itself.

**Design Rationale:**
This converts Kaelum from a static reasoning engine into an *adaptive system* that improves as it operates.
It becomes capable of trading speed vs. accuracy per task, making it context-aware and cost-efficient.

**Tech (planned):**

* Model: ~1–2B parameter distilled policy network (transformer or MLP hybrid).
* Input features: verifier scores, confidence deltas, metadata embeddings.
* Output: routing actions (tool choice, reflection depth, stopping signal).
* Training data: collected reasoning traces and verifier outcomes.

**Example Behavior:**

* Detects “math” → routes to SymPy and small LLM.
* Detects “open question” → uses RAG verifier and deeper reflection.
* Detects “high latency pressure” → disables reflection and trusts symbolic verifier only.

---

| Module                    | Purpose                                                                                       | Notes/Tech                                                                       |
| ------------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Reasoner**              | Generates structured reasoning traces from base LLMs; makes internal thought visible.         | Supports Ollama/vLLM; JSON/step-tag formatting; deterministic decoding.          |
| **Verifier**              | Performs symbolic, factual, and consistency validation; the “truth filter.”                   | Async tasks; SymPy, vector RAG, self-consistency sampling.                       |
| **Reflexor**              | Localized self-correction loop for failed reasoning steps.                                    | Runs short re-prompts; bounded iterations; avoids full re-generation.            |
| **Confidence Engine**     | Aggregates verifier results into a confidence score and decides whether to accept or reflect. | Weighted score fusion; entropy-aware calibration; logs decisions.                |
| **Router (Kaelum Brain)** | Learns adaptive inference policies — tool choice, depth, reflection strategy.                 | Small controller model (1–2B); trained on trace logs; implements meta-reasoning. |

---

Together, these modules form Kaelum’s **cognitive runtime loop** —
a closed feedback system that reasons, verifies, corrects, and learns *without retraining the base LLM*.


---

## 🧮 Benchmark Focus (internal targets)

| Metric                | Target                     | Meaning                      |
| --------------------- | -------------------------- | ---------------------------- |
| Reasoning Accuracy    | **+30→50% vs baseline**    | GSM8K / MATH subset          |
| Tool/Routing Accuracy | **≥85%**                   | ToolBench-style tasks        |
| Hallucination Rate    | **<10%**                   | TruthfulQA mini / HalluEval  |
| Latency Overhead      | **<500 ms**                | Verifier + reflection budget |
| Cost / 1k queries     | **< $0.10** (local models) | 4-bit quantization + caching |

**Harness layout (planned):**

```
/benchmarks
  /gsm8k       # math eval
  /toolbench   # function/tool eval
  /truthfulqa  # hallucination eval
  runner.py    # unified CLI, CSV/JSON output
```

---

* **Inference-time verification** (symbolic/factual) **before** answer is accepted.
* **Bounded self-reflection** that is latency-aware and localized to failing steps.
* **Model-agnostic runtime** (plug in Qwen/Llama/Mistral or an API).
* **Small-model optimization** (3–8B, quantized, local GPU/CPU friendly).
* **Auditable traces** suitable for debugging, compliance, and eventual controller training.

---

## 🧪 Testing

```bash
python example.py          # default config run
python test_settings.py    # sweep a few parameter sets
pytest -q                  # unit tests as they land
```

---

## ⚡ Performance Notes (dev environment)

* **Local dev (your G14: RTX 4050 6GB, 32GB RAM):**
  Qwen-2.5-7B (Q4) / Llama-3.2-3B (Q4) run comfortably; keep `max_reflection_iterations<=2`.
  Use Redis/SQLite for short-lived caches; enable verifier parallelism.

* **Latency guards:**
  Set `latency_budget_ms` to trim reflection or skip factual checks for “math-only” intents.
  Pre-warm RAG index and SymPy imports to avoid cold-start spikes.

---

## 🔮 Roadmap (Internal)

1. **MVP kernel**: Reasoner → Verifier → Reflexor → Confidence; math focus.
2. **Benchmark runner**: GSM8K/TruthfulQA/ToolBench mini-suites + CSV.
3. **Caching & budgets**: KV reuse, LRU, latency-aware cuts.
4. **Router (Kaelum Brain v0)**: heuristic routing; log traces for training.
5. **Domain agents**: Math/Fact/Logic minimal agents (prompt-specialized).
6. **Controller training**: 1–2B policy learns tool choice, depth, and verifier mix.
7. **SDK hardening**: clean API, adapters (LangChain/LlamaIndex), docs.

---

## 🧭 Non-Goals (for now)

* Full general multi-agent ecosystems or heavy UI suites.
* Training large models or dependence on proprietary frontier models.
* Long unbounded “think-forever” loops; we prioritize bounded, explainable control.

---

## 🔐 Privacy & Deployment

* Self-hostable by default (local inference + local RAG).
* No exfiltration of traces unless explicitly configured.
* Toggle `emit_trace=False` for production responses while still logging server-side.

---

## Appendix – Reflection Pseudocode (bounded, localized)

```python
def enhance(query):
    trace = reasoner.run(query)                         # step-tagged draft
    scores = verifier.score(trace)                      # symbolic/factual/consistency
    conf = confidence_engine.aggregate(scores)

    iters = 0
    while conf < THRESHOLD and iters < MAX_REFLECTION_ITERS:
        failing = verifier.failing_regions(trace, scores)
        trace = reflexor.correct(trace, failing, evidence=scores.evidence)
        scores = verifier.score(trace)
        conf = confidence_engine.aggregate(scores)
        iters += 1

    return package_result(trace, conf, scores)
```

