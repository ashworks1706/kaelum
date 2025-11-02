# 🧠 KaelumAI

**Make any LLM reason better - Function calling tool for commercial LLMs**

---

## 🧩 Overview

KaelumAI is a reasoning enhancement tool that makes **any LLM** (GPT-4, Gemini, Claude, etc.) reason better by providing them with a specialized reasoning function.

**Key Innovation:** Instead of making commercial LLMs do all the work, Kaelum acts as a **reasoning middleware** - when your commercial LLM faces a complex problem, it can call Kaelum, which uses a lightweight local model to generate step-by-step reasoning traces. Your LLM then uses these verified reasoning steps to produce a better final answer.

### Why This Architecture?

1. **💰 Cost Effective**: Heavy reasoning runs on your local model, not expensive API calls
2. **⚡ Fast**: Small local models (3-8B) are extremely fast for reasoning generation
3. **🔒 Private**: Reasoning computation happens locally, not sent to external APIs
4. **🎯 Accurate**: Verified reasoning steps reduce hallucinations and errors
5. **🔧 Flexible**: Works with ANY commercial LLM that supports function calling

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/ashworks1706/KaelumAI.git
cd KaelumAI
pip install -e .

# Install local model (for reasoning generation)
ollama pull qwen2.5:7b && ollama serve

# Install your commercial LLM SDK
pip install google-generativeai  # for Gemini
# OR
pip install openai               # for GPT-4/Claude
```

## ⚙️ Configuration

### Model Settings

| Parameter    | Description                                         |
| ------------ | --------------------------------------------------- |
| `provider` | `"ollama"`, `"vllm"`, or `"custom"`           |
| `model`    | Model id (e.g.,`"llama3.2:3b"`, `"qwen2.5:7b"`) |
| `base_url` | Custom endpoint (optional)                          |

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

| Module                      | Purpose                                                                           | Notes/Tech                                |
| --------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------- |
| **Reasoner**          | Wraps the base LLM and yields a**step-tagged reasoning trace**              | Any 3–8B via Ollama/vLLM/API             |
| **Verifier**          | **Symbolic** (SymPy), **Factual** (RAG), **Consistency** checks | Parallelizable; returns per-step scores   |
| **Reflexor**          | Self-reflection prompts driven by low confidence or failed checks                 | Bounded iterations; localized corrections |
| **Confidence Engine** | Aggregates verifier scores + entropy/variance into a single confidence            | Pluggable scoring policy                  |
| **Router (Planned)**  | Heuristic →**trainable micro-controller** for tools/agents/depth           | Future “Kaelum Brain” (1–2B policy)    |

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

##### Our LLM List for this step (so far)

| Name                       | Size   | CoT Score (/10) | Latency Score (/10) | Cost Score (/10) | Tool Use Score (/10) | Speciality              | Reason                                                                                                  |
| -------------------------- | ------ | --------------- | ------------------- | ---------------- | -------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------- |
| **Llama 3.2 0.5 B**        | 0.5 B  | 3.5             | **10**              | **10**           | 3                    | ultra-light baseline    | Tests minimal reasoning floor; blazing fast but very weak logic.                                        |
| **Qwen 2.5 0.5 B**         | 0.5 B  | 4.2             | 9.5                 | 9.5              | 4                    | ultra-cheap generalist  | Slightly smarter than Llama 0.5 B; strong structural formatting.                                        |
| **TinyLlama 1.1 B**        | 1.1 B  | 4.6             | 9.2                 | 9.3              | 4                    | efficient micro-model   | 1 B-scale open model (Apache 2.0); optimized for speed & low VRAM; solid baseline for controller tests. |
| **Llama 3.2 1.5 B**        | 1.5 B  | 4.8             | 9                   | 9                | 4                    | small generalist        | Cheap & fast; useful baseline for reasoning-vs-latency curves.                                          |
| **Qwen 2.5 1.5 B**         | 1.5 B  | 5.4             | 9                   | 9                | 5                    | small balanced          | Better CoT than Llama 1.5 B; possible controller fine-tune sandbox.                                     |
| **Phi-3 Mini**             | ~3.8 B | 6.5             | 8                   | 8.5              | 6                    | efficient mid-size      | Ideal Kaelum-Lite model: sub-4 GB VRAM, fast, good logic for cost.                                      |
| **Mistral 7 B Instruct**   | 7 B    | 7.5             | 7                   | 7                | 7                    | general reasoning       | Stable Apache-2.0 baseline; balanced across reasoning and cost.                                         |
| **Qwen 2.5 7 B Instruct**  | 7 B    | **8.4**         | 6.8                 | 7                | **8**                | generalist + math       | ⭐ **Recommended base** — best reasoning & tool-use trade-off.                                           |
| **Mathstral 7 B**          | 7 B    | 8.1             | 6.5                 | 6.5              | 5                    | math specialist         | Excels on GSM8K/MATH; perfect for Kaelum’s symbolic check mode.                                         |
| **DeepSeek-Math 7 B**      | 7 B    | 7.9             | 6.5                 | 6.5              | 5                    | math specialist         | GRPO-trained math head; complementary to Mathstral.                                                     |
| **Llama 3.1 8 B Instruct** | 8 B    | **8.6**         | 6                   | 6                | **8**                | high-quality generalist | Strongest small-model reasoning; slightly heavier VRAM (≈ 8 GB Q4).                                     |
| **Gemma 2 9 B**            | 9 B    | 8.5             | 5.8                 | 5.5              | 8                    | Google-tier quality     | Upper-bound reference for local inference performance ceiling.                                          |

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
   >
4. The LLM reprocesses *only that step* (and dependent ones).
5. The Verifier rechecks the corrected trace, updating confidence.

This process repeats for a bounded number of iterations (usually ≤2) to prevent runaway loops.

**Design Rationale:**
Most self-reflection systems (e.g., Reflexion, CRITIC) perform *entire query restarts*, wasting compute and time.
Kaelum’s Reflexor focuses correction at the *point of failure*, making it precise, interpretable, and latency-safe.
It functions as an **error-correction lens** rather than a rethinking mechanism.

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

| Module                          | Purpose                                                                                       | Notes/Tech                                                                        |
| ------------------------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **Reasoner**              | Generates structured reasoning traces from base LLMs; makes internal thought visible.         | Supports Ollama/vLLM; JSON/step-tag formatting; deterministic decoding.           |
| **Verifier**              | Performs symbolic, factual, and consistency validation; the “truth filter.”                 | Async tasks; SymPy, vector RAG, self-consistency sampling.                        |
| **Reflexor**              | Localized self-correction loop for failed reasoning steps.                                    | Runs short re-prompts; bounded iterations; avoids full re-generation.             |
| **Confidence Engine**     | Aggregates verifier results into a confidence score and decides whether to accept or reflect. | Weighted score fusion; entropy-aware calibration; logs decisions.                 |
| **Router (Kaelum Brain)** | Learns adaptive inference policies — tool choice, depth, reflection strategy.                | Small controller model (1–2B); trained on trace logs; implements meta-reasoning. |

---

Together, these modules form Kaelum’s **cognitive runtime loop** —
a closed feedback system that reasons, verifies, corrects, and learns *without retraining the base LLM*.

---

## 🧮 Benchmark Focus (internal targets)

| Metric                | Target                           | Meaning                      |
| --------------------- | -------------------------------- | ---------------------------- |
| Reasoning Accuracy    | **+30→50% vs baseline**   | GSM8K / MATH subset          |
| Tool/Routing Accuracy | **≥85%**                  | ToolBench-style tasks        |
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
