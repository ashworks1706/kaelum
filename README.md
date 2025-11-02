

# 🧠 KaelumAI

**Make any LLM reason better - Function calling tool for commercial LLMs**

---

## 🧩 Overview

KaelumAI is a reasoning enhancement tool that makes **any LLM** (GPT-4, Gemini, Claude, etc.) reason better by providing them with a specialized reasoning function. 

**Key Innovation:** Instead of making commercial LLMs do all the work, Kaelum acts as a **reasoning middleware** - when your commercial LLM faces a complex problem, it can call Kaelum, which uses a lightweight local model to generate step-by-step reasoning traces. Your LLM then uses these verified reasoning steps to produce a better final answer.

### How It Works

```
User Question → Commercial LLM (Gemini/GPT-4/Claude)
                      ↓ (recognizes complex reasoning needed)
                Calls Kaelum Tool
                      ↓
            Kaelum (Local 3-8B Model)
                      ↓
    Generates Verified Reasoning Steps
                      ↓
          Returns to Commercial LLM
                      ↓
    LLM Uses Steps for Better Answer
```

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

### Use with Gemini

```python
import google.generativeai as genai
from kaelum import set_reasoning_model, kaelum_enhance_reasoning, get_function_schema

# 1. Set up Kaelum's local reasoning model
set_reasoning_model(
    provider="ollama",
    model="qwen2.5:7b",
    temperature=0.3,
)

# 2. Configure Gemini with Kaelum as a tool
genai.configure(api_key="your-api-key")

kaelum_tool = genai.protos.Tool(
    function_declarations=[
        genai.protos.FunctionDeclaration(
            name="kaelum_enhance_reasoning",
            description="Enhances reasoning for complex math, logic, or multi-step problems",
            parameters=get_function_schema(format="gemini")
        )
    ]
)

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    tools=[kaelum_tool]
)

# 3. Ask Gemini a complex question
chat = model.start_chat()
response = chat.send_message("If a train travels at 60 mph for 2.5 hours, then 80 mph for 1.5 hours, what's the total distance?")

# 4. If Gemini calls Kaelum tool, execute it
if response.candidates[0].content.parts[0].function_call:
    function_call = response.candidates[0].content.parts[0].function_call
    
    # Execute Kaelum reasoning
    result = kaelum_enhance_reasoning(
        query=function_call.args["query"],
        domain=function_call.args.get("domain", "general")
    )
    
    # Send reasoning back to Gemini
    function_response = genai.protos.Part(
        function_response=genai.protos.FunctionResponse(
            name="kaelum_enhance_reasoning",
            response={"result": result}
        )
    )
    
    # Get Gemini's enhanced answer
    final_response = chat.send_message(function_response)
    print(final_response.text)
```

### Use with OpenAI/GPT-4

```python
from openai import OpenAI
from kaelum import set_reasoning_model, kaelum_enhance_reasoning, get_function_schema

# Set up Kaelum
set_reasoning_model(provider="ollama", model="qwen2.5:7b")

# Set up OpenAI
client = OpenAI(api_key="your-api-key")

# Define Kaelum as a tool
tools = [{
    "type": "function",
    "function": get_function_schema(format="openai")
}]

# Chat with GPT-4
messages = [
    {"role": "system", "content": "Use kaelum_enhance_reasoning for complex problems."},
    {"role": "user", "content": "Your complex question here"}
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools
)

# Handle tool calls (similar to Gemini example)
```

### Standalone Use (No Commercial LLM)

You can also use Kaelum directly for quick reasoning:

```python
from kaelum import enhance_stream, set_reasoning_model

set_reasoning_model(provider="ollama", model="qwen2.5:7b")

# Stream reasoning output
for chunk in enhance_stream("What is 25% of 80?"):
    print(chunk, end='', flush=True)
```

---

## 💡 Use Cases

### 1. Math & Computation
```python
query = "Calculate compound interest: $5000 at 3.5% annually for 10 years"
# Kaelum generates verified step-by-step calculation
# Your LLM uses it to explain the answer clearly
```

### 2. Logical Reasoning
```python
query = "If all A are B, and some B are C, can we conclude all A are C?"
# Kaelum breaks down logical steps
# Your LLM formats a complete explanation
```

### 3. Multi-Step Problems
```python
query = "Plan a 3-day trip with budget $500, considering transport, food, and lodging"
# Kaelum structures the reasoning steps
# Your LLM adds creativity and personalization
```

---

## ⚙️ Configuration

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

