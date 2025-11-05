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

## Features

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

## Complete Workflow: From Query to Answer

This section walks through exactly what happens when you ask Kaelum a question, explaining both the **what** (steps) and **why** (concepts behind each step).

### Example Query: "What is the derivative of x² + 3x with respect to x?"

#### **Step 1: Query Embedding & Feature Extraction**

```
Input: "What is the derivative of x² + 3x with respect to x?"
```

**What happens:**

- The router converts your question into a **384-dimensional embedding vector** using a sentence transformer model
- It also extracts **structural features**: query length, presence of math symbols (∂, ∫, √), code keywords, etc.
- These features are concatenated into a **398-dimensional feature vector**

**Why this matters:**
Embeddings capture semantic meaning (not just keywords). "Find d/dx of x²" and "differentiate x squared" have similar embeddings even though they use different words. This lets the router understand intent, not just match patterns.

#### **Step 2: Neural Router Selection**

```
Router → PolicyNetwork(398-dim) → [Worker: "math", Depth: 5, Simulations: 10]
```

**What happens:**

- The 398-dim feature vector goes through a **neural network** (2 hidden layers: 398→256→128)
- Network outputs:
  - **Worker probabilities**: [math: 0.92, logic: 0.04, code: 0.02, ...]
  - **Tree depth**: 5 (how deep to search)
  - **Simulations**: 10 (how many paths to explore)
- Selects "math" worker with 92% confidence

**Why this matters:**
The router is a **learned model** that improves over time. It trains on every query outcome using gradient descent. If it routes a calculus question to the "logic" worker and verification fails, it updates its weights to prefer "math" worker for similar queries. This is **continual learning** - the system gets smarter with use.

#### **Step 3: Cache Lookup**

```
Query embedding → Cosine similarity against cached queries → Check if similarity ≥ 0.85
```

**What happens:**

- System compares query embedding with all cached successful solutions
- Uses **cosine similarity**: `sim = (A · B) / (||A|| × ||B||)`
- If similarity ≥ 0.85 threshold, returns cached tree instantly

**Example:** If you previously asked "derivative of x²", the current query has ~0.91 similarity

- **Cache hit** → Return answer in 0.001s (skip Steps 4-7)
- **Cache miss** → Continue to LATS

**Why this matters:**
Traditional caches match exact strings. Semantic caching matches **meaning**. "What's d/dx of x²?" and "Differentiate x squared" both hit the same cache entry. This gives **1000x speedup** on similar queries while handling natural language variation.

#### **Step 4: LATS - Monte Carlo Tree Search**

```
Root: "derivative of x² + 3x"
 ├─ Node 1: "Apply power rule to x²" [Q=0.85, N=3]
 ├─ Node 2: "Use first principles" [Q=0.62, N=2]
 └─ Node 3: "Apply sum rule first" [Q=0.91, N=5] ← Best path
```

**What happens (10 simulations):**

**Simulation 1-3: Initial Exploration**

- Start at root node
- LLM generates 3 possible first steps: "power rule", "first principles", "sum rule"
- Create child nodes for each option
- **Selection**: All untried, so explore each once

**Simulation 4-6: Exploitation**

- For each node, calculate **UCT score**:
  ```
  UCT = (Total Reward / Visits) + 1.414 × √(ln(Parent Visits) / Node Visits)
         \_________________/       \___________________________________/
          Exploitation term          Exploration term
  ```
- **"Sum rule" node** has Q=0.91, N=5 → UCT = 0.182 + 0.42 = 0.602
- **"Power rule" node** has Q=0.85, N=3 → UCT = 0.283 + 0.56 = 0.843 ← **Selected**
- LLM expands from "power rule" node: "d/dx(x²) = 2x, d/dx(3x) = 3"

**Simulation 7-10: Deep Exploitation**

- **"Sum rule" → individual derivatives → combine** path accumulates highest reward (0.91)
- This path gets selected more often (N=5 visits)
- Final reasoning: "Split into x² and 3x → derivatives are 2x and 3 → sum is 2x + 3"

**Scoring (Domain-Specific Reward Model):**
Each path is scored by the **MathWorker's reward function**:

- Contains mathematical notation: +0.30
- Shows step-by-step work: +0.25
- Valid symbolic form: +0.20 (checked with SymPy)
- Reaches conclusion: +0.16
- **Total reward**: 0.91

**Why this matters:**
MCTS balances **exploration** (trying new approaches) vs **exploitation** (following good paths). The UCT formula automatically handles this:

- High Q/N (exploitation): "This path worked well before"
- High exploration term: "We haven't tried this much yet"

This is why AlphaGo beat world champions - MCTS finds non-obvious strategies by systematically exploring possibilities. For reasoning, it means considering multiple solution approaches before committing to one.

#### **Step 5: Extract Best Path**

```
LATS tree → Traverse from root to leaf → Extract reasoning steps
Result: ["Apply sum rule", "d/dx(x²) = 2x", "d/dx(3x) = 3", "Combine: 2x + 3"]
```

**What happens:**

- After 10 simulations, select path with highest cumulative reward
- Extract the **sequence of reasoning steps** from root to leaf
- This becomes the candidate solution

#### **Step 6: Verification**

```
Candidate: "2x + 3"
SymPy verification: derivative(x**2 + 3*x, x) == 2*x + 3 →  TRUE
```

**What happens:**
The **MathWorker** uses **SymPy** (symbolic math engine) for verification:

```python
import sympy as sp
x = sp.Symbol('x')
expected = sp.diff(x**2 + 3*x, x)  # SymPy calculates: 2x + 3
candidate = sp.sympify("2*x + 3")   # Parse candidate
assert sp.simplify(expected - candidate) == 0  # Algebraically equivalent
```

**For other domains:**

- **Code**: AST parsing (check syntax validity)
- **Logic**: Semantic similarity + conclusion detection
- **Factual**: Information completeness + specificity scoring
- **Creative**: Vocabulary diversity + coherence metrics

**Why this matters:**
This isn't just checking if the answer "looks right" - it's **formal verification**. SymPy uses computer algebra to prove algebraic equivalence. "2x + 3" and "3 + 2x" and "2(x + 1.5)" all verify as correct because they're symbolically equivalent. This catches subtle errors that string matching would miss.

#### **Step 7: Success Path - Cache & Return**

```
Verification passed
→ Store tree in cache with embedding
→ Update router training data: {"query": "...", "worker": "math", "success": true}
→ Return result
```

**What happens:**

- Successful tree stored in **semantic cache** with query embedding
- Router records: "Math worker succeeded on this query type"
- Threshold calibrators record: "Worker selection confidence 0.92 was correct"
- Return answer: "The derivative is **2x + 3**"

**Router learning:**
After 32 successful outcomes, router runs gradient descent:

```python
loss = CrossEntropyLoss(predicted_worker, actual_best_worker)
optimizer.backward(loss)
optimizer.step()  # Update neural network weights
```

#### **Alternative: Verification Failure → Reflection**

```
 Verification failed
→ Reflection Engine analyzes error
→ Generate improved reasoning
→ Retry (up to max_iterations)
```

**What happens if verification fails:**

**Example:** Candidate answer was "2x + x" (wrong)

1. **Error Analysis:**

   ```
   Error: Algebraic simplification incorrect
   Issue: Added x instead of constant 3
   ```
2. **Reflection Prompt:**

   ```
   The previous attempt had an error in algebraic simplification.
   Key mistake: confused the derivative of 3x with additional x term.
   Correct approach: d/dx(3x) = 3 (constant factor rule)

   Please provide corrected reasoning...
   ```
3. **Retry:**

   - LLM generates improved reasoning with reflection context
   - New LATS search with reflection guidance
   - Verify again
   - If still fails, repeat (up to `max_reflection_iterations`, default 2)

**Why this matters:**
This is **self-correction** through reflection. The system doesn't just fail - it analyzes **why** it failed and tries again with that knowledge. Research shows LLMs significantly improve when given feedback about their mistakes (Reflexion, Self-Refine papers). Kaelum automates this process.

### Key Concepts Summary

| Concept                         | What It Does                                      | Why It Matters                                                     |
| ------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------ |
| **Embeddings**            | Convert text to vectors that capture meaning      | Enables semantic similarity, not just keyword matching             |
| **Neural Router**         | Learned model that selects expert worker          | Improves over time via gradient descent on outcomes                |
| **MCTS (UCT)**            | Explores multiple solution paths before deciding  | Finds non-obvious solutions by balancing exploration/exploitation  |
| **Domain Scoring**        | Rewards reasoning quality (not just final answer) | Prefers paths with clear logic, even if answer is partial          |
| **Symbolic Verification** | Formal proof of correctness (e.g., SymPy)         | Catches subtle errors that string matching misses                  |
| **Semantic Cache**        | Stores solutions with meaning-based lookup        | 1000x speedup on similar queries with natural language flexibility |
| **Reflection**            | Self-correction by analyzing failures             | Learns from mistakes like humans do                                |
| **Continual Learning**    | Router + thresholds improve with each query       | System gets smarter over time without manual retraining            |

### Performance Profile

```
Cache Hit:     0.001s  (semantic lookup)
New Query:     2-5s    (LATS + verification)
With Retry:    4-12s   (reflection + re-search)
```

The workflow is designed for **quality over speed** on first attempt, but **speed over recomputation** on similar queries. This makes Kaelum ideal for:

- Interactive problem-solving (tutoring, coding assistants)
- Repeated similar queries (documentation Q&A, support bots)
- Tasks requiring verified correctness (math, code generation)

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

## Testing & Development

```bash
pip install pytest pytest-cov
python -m pytest -v
python -m pytest --cov=core --cov=runtime
```

## Performance & Limits

- Default LATS simulations: 10 (router can increase for complex queries)
- Typical query latency: 2–5s (uncached); cached queries ~0.001s (1000x faster)
- Verification: High accuracy for math (SymPy symbolic validation) and Python AST parsing
- Language support: Python, JavaScript, TypeScript for code verification

## Research & References

Kaelum builds upon several key research areas in AI and reasoning:

- [Browne et al. (2012): &#34;A Survey of Monte Carlo Tree Search Methods&#34;](https://ieeexplore.ieee.org/document/6145622)
- [Silver et al. (2016): &#34;Mastering the game of Go with deep neural networks and tree search&#34; (AlphaGo)](https://www.nature.com/articles/nature16961)
- [Wei et al. (2022): &#34;Chain-of-Thought Prompting Elicits Reasoning in Large Language Models&#34;](https://arxiv.org/abs/2201.11903)
- [Yao et al. (2023): &#34;Tree of Thoughts: Deliberate Problem Solving with Large Language Models&#34;](https://arxiv.org/abs/2305.10601)
- [Shinn et al. (2023): &#34;Reflexion: Language Agents with Verbal Reinforcement Learning&#34;](https://arxiv.org/abs/2303.11366)
- [Madaan et al. (2023): &#34;Self-Refine: Iterative Refinement with Self-Feedback&#34;](https://arxiv.org/abs/2303.17651)
- [Shazeer et al. (2017): &#34;Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer&#34;](https://arxiv.org/abs/1701.06538)
- [Fedus et al. (2021): &#34;Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity&#34;](https://arxiv.org/abs/2101.03961)
- [Welleck et al. (2022): &#34;Symbolic Knowledge Distillation: from General Language Models to Commonsense Models&#34;](https://arxiv.org/abs/2110.07178)
- [Settles (2009): &#34;Active Learning Literature Survey&#34;](https://minds.wisconsin.edu/handle/1793/60660)
- [Reimers &amp; Gurevych (2019): &#34;Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks&#34;](https://arxiv.org/abs/1908.10084)
