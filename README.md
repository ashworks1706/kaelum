# Kaelum

<img width="1983" height="1098" alt="image" src="https://github.com/user-attachments/assets/97f5601e-e660-44b1-9338-80308e0d80d4" />
<img width="1983" height="915" alt="image" src="https://github.com/user-attachments/assets/1d810ebb-496f-494b-9f4a-cb3022dd22fe" />
<img width="1983" height="844" alt="image" src="https://github.com/user-attachments/assets/6b000d29-d8bc-4219-8157-de5bf966f229" />

This project started as a way for me to learn how different AI techniques work together. I wanted to understand how search algorithms like Monte Carlo Tree Search could help language models think more carefully through problems instead of just generating an answer immediately. The idea is to explore multiple reasoning paths before committing to a solution, kind of like how you might sketch out different approaches to a math problem before deciding which one works best.

The system routes queries to specialized workers depending on the question type. There's a math worker that uses symbolic verification with SymPy, a code worker that parses syntax trees, and workers for logic, factual questions, creative tasks, and analysis. Each one has different reward functions tuned for their domain. I added a semantic cache so identical questions don't need to be recomputed, and a neural router that learns which worker to use based on past performance.

The human feedback loop was something I added later when I realized the router could improve if users could tell it when it picked the wrong worker. Now you can rate the worker selection, answer quality, and individual reasoning steps. Those adjustments persist across sessions and actually influence future routing decisions and reward calculations.

---

### How a Query Flows Through the System

Here's what happens when you ask a question. I'll use a calculus problem as an example to make it concrete.

```
Query Input
     ↓
[1] Query Embedding
     - Convert to 384-dimensional vector using sentence-transformers
     ↓
[2] Completeness Detection
     - Quick check if the query is actually answerable
     ↓
[3] Cache Lookup (happens first, before routing)
     - Check cosine similarity with cached queries (threshold: 0.85)
     - Only return high-quality cached results
     - Ask LLM if cached answer actually satisfies new query
     - Cache hit? Return immediately (0.001s)
     ↓
[4] Query Classification (only on cache miss)
     - Determine task type, worker type, domain
     ↓
[5] Neural Router
     - Extract features from query (length, math symbols, code keywords, etc.)
     - Feed through neural network: 398 → 256 → 128 → outputs
     - Apply human feedback adjustments to worker probabilities
     - Select worker + tree depth + simulation count
     ↓
[6] LATS Search
     - Run N simulations (default 10)
     - Selection: UCT formula balances exploration vs exploitation
     - Expansion: LLM generates next reasoning steps
     - Simulation: Score with domain-specific rewards
     - Backpropagate: Update ancestor nodes, prune bad branches (visits ≥3, reward <0.3)
     - Extract best path after all simulations
     ↓
[7] Verification
     - Math: SymPy symbolic checks
     - Code: AST parsing and syntax validation
     - Logic/Factual: Semantic coherence
     - Creative: Vocabulary diversity
     - Pass? Go to [9], Fail? Go to [8]
     ↓
[8] Reflection (on failure)
     - Analyze what went wrong
     - Generate reflection prompt with guidance
     - Retry with new LATS search (up to 2 iterations)
     ↓
[9] Success
     - Cache the result with quality="high"
     - Save routing outcome to persistent storage
     - Update router training data
     - Return answer
     ↓
[10] Human Feedback (optional)
     - User rates worker correctness, answer quality, reasoning steps
     - Adjustments stored in .kaelum/cache/feedback/
     - Active for all future queries
```

The cache-first design is important because it avoids unnecessary work. About 23% of queries hit the cache, which means they return instantly instead of spending 2-5 seconds on tree search. The quality filter ensures we only serve verified solutions.

---

### Semantic Cache with Two-Stage Validation

The cache converts your query into a 384-dimensional embedding and checks cosine similarity against all cached queries. If the similarity is above 0.85, it doesn't just return the cached answer immediately. Instead, it asks the LLM: "Would this cached answer fully and correctly satisfy the new query?"

This prevents a problem I ran into early on. Queries like "integral of x²" and "integral of x² from 0 to 1" have 0.89 similarity, but they need different answers (one is indefinite, one is definite). The LLM validation catches these nuances that pure embedding similarity misses.

Only high-quality results get cached. If verification failed or confidence was low, the tree gets logged but never served on future lookups. Every validation decision goes into `.kaelum/cache_validation/validation_log.jsonl`, so I can export the data later and fine-tune the validator model to get better over time.

---

### Human Feedback System

I added this after realizing the router makes mistakes sometimes. If it sends a calculus question to the logic worker instead of the math worker, there should be a way to correct it. Now after each query, you can tell the system whether it picked the right worker, rate the answer quality, and rate individual reasoning steps.

The feedback directly adjusts probabilities for next time:

```python
# If user says math worker was wrong, penalize it
worker_adjustment["math"] -= 0.03

# If user suggests code worker instead, boost it
worker_adjustment["code"] += 0.05

# Next similar query, router applies these adjustments
router_probs = softmax(neural_net_output + feedback_adjustments * 0.3)
```

Answer quality ratings adjust the reward calculations. If you rate an answer 4-5 stars, future similar reasoning paths get a small reward boost. Steps rated highly get their reward multipliers increased (up to 1.3x), and poorly rated steps get reduced (down to 0.8x).

Everything persists to `.kaelum/cache/feedback/feedback.jsonl`, so the adjustments are active even after restart. The system tracks accuracy per worker, total corrections, and reward adjustments. You can check current adjustments via the `/api/feedback/router-impact` endpoint.

---

### Neural Router

The router learns which worker should handle each query type. It extracts a 398-dimensional feature vector: 384-dim embedding from sentence-transformers plus 14 structural features like query length, count of math symbols (∂, ∫, √), code keywords (def, class, function), question words, and special tokens.

The neural network has two hidden layers (398 → 256 → 128) with ReLU activation and 30% dropout. It outputs three things: worker probabilities (6 workers), tree depth (scaled to 3-10), and simulation count (scaled to 5-25).

Every query outcome gets saved with the worker used, success/failure, average tree reward, and actual depth/simulations. After collecting 32 outcomes, the router trains using gradient descent. The loss function combines worker classification (CrossEntropyLoss) with quality regression (MSE on average rewards) and parameter prediction (MSE on depth/simulations).

This means it learns patterns like "calculus queries → math worker with depth=6, sims=15" or "algorithm questions → code worker with depth=8, sims=20". The average reward feedback is particularly useful because it teaches the router to prefer workers that generate high-quality reasoning trees, not just workers that happen to succeed.

Implementation is in `core/search/router.py`.

---

### The Six Workers

Each worker has domain-specific prompting, scoring, and verification:

**Math Worker**: Uses SymPy for symbolic verification. Can check if "2x+3", "3+2x", and "2(x+1.5)" are all mathematically equivalent. Rewards: +0.30 for notation, +0.25 for step-by-step work, +0.20 for symbolic validity, +0.16 for reaching a conclusion. Best for calculus, algebra, equations, proofs.

**Logic Worker**: Checks semantic coherence and validates premise-conclusion structure. Rewards: +0.30 for logical structure, +0.25 for coherence, +0.20 for premises, +0.16 for conclusion. Best for logical reasoning, arguments, deduction.

**Code Worker**: Parses AST for Python/JavaScript/TypeScript, validates syntax, can sandbox execution. Rewards: +0.30 for syntax correctness, +0.25 for documentation, +0.20 for modularity, +0.16 for correctness. Best for programming, algorithms, debugging.

**Factual Worker**: Scores information completeness and uses joint embedding validation. Rewards cite specific evidence and comprehensive coverage. Best for knowledge queries, explanations, definitions.

**Creative Worker**: Uses vocabulary diversity metrics (unique words / total words) and coherence detection. Balances originality with structure. Best for writing, brainstorming, creative tasks.

**Analysis Worker**: Depth scoring, keyword presence, multi-perspective evaluation. Rewards comprehensive multi-angle analysis. Best for complex reasoning, trade-offs, evaluations.

---

### LATS - Language Agent Tree Search

This is the core search algorithm. It runs Monte Carlo Tree Search where each node is a reasoning step generated by the LLM.

During each simulation:

1. **Selection**: Walk down the tree using UCT (Upper Confidence Trees). The formula is `Q/N + c×√(ln N_parent / N_node)`. The first term (Q/N) is exploitation - prefer paths that worked well before. The second term is exploration - try paths we haven't visited much. Pruned nodes get skipped.

2. **Expansion**: When we reach a leaf, the LLM generates possible next reasoning steps. These become new child nodes.

3. **Simulation**: Score the reasoning path using the worker's domain-specific reward function. Math worker checks for mathematical notation, step-by-step work, symbolic validity. Code worker checks syntax, documentation, modularity.

4. **Backpropagation**: Update all ancestor nodes with the rewards. Check if any node should be pruned (visits ≥ 3 AND avg_reward < 0.3). This eliminates bad branches early so we don't waste simulations exploring them.

After all simulations finish, extract the path with the highest cumulative reward. This becomes the candidate solution.

The pruning was a key addition. Early versions would waste simulations exploring obviously wrong approaches. Now if a "use first principles" branch has been tried 3 times and averages 0.28 reward, it gets marked pruned and skipped in future selections. This is similar to alpha-beta pruning in chess engines but adapted for MCTS.

---

### Verification

Each worker has different verification methods:

**Math**: SymPy symbolic engine. Parses both the expected answer and candidate answer into symbolic expressions, then checks if `simplify(expected - candidate) == 0`. This catches subtle equivalences that string matching would miss.

**Code**: AST parsing. Builds an abstract syntax tree and checks for syntax errors. Can also sandbox execute the code and check if it runs without exceptions.

**Logic/Factual**: Embedding-based semantic verification. Encodes the answer with sentence-transformers and measures coherence. Checks if a conclusion is present, if information is complete and specific.

**Creative**: Vocabulary diversity (unique words / total words) and sentence coherence. Needs a balance - too repetitive is bad, but incoherent is also bad.

If verification fails, the system doesn't just return an error. It goes to the reflection engine.

---

### Reflection Engine

When verification fails, the system analyzes what went wrong and tries again with guidance.

For a math error, it might identify "Algebraic simplification error in step 3". For code, it might say "Syntax error on line 12: missing closing parenthesis". For logic, it might notice "Conclusion doesn't follow from premises".

It generates a reflection prompt:

```
Previous attempt failed verification.
Error: [specific issue identified]
Key mistake: [detailed explanation]
Correct approach: [guidance for improvement]

Please provide corrected reasoning...
```

Then it runs a new LATS search with this reflection context. The LLM can see what went wrong and adjust. Default is 2 iterations, but it stops early if verification passes.

This is based on the Reflexion paper - LLMs improve significantly when given feedback about their mistakes. Kaelum automates this self-correction loop.

---

### Adaptive Threshold Calibration

This solves a problem with binary decisions. The model outputs a confidence score between 0 and 1, but we need to decide yes/no. What threshold should we use? 0.5 is arbitrary.

The system records every decision: (score, threshold, outcome). After collecting 20+ samples, it runs a grid search over thresholds [0.20, 0.25, ..., 0.85] and calculates F1 score for each: `2 * (precision × recall) / (precision + recall)`. It selects the threshold that maximizes F1.

These optimal thresholds persist to `.kaelum/calibration/optimal_thresholds.json` and get used for future decisions. The system automatically recalibrates as more data comes in.

---

### Active Learning

The system can intelligently select which queries to export as training data. There are several strategies:

**Uncertainty**: Queries where the model had low confidence. Training on these improves weak areas.

**Diversity**: Max-min distance in embedding space. This ensures broad coverage of the query distribution.

**Error**: Failed verifications with reflections. Learning from mistakes is valuable.

**Complexity**: High depth, many simulations. Training on hard problems improves capability.

**Mixed** (recommended): Balanced combination of all strategies.

Data gets formatted as `{instruction, input, output}` for instruction-tuning. The export is in `runtime/orchestrator.py`. This creates a continual learning loop where the system generates its own training data from real usage.

---

### Metrics and Analytics

The system tracks everything:

- **Cache**: Hit rate, similarity distribution, quality distribution, validation stats
- **Router**: Worker selection accuracy, prediction errors, learning curves
- **LATS**: Average tree depth, simulation count, pruning efficiency, branch rewards
- **Verification**: Pass/fail rates by domain, error types, reflection success rates
- **Tokens**: Input/output per worker, cost estimation
- **Latency**: Time spent in cache/routing/search/verification/reflection

Metrics export to JSON/CSV formats. There's a web dashboard for real-time monitoring.

---

## Example Walkthrough: Derivative of x² + 3x

Let me walk through exactly what happens with a concrete example.

**Query**: "What is the derivative of x² + 3x with respect to x?"

**Step 1: Query Embedding & Cache Lookup**

The query gets converted to a 384-dimensional embedding vector using sentence-transformers. The cache immediately checks cosine similarity with all cached queries. If there's a match above 0.85 threshold AND quality="high", it asks the LLM validator: "Would the cached answer fully satisfy this query?" If yes, return cached result in 0.001s and skip everything else.

Let's assume cache miss, so we continue.

**Step 2: Feature Extraction & Routing**

The router extracts structural features:
- Query length: 53 characters (normalized)
- Math symbols: "²" detected (count: 1)
- Code keywords: none
- Question words: "what" (present)
- Special tokens: count parentheses, operators

Concatenate the 384-dim embedding with 14 structural features to get 398-dim feature vector. Feed through neural network:

```
398 → Linear + ReLU + Dropout(0.3) → 256
256 → Linear + ReLU + Dropout(0.3) → 128
128 → Output heads
```

Outputs:
- Worker probabilities: [math: 0.92, logic: 0.04, code: 0.02, factual: 0.01, creative: 0.005, analysis: 0.005]
- Tree depth: 5
- Simulations: 10

Selects math worker with 92% confidence.

**Step 3: LATS - Monte Carlo Tree Search**

Start with 10 simulations.

**Simulation 1-3**: Initial exploration from root node. LLM generates three possible approaches:
- Node 1: "Apply power rule to x²"
- Node 2: "Use first principles definition of derivative"
- Node 3: "Apply sum rule first to split the terms"

Each node gets visited once.

**Simulation 4-6**: UCT selection kicks in.

For "power rule" node: Q=0.85, N=3
- UCT = 0.85/3 + 1.414×√(ln(6)/3) = 0.283 + 0.56 = 0.843

For "sum rule" node: Q=0.91, N=5
- UCT = 0.91/5 + 1.414×√(ln(6)/5) = 0.182 + 0.42 = 0.602

"Power rule" node has higher UCT, so it gets selected. LLM expands from this node: "d/dx(x²) = 2x, d/dx(3x) = 3".

**Simulation 7-9**: "Sum rule → split terms → individual derivatives → combine" path accumulates highest reward (0.91). Gets visited more often (N=5).

Meanwhile, "first principles" node has accumulated 3 visits with avg_reward = 0.28. Since visits ≥ 3 AND avg_reward < 0.3, it gets marked as pruned. No future simulations will explore this branch.

**Simulation 10**: Final simulation goes down the "sum rule" path again, reinforcing it as the best approach.

**Scoring**: Each path gets scored by the MathWorker reward function:
- Contains mathematical notation: +0.30 ✓
- Shows step-by-step work: +0.25 ✓
- Valid symbolic form: +0.20 ✓
- Reaches conclusion: +0.16 ✓
- **Total: 0.91**

**Best Path Extraction**: Traverse from root to leaf following highest rewards. Extract reasoning steps:
1. "Apply sum rule to split into x² and 3x"
2. "Take derivative of x²: d/dx(x²) = 2x"
3. "Take derivative of 3x: d/dx(3x) = 3"
4. "Combine results: 2x + 3"

**Step 4: Verification**

MathWorker uses SymPy:

```python
import sympy as sp
x = sp.Symbol('x')
expected = sp.diff(x**2 + 3*x, x)  # SymPy calculates: 2x + 3
candidate = sp.sympify("2*x + 3")   # Parse candidate
assert sp.simplify(expected - candidate) == 0  # Check equivalence
```

Verification passes. ✓

**Step 5: Success Path**

- Store tree in cache with quality="high", query embedding, node count
- Save routing outcome: {query, worker="math", success=True, avg_reward=0.91, depth=5, sims=10}
- Record threshold calibration: "Worker selection confidence 0.92 was correct"
- Return answer: "The derivative is **2x + 3**"

After 32 routing outcomes accumulate, the router will train:

```python
loss = CrossEntropyLoss(predicted_worker, actual_worker)
reward_loss = MSELoss(predicted_quality, actual_avg_reward)
total_loss = loss + 0.5 * reward_loss
optimizer.backward(total_loss)
optimizer.step()
```

The reward loss is important - it teaches the router that math worker not only got the answer right, but generated a high-quality reasoning tree (0.91 reward). For future calculus queries, the router will be even more confident about selecting math worker.

**Alternative: Verification Failure**

If the candidate answer was wrong (say "2x + x"), verification would fail. The reflection engine would analyze:

```
Error: Algebraic simplification incorrect
Issue: Confused derivative of 3x with additional x term
```

Generate reflection prompt:

```
Previous attempt had an error in algebraic simplification.
Key mistake: Derivative of 3x should be 3 (constant factor rule), not x.
Correct approach: d/dx(c·f(x)) = c·f'(x), so d/dx(3x) = 3·1 = 3

Please provide corrected reasoning...
```

Run new LATS search with this reflection context. The LLM sees the mistake and adjusts. Try again (up to 2 iterations). If still failing after max iterations, return best attempt with a note about verification failure.

The failed tree would be stored with quality="low" so it never gets served from cache later.

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/ashworks1706/KaelumAI.git
cd KaelumAI

# Backend dependencies
pip install -r requirements.txt

# Frontend dependencies  
cd frontend
npm install
cd ..
```

### 2. Configuration (Optional)

Copy the example environment file and customize:

```bash
cp .env.example .env
# Edit .env to change ports, URLs, or API keys
```

Default configuration:
- Backend: `http://localhost:5000`
- LLM endpoint: `http://localhost:8000/v1`
- Frontend: `http://localhost:3000`

You can override these with environment variables:
```bash
BACKEND_PORT=8080 python backend/app.py
# or edit .env file
```

### 3. Start vLLM Backend

I recommend vLLM because it's way faster than standard transformers inference. Here are some models I've tested:

```bash
# Install vLLM
pip install vllm

# Small and fast (good for testing, 3GB VRAM)
python -m vllm.entrypoints.openai.api_server \
    --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --port 8000

# Balanced (recommended for actual use, 8GB VRAM)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.7

# High quality reasoning (if you have 16GB VRAM)
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/phi-4 \
    --port 8000 \
    --gpu-memory-utilization 0.9
```

### 3. Start Kaelum

**Automatic:**
```bash
./start_demo.sh
```

**Manual:**
```bash
# Terminal 1 - Backend (port 5000)
cd backend
python app.py

# Terminal 2 - Frontend (port 3000)
cd frontend
npm run dev
```

Open http://localhost:3000 in your browser.

---

## Model Recommendations

Kaelum works with any OpenAI-compatible API. Here are models I've tested:

| Model | Size | VRAM | Speed | Reasoning | Notes |
|-------|-----:|-----:|-------|-----------|-------|
| SmolLM2 | 1.7B | 3GB | Very fast | Decent | Good for edge/mobile, trained on 11T tokens |
| Qwen 2.5 | 3B | 4GB | Very fast | Good | Solid for development |
| Phi-3-mini | 3.8B | 5GB | Very fast | Excellent | Best small reasoning model (GSM8K: 85.7%) |
| Qwen 2.5 | 7B | 8GB | Fast | Excellent | My go-to for balanced performance |
| Llama 3.1 | 8B | 8GB | Fast | Good | General purpose |
| DeepSeek-R1 | 7B | 8GB | Fast | Excellent | Specialized for math/logic |
| Phi-4 | 14B | 16GB | Fast | Excellent | SOTA small model (MMLU: 84.8%) |
| Qwen 2.5 | 14B | 16GB | Fast | Excellent | Production quality |
| Mixtral | 8×7B | 24GB | Fast | Good | MoE architecture |

The Phi models from Microsoft are particularly good at reasoning tasks. DeepSeek-R1 is great for math. Qwen 2.5 is my general recommendation for the best balance.

**Using OpenAI instead:**

```bash
export OPENAI_API_KEY="sk-..."
python run.py --model gpt-4 --base-url https://api.openai.com/v1
```

---

## Configuration

You can configure everything through the web interface or the Flask API at `/api/config`.

Default settings:

```json
{
  "base_url": "http://localhost:8000/v1",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "temperature": 0.7,
  "max_tokens": 512,
  "embedding_model": "all-MiniLM-L6-v2",
  "enable_routing": true,
  "use_symbolic_verification": true,
  "use_factual_verification": false,
  "max_reflection_iterations": 2,
  "parallel": false,
  "max_workers": 4,
  "router_learning_rate": 0.001,
  "router_buffer_size": 32,
  "router_exploration_rate": 0.1,
  "cache_dir": ".kaelum/cache",
  "router_data_dir": ".kaelum/routing",
  "enable_active_learning": true
}
```

---

## What I Learned

This project taught me a lot about how different AI techniques fit together:

**Monte Carlo Tree Search** is really powerful for reasoning tasks. Instead of generating an answer immediately, you explore multiple solution paths and pick the best one. The UCT formula automatically balances trying new approaches vs following paths that worked before. Early pruning prevents wasting time on obviously wrong branches.

**Domain-specific verification** is essential. You can't just trust that the LLM got it right. For math, symbolic verification with SymPy catches errors that would look correct to a human reading the text. For code, AST parsing finds syntax issues. The verification step provides a training signal - you know definitively whether the reasoning was correct.

**Reflection improves results significantly**. When verification fails, analyzing the error and retrying with guidance often fixes it. This is way better than just returning "verification failed". The Reflexion paper showed LLMs can self-correct when given feedback, and it's true in practice.

**Neural routing beats hand-written rules**. I initially tried routing based on keyword matching (if query contains "∂" or "∫", use math worker). The learned router performs way better because it picks up on subtle patterns in the embeddings. Training on actual outcomes with gradient descent means it continuously improves.

**Human feedback is incredibly valuable**. Sometimes the router picks the wrong worker, or the reward function scores a bad reasoning path highly. Letting users provide corrections creates a feedback loop that fixes systematic errors. The adjustments persist across sessions, so the system actually learns from mistakes.

**Quality-aware caching prevents serving bad answers**. Early versions cached everything and sometimes returned incorrect solutions for similar queries. Filtering to only cache high-quality verified results fixed this. The LLM validation layer catches cases where embedding similarity is high but the queries need different answers.

**Active learning helps with data efficiency**. Instead of treating all queries equally for training data, selecting based on uncertainty, diversity, and errors means you get more value from fewer examples. This is especially important when fine-tuning is expensive.

The hardest parts were getting the MCTS pruning right (too aggressive and you miss good paths, too lenient and you waste simulations) and tuning the domain-specific reward functions (they need to actually correlate with answer quality for the search to work).

---

## Papers I Found Helpful

These papers influenced the design:

- **Monte Carlo Tree Search**: Browne et al. (2012), "A Survey of Monte Carlo Tree Search Methods". Good overview of MCTS algorithms and UCT formula.

- **AlphaGo**: Silver et al. (2016), "Mastering the game of Go with deep neural networks and tree search". Showed how MCTS + neural networks can solve complex problems.

- **Chain-of-Thought**: Wei et al. (2022), "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models". Step-by-step reasoning improves LLM performance.

- **Tree of Thoughts**: Yao et al. (2023), "Tree of Thoughts: Deliberate Problem Solving with Large Language Models". Explores multiple reasoning paths before deciding.

- **Reflexion**: Shinn et al. (2023), "Reflexion: Language Agents with Verbal Reinforcement Learning". Self-reflection and error correction for LLMs.

- **Self-Refine**: Madaan et al. (2023), "Self-Refine: Iterative Refinement with Self-Feedback". Similar idea of iterative improvement.

- **Mixture of Experts**: Shazeer et al. (2017), "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer". Routing to specialized experts.

- **Switch Transformers**: Fedus et al. (2021), "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity". More on MoE architectures.

- **Symbolic Knowledge Distillation**: Welleck et al. (2022), "Symbolic Knowledge Distillation: from General Language Models to Commonsense Models". Using symbolic verification to guide LLM training.

- **Active Learning**: Settles (2009), "Active Learning Literature Survey". Selecting valuable training examples.

- **Sentence-BERT**: Reimers & Gurevych (2019), "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks". Semantic similarity with embeddings.

Reading these papers helped me understand the techniques, but implementing them taught me where the tricky parts are. A lot of the details that matter for making things work aren't in the papers - they come from experimentation and debugging.

---

## Project Structure

```
Kaelum/
├── backend/           # Flask API server
├── frontend/          # Next.js web interface
├── core/
│   ├── detectors/     # Query classification and detection
│   ├── learning/      # Human feedback, active learning, metrics
│   ├── search/        # LATS, router, tree cache, reward models
│   ├── verification/  # SymPy, syntax validation, calibrators
│   └── workers/       # Specialized workers (Math, Code, Logic, etc.)
├── runtime/           # Orchestrator that ties everything together
├── docs/              # Technical documentation
└── .kaelum/          # Runtime data (cache, routing, feedback)
```

---

## Code Quality Notes

This codebase has been cleaned up for production quality:
- ✅ Proper error handling with logging throughout
- ✅ No silent failures (all exceptions logged)
- ✅ Configurable via environment variables (`.env.example`)
- ✅ Centralized API configuration
- ✅ Magic numbers documented with rationale
- ✅ No dead code or unused imports

Some components are intentionally complex for educational exploration:
- Large ML models (BART) demonstrate NLI and zero-shot classification
- Task classifier with exemplars shows semantic similarity approaches
- Multiple calibrators explore adaptive thresholding techniques
- Human feedback system implements learning from user corrections

See `docs/HARDCODED_ISSUES.md` for detailed technical analysis of design decisions and trade-offs.
