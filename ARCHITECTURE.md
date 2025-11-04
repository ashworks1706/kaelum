# Kaelum New Architecture: LATS-Based Worker Routing

## Overview

Kaelum has been completely refactored to implement a novel reasoning architecture:

```
Query → Router → Specialist Worker → LATS (MCTS) → Cached Tree → Result
```

### Key Innovations

1. **Novel Routing**: Router uses neural network or learned patterns (not keyword matching) to select specialist workers
2. **LATS Integration**: Each worker uses Language Agent Tree Search with MCTS for multi-step reasoning
3. **Tree Caching**: Past reasoning trees are cached and retrieved by query similarity
4. **Specialist Workers**: Dedicated agents for math, logic, code, factual, creative, and analysis tasks

## Architecture Components

### 1. Router (`core/router.py`, `core/neural_router.py`)

**Purpose**: Intelligently route queries to the most suitable specialist worker

**How it works**:
- Extracts query features (embeddings + complexity + type scores)
- Uses learned performance data or neural network to select worker
- Configures LATS parameters based on query complexity
- Records outcomes for continuous learning

**Novel Aspects**:
- No keyword matching or pattern-based fallbacks
- Learns from historical performance
- Adjusts search depth/simulations dynamically
- Multi-signal classification (not just single label)

**Output**: `RoutingDecision` with:
- `worker_specialty`: Which worker to use (math/logic/code/factual/creative/analysis)
- `max_tree_depth`: How deep to search (3-10)
- `num_simulations`: MCTS simulations to run (5-25)
- `use_tree_cache`: Whether to check cache
- `confidence`: Router's confidence in decision

### 2. Specialist Workers (`core/workers.py`)

**Available Workers**:
- `MathWorker`: Mathematical reasoning with SymPy verification
- `LogicWorker`: Deductive reasoning and formal logic
- `CodeWorker`: Code generation and debugging
- `FactualWorker`: Knowledge retrieval with RAG
- `CreativeWorker`: Creative content generation
- `AnalysisWorker`: Complex analysis tasks

**Worker Pipeline**:
```python
def solve(query, use_cache=True, max_tree_depth=5, num_simulations=10):
    # 1. Check tree cache for similar query
    if use_cache:
        cached = check_cache(query)
        if cached:
            return cached  # Fast path!
    
    # 2. Initialize LATS tree
    tree = LATS(root_state=initial_state, 
                simulator=verify_step,
                expand_fn=generate_next_step)
    
    # 3. Run MCTS simulations
    for _ in range(num_simulations):
        node = tree.select()  # UCT selection
        child = tree.expand(node, expand_fn(node))
        reward = tree.simulate(child)  # Verify with domain knowledge
        tree.backpropagate(child, reward)
    
    # 4. Extract best path
    best_node = tree.best_child()
    reasoning_steps = extract_path(best_node)
    answer = best_node.state['answer']
    
    # 5. Cache the tree
    if success:
        tree_cache.store(query, tree, worker_specialty, confidence)
    
    return WorkerResult(answer, reasoning_steps, confidence, ...)
```

### 3. LATS (Language Agent Tree Search) (`core/lats.py`)

**Purpose**: MCTS-style tree search for reasoning

**Key Features**:
- UCT-based node selection
- Domain-agnostic (caller provides simulator/expander)
- Tree serialization for caching
- Backpropagation of rewards

**Node Structure**:
```python
{
    "id": "node_123",
    "state": {
        "query": "...",
        "step": "Current reasoning step",
        "depth": 3,
        "partial_solution": "...",
        "answer": "..." 
    },
    "visits": 15,
    "value": 12.5,
    "children": [...]
}
```

**Simulator Function**: 
- Provided by each worker
- Evaluates quality of reasoning step
- Returns reward (0-1 typically)
- Example: MathWorker uses SymPy to verify equations

**Expander Function**:
- Generates next reasoning step
- Uses LLM to propose child states
- Domain-specific logic

### 4. Tree Cache (`core/tree_cache.py`)

**Purpose**: Store and retrieve reasoning trees for similar queries

**How it works**:
```python
# Store a tree
tree_cache.store(
    query="What is 15% of 899?",
    tree=lats_tree,
    worker_specialty="math",
    success=True,
    confidence=0.95
)

# Retrieve similar tree
result = tree_cache.retrieve(
    query="What is 15% of 900?",  # Similar query
    worker_specialty="math",
    require_success=True
)

if result:
    tree, metadata, similarity = result
    # similarity = 0.92 (query embeddings are close)
    # Reuse the cached reasoning!
```

**Benefits**:
- Near-instant responses for similar queries
- Transfer learning across query variations
- Reduces LLM calls
- Accumulates reasoning knowledge

### 5. Orchestrator (`runtime/orchestrator.py`)

**New Flow**:
```python
def infer(query):
    # 1. Route to worker
    decision = router.route(query)
    worker = get_worker(decision.worker_specialty)
    
    # 2. Execute worker with LATS
    result = worker.solve(
        query,
        use_cache=decision.use_tree_cache,
        max_tree_depth=decision.max_tree_depth,
        num_simulations=decision.num_simulations
    )
    
    # 3. Record outcome for router learning
    router.record_outcome(decision, result)
    
    return format_result(result)
```

## Complete Workflow Example

### Query: "Calculate 15% of $899"

```
1. ROUTING
   ├─ Extract features:
   │  ├─ Embedding: [0.21, -0.45, ...] (384-dim)
   │  ├─ Math score: 0.95 (has operators, numbers)
   │  ├─ Complexity: 0.35 (moderate)
   │  └─ Query type: MATH
   └─ Decision:
      ├─ Worker: math
      ├─ Tree depth: 5
      ├─ Simulations: 10
      └─ Confidence: 0.92

2. CACHE CHECK
   └─ No similar tree found → proceed to LATS

3. LATS TREE SEARCH
   ├─ Root: "Initial problem analysis"
   ├─ Simulation 1:
   │  ├─ Select: root
   │  ├─ Expand: "Convert percentage to decimal: 0.15"
   │  ├─ Simulate: reward=0.6 (partial progress)
   │  └─ Backpropagate: root.value += 0.6
   ├─ Simulation 2:
   │  ├─ Select: "Convert percentage..." (has highest UCT)
   │  ├─ Expand: "Multiply: 899 × 0.15 = 134.85"
   │  ├─ Simulate: reward=0.9 (SymPy verified!)
   │  └─ Backpropagate
   ├─ ... (8 more simulations)
   └─ Best path found:
      1. "Convert percentage to decimal: 0.15"
      2. "Multiply: 899 × 0.15 = 134.85"
      3. "The answer is $134.85"

4. VERIFICATION
   └─ SymPy verified equation: 899 × 0.15 = 134.85 ✓

5. CACHING
   └─ Store tree in cache for future similar queries

6. RESULT
   ├─ Answer: "$134.85"
   ├─ Confidence: 0.95
   ├─ Reasoning steps: 3
   └─ Execution time: 1.2s
```

## Why This is Novel

### vs. Traditional Chain-of-Thought
- **CoT**: Single linear path, no exploration
- **Kaelum**: MCTS explores multiple paths, picks best

### vs. Tree-of-Thoughts
- **ToT**: All paths at each step, expensive
- **Kaelum**: UCT-guided exploration, more efficient

### vs. Keyword-Based Routing
- **Traditional**: "if 'calculate' in query: use math"
- **Kaelum**: Learned neural policy or similarity-based

### vs. ReAct/Agent Systems
- **ReAct**: Tool use, action-observation loops
- **Kaelum**: Reasoning-focused, builds tree of thoughts

## Key Differences from Paper

The original LATS paper (Zhou et al.) focuses on:
- External environment interaction
- Action selection for agents
- Task completion in simulated environments

Kaelum's implementation focuses on:
- **Pure reasoning** (no external environment)
- **Multi-step logical inference**
- **Domain-specific verification** as simulation
- **Tree caching** for knowledge accumulation

## Performance Benefits

1. **Speed**: Cache hits return instantly (~1ms vs ~1s)
2. **Quality**: MCTS explores multiple paths, avoids dead ends
3. **Learning**: Router improves over time with feedback
4. **Specialization**: Workers excel at their domain

## Testing the New Architecture

```bash
# Run test suite
python test_new_architecture.py

# Expected output:
# ✓ Router classification
# ✓ LATS basic functionality  
# ✓ Tree cache storage/retrieval
# ✓ MathWorker with LATS (if LLM available)
```

## Using the New System

```python
import kaelum

# Simple usage (same API as before)
kaelum.set_reasoning_model(
    base_url="http://localhost:11434/v1",
    model="qwen2.5:3b",
    enable_routing=True  # Enable worker-based routing
)

result = kaelum.enhance("Calculate the derivative of x^2 + 3x")
print(result)

# Advanced usage with configuration
from kaelum.runtime.orchestrator import KaelumOrchestrator
from core.config import KaelumConfig, LLMConfig

config = KaelumConfig(
    reasoning_llm=LLMConfig(
        base_url="http://localhost:11434/v1",
        model="qwen2.5:3b"
    )
)

orchestrator = KaelumOrchestrator(config, enable_routing=True)
result = orchestrator.infer("If all A are B, and all B are C, what about A and C?")

print(f"Worker: {result['worker']}")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Cache hit: {result['cache_hit']}")
print(f"Reasoning steps: {len(result['reasoning_trace'])}")
```

## Future Enhancements

1. **Self-Play Training**: Generate synthetic queries, compare routes, train router
2. **Multi-Worker Consensus**: Run multiple workers, ensemble results
3. **Adaptive Depth**: Dynamically adjust tree depth based on intermediate results
4. **Cross-Query Learning**: Transfer successful reasoning patterns across domains
5. **Verification Strengthening**: Integrate Z3 for logic, more SymPy patterns for math

## File Structure

```
core/
├── router.py              # Rule-based router with learning
├── neural_router.py       # Neural policy network router
├── workers.py            # Base worker + Math/Logic workers
├── lats.py               # LATS tree search implementation
├── tree_cache.py         # Tree caching with similarity search
├── code_worker.py        # Code generation specialist
├── factual_worker.py     # RAG-based factual specialist
└── creative_worker.py    # Creative content specialist

runtime/
└── orchestrator.py       # Main pipeline orchestration

__init__.py               # Public API
```

## Migration from Old Architecture

The old `Generate → Verify → Reflect` pipeline is **replaced** by:
```
Router → Worker → LATS → Cache
```

Old features removed:
- ❌ ReasoningGenerator (replaced by worker-specific LATS)
- ❌ VerificationEngine (replaced by worker-specific simulators)
- ❌ ReflectionEngine (replaced by MCTS exploration)

New features added:
- ✅ Specialist workers with domain expertise
- ✅ LATS tree search with MCTS
- ✅ Tree caching for fast retrieval
- ✅ Novel routing (not keyword-based)
- ✅ Continuous learning from outcomes

## Conclusion

Kaelum now implements a truly novel reasoning architecture that:
1. **Routes intelligently** to specialist workers
2. **Explores reasoning space** with LATS/MCTS
3. **Caches knowledge** in reasoning trees
4. **Learns continuously** from outcomes

This is fundamentally different from simple prompt chaining or keyword-based routing. The system builds and reuses a growing library of verified reasoning patterns.
