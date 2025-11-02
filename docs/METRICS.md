# 📊 Metrics & Cost Tracking - Real-Time Performance Monitoring

The CostTracker provides real-time visibility into token usage, latency, and cost savings when using Kaelum vs commercial LLMs.

## Overview

Every inference through Kaelum is tracked with:
- **Tokens**: Approximate token count per stage
- **Latency**: Milliseconds per operation
- **Cost**: Estimated cost vs commercial LLMs
- **Savings**: How much you saved using local models

## Architecture

```
Inference Start
      ↓
┌─────────────────┐
│  Start Session  │  session_id = "stream_1730567890"
└────────┬────────┘
         │
    ┌────┴─────┬──────────┬──────────┐
    ▼          ▼          ▼          ▼
 Reason     Verify    Reflect    Answer
 180ms      0.5ms     200ms      150ms
 ~150 tok   N/A       ~200 tok   ~100 tok
    │          │          │          │
    └──────────┴──────────┴──────────┘
                  ↓
           ┌──────────────┐
           │ Calculate    │
           │ - Total cost │
           │ - Savings    │
           │ - Metrics    │
           └──────────────┘
```

## Usage

### Automatic Tracking (Built-in)

Metrics are tracked **automatically** when using Kaelum:

```python
from kaelum import set_reasoning_model, enhance_stream

set_reasoning_model(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://localhost:8000/v1"
)

# Metrics tracked automatically
for chunk in enhance_stream("Calculate 15 × $12.99 + 8.5% tax"):
    print(chunk, end="")
```

**Output includes metrics**:
```
🧠 [REASON]
...reasoning steps...

🔍 [VERIFY]
   3 steps | Symbolic: 3/3 | 0.8ms

🔄 [REFLECT]
   Iteration 1: No issues found

✅ [ANSWER]
The total cost is $211.41

======================================================================
📊 [METRICS]
   Total: 350.2ms | Reason: 180.5ms | Verify: 0.8ms | Reflect: 150.0ms | Answer: 18.9ms
   Tokens: 450 | Cost: $0.00000450 | vs Commercial: $0.0045 | 💰 99.9% savings
======================================================================
```

### Manual Tracking

```python
from kaelum.core.metrics import CostTracker

tracker = CostTracker()

# Start a session
session_id = "custom_session_123"
tracker.start_session(session_id, metadata={"query": "Calculate..."})

# Log inference operations
tracker.log_inference(
    model_type="local_reasoning",
    tokens=150,
    latency_ms=180,
    cost=0.0000015,  # Local model cost
    session_id=session_id
)

# Get session metrics
metrics = tracker.get_session_metrics(session_id)
print(f"Total tokens: {metrics['total_tokens']}")
print(f"Total cost: ${metrics['total_cost']:.8f}")
print(f"Avg latency: {metrics['avg_latency_ms']:.1f}ms")

# Calculate savings vs commercial LLMs
savings = tracker.calculate_savings(session_id)
print(f"Saved: ${savings['savings']:.4f} ({savings['savings_percent']:.1f}%)")
```

## Metrics Breakdown

### Per-Stage Tracking

Each stage of the pipeline is tracked separately:

```python
# 1. Reasoning generation
tracker.log_inference(
    model_type="local_reasoning",
    tokens=150,
    latency_ms=180,
    cost=0.0000015,
    session_id=session_id
)

# 2. Verification (no tokens - symbolic computation)
# Tracked via latency only

# 3. Reflection (if triggered)
tracker.log_inference(
    model_type="local_reflection", 
    tokens=200,
    latency_ms=150,
    cost=0.0000020,
    session_id=session_id
)

# 4. Final answer generation
tracker.log_inference(
    model_type="local_answer",
    tokens=100,
    latency_ms=90,
    cost=0.0000010,
    session_id=session_id
)
```

### Session Metrics

```python
metrics = tracker.get_session_metrics(session_id)

{
    "session_id": "stream_1730567890",
    "total_tokens": 450,
    "total_cost": 0.00000450,
    "total_latency_ms": 420.5,
    "avg_latency_ms": 140.2,
    "inference_count": 3,
    "created_at": "2025-11-02T10:30:45"
}
```

### Savings Calculation

```python
savings = tracker.calculate_savings(session_id)

{
    "local_cost": 0.00000450,      # What you paid (local model)
    "commercial_cost": 0.0045,     # What GPT-4 would cost
    "savings": 0.00449550,         # Money saved
    "savings_percent": 99.9        # Percentage saved
}
```

## Cost Models

### Local Model Pricing

Kaelum uses these rough estimates for local models:

| Model Size | Cost per Token | Cost per 1M Tokens |
|-----------|----------------|-------------------|
| 1-3B | $0.00000001 | $0.01 |
| 7B | $0.00000001 | $0.01 |
| 8B+ | $0.00000002 | $0.02 |

**Note**: These are illustrative. Actual cost depends on:
- Hardware (GPU vs CPU)
- Power consumption
- Amortized infrastructure cost
- For Kaelum, it's essentially **free** (local inference)

### Commercial LLM Pricing (for comparison)

| Provider | Model | Cost per 1M Tokens |
|----------|-------|-------------------|
| OpenAI | GPT-4 Turbo | $10.00 |
| OpenAI | GPT-4o | $2.50 |
| Anthropic | Claude 3.5 Sonnet | $3.00 |
| Google | Gemini 1.5 Pro | $1.25 |

**Typical savings**: 99%+ when using Kaelum for reasoning

## Token Counting

### Approximation Method (Current)

```python
# Simple word-based estimate
tokens = len(text.split())
```

**Accuracy**: ±20% (good enough for cost tracking)

### Accurate Counting (Planned)

```python
import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")
tokens = len(encoder.encode(text))
```

**Accuracy**: Exact tokenization matching OpenAI

## Performance Tracking

### Latency Breakdown

Track where time is spent:

```python
metrics = tracker.get_session_metrics(session_id)

print("Latency breakdown:")
print(f"  Reasoning:  {180}ms  (51%)")
print(f"  Verification: {0.8}ms  (0.2%)")
print(f"  Reflection: {150}ms  (43%)")
print(f"  Answer:     {90}ms   (26%)")
print(f"  Total:      {420.8}ms")
```

### Throughput Monitoring

```python
from time import time

start = time()
sessions = []

for i in range(100):
    session_id = f"batch_{i}"
    tracker.start_session(session_id)
    # ... process query ...
    sessions.append(session_id)

elapsed = time() - start
throughput = len(sessions) / elapsed

print(f"Throughput: {throughput:.1f} queries/sec")
```

## Session Management

### Creating Sessions

```python
# Auto-generated ID
session_id = f"stream_{int(time.time() * 1000)}"

# With metadata
tracker.start_session(
    session_id,
    metadata={
        "query": "Calculate compound interest",
        "user_id": "user_123",
        "routing_strategy": "balanced"
    }
)
```

### Querying Sessions

```python
# Get specific session
metrics = tracker.get_session_metrics("stream_123")

# Get all sessions (if persistent storage added)
all_sessions = tracker.list_sessions()

# Filter by date/user/etc
recent = tracker.list_sessions(since="2025-11-01")
```

### Session Lifecycle

```
1. start_session(id, metadata)
   ↓
2. log_inference(..., session_id=id)  [repeat per stage]
   ↓
3. get_session_metrics(id)
   ↓
4. calculate_savings(id)
```

## Cost Analysis Examples

### Example 1: Math Query

```
Query: "Calculate 15 × $12.99 + 8.5% tax"

Local (Kaelum):
- Tokens: 450
- Cost: $0.0000045
- Latency: 350ms

Commercial (GPT-4):
- Tokens: 450
- Cost: $0.0045
- Latency: 800ms

Savings: $0.0044955 (99.9%)
Also 2.3x faster!
```

### Example 2: Complex Reasoning

```
Query: "Multi-step financial planning scenario"

Local (Kaelum with reflection):
- Tokens: 1200
- Cost: $0.000012
- Latency: 800ms

Commercial (GPT-4):
- Tokens: 1200  
- Cost: $0.012
- Latency: 2000ms

Savings: $0.011988 (99.9%)
Also 2.5x faster!
```

### Example 3: High Volume

```
Scenario: 10,000 queries/day

Local (Kaelum):
- Cost: $0.045/day = $1.35/month
- Latency: 350ms avg

Commercial (GPT-4):
- Cost: $45/day = $1,350/month
- Latency: 800ms avg

Annual savings: $16,178
```

## Persistent Storage (Coming Soon)

### Current: In-Memory

```python
tracker = CostTracker()
# Sessions lost on restart
```

### Planned: Database Backend

```python
tracker = CostTracker(
    backend="sqlite",
    db_path="metrics.db"
)

# Sessions persisted across restarts
# Query historical data
# Generate reports
```

## Metrics API

### Core Methods

```python
class CostTracker:
    def start_session(self, session_id: str, metadata: Dict = None):
        """Initialize a new tracking session."""
        
    def log_inference(self, model_type: str, tokens: int, 
                     latency_ms: float, cost: float, session_id: str):
        """Log a single inference operation."""
        
    def get_session_metrics(self, session_id: str) -> Dict:
        """Get aggregated metrics for a session."""
        
    def calculate_savings(self, session_id: str, 
                         commercial_model: str = "gpt-4") -> Dict:
        """Calculate cost savings vs commercial LLM."""
```

### Return Types

**get_session_metrics**:
```python
{
    "session_id": str,
    "total_tokens": int,
    "total_cost": float,
    "total_latency_ms": float,
    "avg_latency_ms": float,
    "inference_count": int,
    "created_at": str
}
```

**calculate_savings**:
```python
{
    "local_cost": float,
    "commercial_cost": float,
    "savings": float,
    "savings_percent": float,
    "comparison_model": str
}
```

## Best Practices

### 1. Always Track Sessions

```python
# ✓ Good
session_id = "query_123"
tracker.start_session(session_id)
tracker.log_inference(..., session_id=session_id)

# ✗ Bad (no session context)
tracker.log_inference(..., session_id=None)
```

### 2. Include Metadata

```python
tracker.start_session(
    session_id,
    metadata={
        "query_type": "math",
        "user": "user_123",
        "routing_strategy": "balanced"
    }
)
```

### 3. Track All Stages

```python
# Track reasoning, reflection, AND answer generation
# Don't skip any stage for accurate metrics
```

### 4. Monitor Trends

```python
# Periodically check metrics
if avg_latency > 500:
    print("Warning: Latency degrading")
if savings_percent < 95:
    print("Warning: Not achieving target savings")
```

## Integration with Router

The router uses metrics to learn performance:

```python
from kaelum.core.router import Router

router = Router(learning_enabled=True)

# Router records outcomes with metrics
decision = router.route(query)
result = orchestrator.infer(query)
router.record_outcome(decision, result)  # Includes metrics!
```

## Files

```
kaelum/core/metrics.py         # CostTracker implementation
kaelum/runtime/orchestrator.py # Automatic metrics integration
```

## Related Documentation

- [Routing System](./ROUTING.md) - Uses metrics for strategy learning
- [Verification Engine](./VERIFICATION.md) - Latency tracked separately
- [Reflection Engine](./REFLECTION.md) - Token/latency impact of reflection

---

**Metrics give you visibility into the value Kaelum provides** - see exactly how much you're saving in real-time.
