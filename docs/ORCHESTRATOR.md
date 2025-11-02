# 🎭 Orchestrator - The Reasoning Pipeline Conductor

The KaelumOrchestrator is the central coordinator that manages the entire reasoning workflow: **Generate → Verify → Reflect → Answer**.

## Overview

Think of the orchestrator as a conductor leading an orchestra. Each component (reasoning, verification, reflection) is an instrument, and the orchestrator ensures they all play together in perfect harmony.

```
                    ┌─────────────────────┐
                    │  KaelumOrchestrator │
                    │   "The Conductor"   │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
  │  Generator  │      │  Verifier   │      │  Reflector  │
  │ (Reasoning) │      │ (Checking)  │      │(Correction) │
  └─────────────┘      └─────────────┘      └─────────────┘
         │                     │                     │
         └─────────────────────┴─────────────────────┘
                               │
                        Final Answer
```

## The Four-Stage Pipeline

### Stage 1: Generate Reasoning 🧠

```python
# LLM generates step-by-step reasoning
trace_text = self.generator.generate_reasoning(query, stream=True)

# Example output:
"""
1. Base cost: 15 × $12.99 = $194.85
2. Calculate tax: $194.85 × 0.085 = $16.56
3. Total: $194.85 + $16.56 = $211.41
"""

# Parse into structured steps
trace = ["Base cost: 15 × $12.99 = $194.85", ...]
```

**Purpose**: Get the LLM to think step-by-step
**Latency**: 150-300ms
**Tokens**: 100-300

### Stage 2: Verify Steps 🔍

```python
# Check reasoning correctness
errors, details = self.verification.verify_trace(trace)

# Example details:
{
    "total_steps": 3,
    "symbolic_checks": 3,
    "symbolic_passed": 3,
    "factual_checks": 0,
    "factual_passed": 0
}

# Example errors (if any):
["Step 2: Expected 16.56, got 16.5"]
```

**Purpose**: Catch errors before they propagate
**Latency**: <5ms (symbolic), 50-200ms (factual)
**Tokens**: 0 (no LLM calls)

### Stage 3: Reflect (If Needed) 🔄

```python
# If errors found, trigger reflection
if errors or self.config.max_reflection_iterations > 0:
    trace = self.reflection.enhance_reasoning(query, trace)
    
# Reflection loop:
# 1. Ask LLM: "What's wrong with your reasoning?"
# 2. Ask LLM: "Fix the problems"
# 3. Re-verify the fixes
# 4. Repeat up to max_iterations times
```

**Purpose**: Self-correction of reasoning errors
**Latency**: 100-300ms per iteration
**Tokens**: 200-400 per iteration

### Stage 4: Generate Answer ✅

```python
# Generate final answer based on verified reasoning
answer = self.generator.generate_answer(query, trace, stream=True)

# Example:
"""
The total cost is $211.41.

This includes the base cost of $194.85 
plus $16.56 in sales tax (8.5%).
"""
```

**Purpose**: Produce human-friendly final answer
**Latency**: 80-150ms
**Tokens**: 50-150

## Usage

### Basic Setup

```python
from kaelum import set_reasoning_model

# Initialize orchestrator
set_reasoning_model(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://localhost:8000/v1",
    max_reflection_iterations=2,
    use_symbolic_verification=True,
    use_factual_verification=False
)
```

### Streaming Inference

```python
from kaelum import enhance_stream

# Get real-time updates as reasoning happens
for chunk in enhance_stream("Calculate 15 × $12.99 + 8.5% tax"):
    print(chunk, end="", flush=True)
```

**Output**:
```
🧠 [REASON]

1. Base cost: 15 × $12.99 = $194.85
2. Tax: $194.85 × 0.085 = $16.56
3. Total: $194.85 + $16.56 = $211.41

🔍 [VERIFY]
   3 steps | Symbolic: 3/3 | 0.8ms

🔄 [REFLECT]
   Iteration 1: No issues found

✅ [ANSWER]

The total cost is $211.41

======================================================================
📊 [METRICS]
   Total: 350.2ms | Reason: 180.5ms | Verify: 0.8ms | Answer: 168.9ms
   Tokens: 450 | Cost: $0.00000450 | vs Commercial: $0.0045 | 💰 99.9% savings
======================================================================
```

### Synchronous Inference

```python
from kaelum.runtime.orchestrator import KaelumOrchestrator
from kaelum.core.config import KaelumConfig, LLMConfig

# Create orchestrator
config = KaelumConfig(
    reasoning_llm=LLMConfig(
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen2.5-7B-Instruct"
    ),
    max_reflection_iterations=2,
    use_symbolic_verification=True,
    use_factual_verification=False
)

orchestrator = KaelumOrchestrator(config)

# Get complete result at once
result = orchestrator.infer(query, stream=False)

# Access components
print(f"Reasoning: {result['reasoning_trace']}")
print(f"Answer: {result['answer']}")
print(f"Errors: {result['verification_errors']}")
print(f"Metrics: {result['metrics']}")
```

### With Adaptive Routing

```python
from kaelum import set_reasoning_model

# Enable router to adapt strategy per query
set_reasoning_model(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://localhost:8000/v1",
    enable_routing=True  # 🎯 Adaptive strategy selection
)

# Math query → Router selects SYMBOLIC_HEAVY
result1 = enhance("Calculate compound interest")

# Factual query → Router selects FACTUAL_HEAVY
result2 = enhance("When did WWII end?")

# Code query → Router selects FAST
result3 = enhance("Is this Python valid?")
```

## Configuration Options

### Orchestrator Parameters

```python
KaelumOrchestrator(
    config: KaelumConfig,              # Main configuration
    rag_adapter=None,                  # RAG adapter for factual verification
    reasoning_system_prompt=None,      # Custom system prompt
    reasoning_user_template=None,      # Custom user template
    enable_routing=False               # Enable adaptive routing
)
```

### KaelumConfig Options

```python
KaelumConfig(
    reasoning_llm: LLMConfig,          # LLM configuration
    max_reflection_iterations: int,    # 0-5, default 2
    use_symbolic_verification: bool,   # Math checks, default True
    use_factual_verification: bool     # RAG checks, default False
)
```

### LLMConfig Options

```python
LLMConfig(
    base_url: str,        # e.g., "http://localhost:8000/v1"
    model: str,           # e.g., "Qwen/Qwen2.5-7B-Instruct"
    api_key: str = None,  # Optional for local servers
    temperature: float = 0.7,
    max_tokens: int = 2048
)
```

## Custom Prompts

### Reasoning System Prompt

```python
custom_system = """
You are an expert mathematician and logician.
Think step-by-step and show all your work.
Use precise calculations and clear explanations.
"""

set_reasoning_model(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://localhost:8000/v1",
    reasoning_system_prompt=custom_system
)
```

### Reasoning User Template

```python
custom_template = """
Problem: {query}

Solve this step-by-step:
1. Identify what's being asked
2. Break down the calculation
3. Show all work
4. Verify your answer

Your reasoning:
"""

set_reasoning_model(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://localhost:8000/v1",
    reasoning_user_template=custom_template
)
```

## Internal Flow

### Streaming Mode (_infer_stream)

```python
def _infer_stream(self, query: str):
    session_id = f"stream_{int(time.time() * 1000)}"
    self.metrics.start_session(session_id, metadata={"query": query[:50]})
    
    # Stage 1: Generate reasoning (streaming)
    yield "🧠 [REASON]\n\n"
    trace_text = ""
    for chunk in self.generator.generate_reasoning(query, stream=True):
        trace_text += chunk
        yield chunk
    
    trace = self._parse_trace(trace_text)
    
    # Stage 2: Verify
    yield "\n\n🔍 [VERIFY]\n"
    errors, details = self.verification.verify_trace(trace)
    yield f"   {details['total_steps']} steps | Symbolic: {details['symbolic_passed']}/{details['symbolic_checks']} | {verify_time:.1f}ms\n"
    
    # Stage 3: Reflect (if needed)
    if errors or self.config.max_reflection_iterations > 0:
        yield "\n🔄 [REFLECT]\n"
        trace = self.reflection.enhance_reasoning(query, trace)
    
    # Stage 4: Generate answer (streaming)
    yield "\n✅ [ANSWER]\n\n"
    for chunk in self.generator.generate_answer(query, trace, stream=True):
        yield chunk
    
    # Metrics summary
    yield "\n\n" + "="*70 + "\n"
    yield f"📊 [METRICS]\n"
    yield f"   Total: {total_time:.1f}ms | ..."
```

### Synchronous Mode (_infer_sync)

```python
def _infer_sync(self, query: str, routing_decision=None) -> Dict:
    session_id = f"sync_{int(time.time() * 1000)}"
    self.metrics.start_session(session_id)
    
    # Stage 1: Generate
    trace_text = self.generator.generate_reasoning(query, stream=False)
    trace = self._parse_trace(trace_text)
    
    # Stage 2: Verify
    errors, details = self.verification.verify_trace(trace)
    
    # Stage 3: Reflect
    if errors or self.config.max_reflection_iterations > 0:
        trace = self.reflection.enhance_reasoning(query, trace)
    
    # Stage 4: Answer
    answer = self.generator.generate_answer(query, trace, stream=False)
    
    # Return structured result
    return {
        "query": query,
        "reasoning_trace": trace,
        "answer": answer,
        "verification_errors": errors,
        "verification_details": details,
        "metrics": {...}
    }
```

## Error Handling

### Verification Failures

```python
# If verification fails, reflection is automatically triggered
errors, details = self.verification.verify_trace(trace)

if errors:
    # Reflection fixes errors
    trace = self.reflection.enhance_reasoning(query, trace)
    
    # Re-verify
    errors, details = self.verification.verify_trace(trace)
```

### LLM Failures

```python
try:
    trace_text = self.generator.generate_reasoning(query)
except Exception as e:
    # Fallback: return error message
    return {
        "error": str(e),
        "query": query,
        "reasoning_trace": [],
        "answer": "Error generating reasoning"
    }
```

### Parsing Failures

```python
# If trace parsing fails, treat entire output as one step
trace = []
for line in trace_text.split("\n"):
    if line.strip() and (line[0].isdigit() or line.startswith("-")):
        trace.append(line.strip())

if not trace:
    # Fallback: use entire text as single step
    trace = [trace_text.strip()]
```

## Performance Tuning

### Speed Priority

```python
set_reasoning_model(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://localhost:8000/v1",
    max_reflection_iterations=0,       # No reflection
    use_symbolic_verification=True,    # Fast checks only
    use_factual_verification=False     # Skip RAG
)
```

**Expected latency**: 150-200ms

### Accuracy Priority

```python
set_reasoning_model(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://localhost:8000/v1",
    max_reflection_iterations=3,       # Deep reflection
    use_symbolic_verification=True,    # All checks
    use_factual_verification=True,     # Enable RAG
    rag_adapter=rag
)
```

**Expected latency**: 800-1200ms

### Balanced (Recommended)

```python
set_reasoning_model(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://localhost:8000/v1",
    max_reflection_iterations=2,       # Standard reflection
    use_symbolic_verification=True,    # Math checks
    use_factual_verification=False     # Skip RAG unless needed
)
```

**Expected latency**: 300-500ms

## Integration Patterns

### With LangChain

```python
from langchain.tools import Tool
from kaelum import kaelum_enhance_reasoning

reasoning_tool = Tool(
    name="kaelum_reasoning",
    func=kaelum_enhance_reasoning,
    description="Enhanced reasoning for complex problems"
)

agent = create_agent(
    llm=ChatOpenAI(model="gpt-4"),
    tools=[reasoning_tool, ...]
)

agent.invoke("Calculate compound interest...")
```

### With RAG Systems

```python
from kaelum import set_reasoning_model
from your_rag import ChromaDBAdapter

rag = ChromaDBAdapter(collection="knowledge_base")

set_reasoning_model(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://localhost:8000/v1",
    use_factual_verification=True,
    rag_adapter=rag
)
```

### As Microservice

```python
from fastapi import FastAPI
from kaelum import set_reasoning_model, enhance

app = FastAPI()

set_reasoning_model(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://localhost:8000/v1"
)

@app.post("/reason")
async def reason(query: str):
    result = enhance(query)
    return {"result": result}
```

## Best Practices

1. **Always verify**: Enable at least symbolic verification
2. **Use streaming**: Better UX for users waiting for results
3. **Monitor metrics**: Track latency and token usage
4. **Tune reflection**: Start with 2 iterations, adjust based on accuracy needs
5. **Custom prompts**: Tailor system prompts to your domain
6. **Enable routing**: Let the system learn optimal strategies

## Files

```
kaelum/runtime/orchestrator.py  # Main orchestrator implementation
kaelum/core/config.py           # Configuration classes
kaelum/__init__.py              # Public API wrappers
```

## Related Documentation

- [Verification Engine](./VERIFICATION.md) - Stage 2: Verification
- [Reflection Engine](./REFLECTION.md) - Stage 3: Self-correction
- [Routing System](./ROUTING.md) - Adaptive strategy selection
- [Metrics Tracking](./METRICS.md) - Performance monitoring

---

**The Orchestrator is the heart of Kaelum** - it ensures every reasoning task flows through verification and reflection for maximum accuracy.
