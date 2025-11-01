# KaelumAI

Reasoning layer using your own local LLM.

## How It Works

```
User Query → Your Model → Reasoning → Verification → Reflection → Answer
```

## Usage

```python
from kaelum import enhance, set_reasoning_model

# Set YOUR reasoning model with tweakable parameters
set_reasoning_model(
    provider="ollama",
    model="llama3.2:3b",
    temperature=0.7,              # 0.0-2.0 (lower = deterministic)
    max_tokens=2048,              # Max response length
    max_reflection_iterations=2,  # Self-correction cycles (0-5)
    use_symbolic_verification=True,   # Math checking
    use_factual_verification=False,   # RAG checking
)

# User sends query, get enhanced answer
result = enhance("What is 15% of 200?")
```

## Install

```bash
git clone https://github.com/ashworks1706/KaelumAI.git
cd KaelumAI
pip install -e .

# Run YOUR model:
ollama pull llama3.2:3b && ollama serve
```

## Tweakable Parameters

### Model Settings
- **`provider`**: `"ollama"`, `"vllm"`, or `"custom"`
- **`model`**: Model name (e.g., `"llama3.2:3b"`, `"qwen2.5:7b"`)
- **`base_url`**: Model endpoint (optional, defaults set)

### Generation Settings
- **`temperature`**: `0.0-2.0`
  - `0.1-0.3`: Deterministic (math, code, facts)
  - `0.7-0.9`: Balanced (general reasoning)
  - `1.0-1.5`: Creative (brainstorming)
- **`max_tokens`**: `1-128000` (response length limit)

### Reasoning Settings
- **`max_reflection_iterations`**: `0-5`
  - `0`: No self-correction (fastest)
  - `2`: Balanced (default)
  - `3-5`: Best quality (slower)

### Verification Settings
- **`use_symbolic_verification`**: `True/False` (math checking with SymPy)
- **`use_factual_verification`**: `True/False` (RAG-based fact checking)
- **`rag_adapter`**: RAG adapter instance (if factual verification enabled)

## Testing

```bash
# Test with default settings
python example.py

# Test different configurations
python test_settings.py
```

## Features

- **Reasoning**: Structured step-by-step reasoning
- **Verification**: Math checking with SymPy
- **Reflection**: Self-correction through iteration
- **RAG**: Optional factual verification with vector DB
