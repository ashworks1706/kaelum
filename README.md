# KaelumAI

Reasoning layer that uses **your own local LLM** for enhanced reasoning.

## How It Works

```
User Query → KaelumAI → YOUR Local Model → Reasoning + Verification → Answer
```

Simple: User sends query, your model does the reasoning, return answer.

## Usage

```python
from kaelum import enhance, set_reasoning_model

# 1. Set YOUR reasoning model (once at startup)
set_reasoning_model(
    provider="ollama",  # or "vllm", "custom"
    model="llama3.2:3b",
)

# 2. Users call enhance()
result = enhance("What is 15% of 200?")
```

## Install

```bash
git clone https://github.com/ashworks1706/KaelumAI.git
cd KaelumAI
pip install -e .

# Run YOUR reasoning model (pick one):
# Ollama: ollama pull llama3.2:3b && ollama serve
# vLLM: python -m vllm.entrypoints.openai.api_server --model <model>
# Custom: Any OpenAI-compatible server
```

## Structure

```
kaelum/
├── __init__.py         # enhance() function
├── core/
│   ├── reasoning.py    # LLM client (Ollama/vLLM/custom)
│   ├── verification.py # Math + factual verification
│   ├── reflection.py   # Self-correction
│   └── rag_adapter.py  # Optional RAG verification
└── runtime/
    └── orchestrator.py # Reasoning pipeline
```
