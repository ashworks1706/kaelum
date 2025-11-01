# KaelumAI

Reasoning layer for lightweight LLMs.

## Usage

```python
from kaelum import enhance

# Fast (1-2s)
enhance("What is 15% of 200?")

# With reasoning trace (8-12s)
enhance("What is 15% of 200?", fast=False)

# Different model
enhance("Question", model="qwen2.5:7b")
```

## Install

```bash
git clone https://github.com/ashworks1706/KaelumAI.git
cd KaelumAI
pip install -e .
ollama pull llama3.2:3b
```

## Structure

```
kaelum/
├── __init__.py         # enhance() function
├── core/
│   ├── reasoning.py    # LLM client
│   ├── verification.py # Math verification
│   └── reflection.py   # Self-reflection
└── runtime/
    └── orchestrator.py # Pipeline

test_notebooks/
└── testing.ipynb       # Testing ground
```
