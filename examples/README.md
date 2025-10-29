# KaelumAI Examples

This directory contains example scripts demonstrating various usage patterns for KaelumAI.

## Prerequisites

```bash
# Install dependencies
pip install -r ../requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Examples

### 1. Basic Usage (`example_basic.py`)
Simple demonstration of KaelumAI's core functionality.

```bash
python example_basic.py
```

### 2. ModelRuntime (`example_runtime.py`)
Shows how to use the ModelRuntime interface for tool-based integration.

```bash
python example_runtime.py
```

### 3. Custom Configuration (`example_custom_config.py`)
Demonstrates advanced configuration options including custom LLM settings.

```bash
python example_custom_config.py
```

### 4. LangChain Integration (`example_langchain.py`)
Shows integration with LangChain agents (requires LangChain installation).

```bash
pip install langchain langchain-openai
python example_langchain.py
```

### 5. API Usage (`example_api.py`)
Demonstrates calling the FastAPI endpoints.

First start the API server:
```bash
cd ..
uvicorn app.main:app --reload
```

Then run the example:
```bash
python example_api.py
```

## API Keys

All examples require API keys for the LLM providers:

- **OpenAI**: Set `OPENAI_API_KEY` environment variable
- **Anthropic** (optional): Set `ANTHROPIC_API_KEY` environment variable

## Notes

- Most examples use GPT-4o-mini by default for cost efficiency
- You can modify the model configurations in each example
- The examples demonstrate both verified and unverified modes
