# KaelumAI Setup Guide - Open Source Version

## 🚀 Quick Start with Ollama (Recommended)

### 1. Install Ollama

**Linux/WSL:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**
```bash
brew install ollama
```

**Windows:**
Download from https://ollama.com/download

### 2. Start Ollama Service

```bash
ollama serve
```

### 3. Pull Recommended Models

```bash
# Qwen 2.5 - Excellent reasoning (7B model, ~4.7GB)
ollama pull qwen2.5:7b

# Alternative: Llama 3.2 (3B model, smaller/faster)
ollama pull llama3.2:3b

# Alternative: Mistral (7B model)
ollama pull mistral:7b
```

### 4. Install KaelumAI Dependencies

```bash
cd KaelumAI
pip install -r requirements.txt
```

### 5. Configure Environment

Create `.env` file:
```bash
# Ollama is the default, no API key needed!
# Just make sure Ollama is running

# Optional: Enable RAG
USE_RAG=false

# Optional: Enable Redis cache (install Redis first)
USE_CACHE=true
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 6. Run the Server

```bash
python app/main.py
```

### 7. Test It!

```bash
curl -X POST http://localhost:8000/verify_reasoning \
  -H "Content-Type: application/json" \
  -d '{
    "query": "If 3x + 5 = 11, what is x?",
    "config": {
      "llm": {
        "model": "qwen2.5:7b",
        "provider": "ollama",
        "temperature": 0.7
      }
    }
  }'
```

---

## 🔧 Advanced Setup

### Option 1: vLLM (Production, High Performance)

**Install vLLM:**
```bash
pip install vllm
```

**Run vLLM Server:**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8000
```

**Configure KaelumAI:**
```python
from kaelum.core.config import LLMConfig, MCPConfig

config = MCPConfig(
    llm=LLMConfig(
        model="Qwen/Qwen2.5-7B-Instruct",
        provider="vllm",
        base_url="http://localhost:8000/v1"
    )
)
```

### Option 2: Enable RAG with ChromaDB

**Install ChromaDB (already in requirements):**
```bash
pip install chromadb
```

**Add Knowledge to RAG:**
```python
from kaelum.runtime.orchestrator import MCP
from kaelum.core.config import MCPConfig

# Enable RAG
config = MCPConfig(use_rag=True)
mcp = MCP(config)

# Add knowledge base documents
verifier = mcp.verification_engine.factual_verifier
verifier.add_to_knowledge_base(
    texts=[
        "The capital of France is Paris.",
        "Water boils at 100°C at sea level.",
        "Python is a programming language created by Guido van Rossum."
    ],
    metadatas=[
        {"source": "geography"},
        {"source": "physics"},
        {"source": "programming"}
    ]
)

print(f"Knowledge base size: {verifier.get_knowledge_base_size()}")
```

### Option 3: Enable Redis Caching

**Install Redis:**

**Linux:**
```bash
sudo apt install redis-server
sudo systemctl start redis
```

**macOS:**
```bash
brew install redis
brew services start redis
```

**Docker:**
```bash
docker run -d -p 6379:6379 redis:alpine
```

**Test Cache:**
```python
from kaelum.runtime.orchestrator import MCP
from kaelum.core.config import MCPConfig

config = MCPConfig(use_cache=True)
mcp = MCP(config)

# First call - will hit LLM
result1 = mcp.infer("What is 2+2?")
print(f"Latency: {result1.diagnostics['latency']:.2f}s")

# Second call - will hit cache
result2 = mcp.infer("What is 2+2?")
print(f"Cached: {result2.diagnostics.get('cached', False)}")
print(f"Latency: {result2.diagnostics['latency']:.3f}s")
```

---

## 🎯 Model Recommendations

### For Reasoning Tasks

| Model | Size | Best For | Speed |
|-------|------|----------|-------|
| **qwen2.5:7b** | 4.7GB | Math, logic, reasoning | Medium |
| **qwen2.5:14b** | 9GB | Complex reasoning | Slow |
| **llama3.2:3b** | 2GB | Fast responses | Fast |
| **mistral:7b** | 4.1GB | General tasks | Medium |
| **deepseek-r1:7b** | 4.7GB | Specialized reasoning | Medium |

### Configuration Examples

**Fast & Light (Good for Development):**
```python
config = MCPConfig(
    llm=LLMConfig(model="llama3.2:3b", provider="ollama"),
    max_reflection_iterations=1,
    use_symbolic=True,
    use_rag=False
)
```

**Balanced (Recommended for Production):**
```python
config = MCPConfig(
    llm=LLMConfig(model="qwen2.5:7b", provider="ollama"),
    verifier_llm=LLMConfig(model="qwen2.5:7b", provider="ollama", temperature=0.3),
    max_reflection_iterations=2,
    use_symbolic=True,
    use_rag=True,
    use_cache=True
)
```

**Maximum Accuracy (Slow but Thorough):**
```python
config = MCPConfig(
    llm=LLMConfig(model="qwen2.5:14b", provider="ollama"),
    verifier_llm=LLMConfig(model="qwen2.5:14b", provider="ollama", temperature=0.2),
    reflector_llm=LLMConfig(model="qwen2.5:14b", provider="ollama", temperature=0.4),
    max_reflection_iterations=3,
    use_symbolic=True,
    use_rag=True,
    use_cache=True
)
```

---

## 📊 Performance Comparison

### Ollama vs Gemini

| Metric | Gemini API | Ollama (Local) |
|--------|------------|----------------|
| **Cost per 1K requests** | $15-30 | $0 (electricity) |
| **Latency** | 2-4s | 1-3s (depends on hardware) |
| **Privacy** | Sent to Google | 100% local |
| **Fine-tuning** | No | Yes |
| **Offline** | No | Yes |
| **Setup** | API key | Install + download model |

### Hardware Requirements

**Minimum:**
- RAM: 8GB
- Model: llama3.2:3b
- Inference: ~1-2s per request

**Recommended:**
- RAM: 16GB
- GPU: Optional (10x faster)
- Model: qwen2.5:7b
- Inference: ~0.5-1s per request

**Optimal:**
- RAM: 32GB
- GPU: NVIDIA with 8GB+ VRAM
- Model: qwen2.5:14b
- Inference: ~0.3-0.5s per request

---

## 🔄 Migration from Gemini

If you were using Gemini before:

**Old config:**
```python
config = MCPConfig(
    llm=LLMConfig(
        model="gemini-1.5-flash",
        provider="gemini",
        api_key=os.getenv("GEMINI_API_KEY")
    )
)
```

**New config (Ollama):**
```python
config = MCPConfig(
    llm=LLMConfig(
        model="qwen2.5:7b",
        provider="ollama",
        base_url="http://localhost:11434"  # Optional, this is default
    )
)
```

No code changes needed - just change the config!

---

## 🐛 Troubleshooting

### Issue: "Ollama connection refused"
**Solution:**
```bash
# Make sure Ollama is running
ollama serve

# Check if it's accessible
curl http://localhost:11434/api/tags
```

### Issue: "Model not found"
**Solution:**
```bash
# Pull the model
ollama pull qwen2.5:7b

# List available models
ollama list
```

### Issue: "Out of memory"
**Solution:**
- Use a smaller model: `ollama pull llama3.2:3b`
- Close other applications
- Reduce `max_tokens` in config

### Issue: "Slow inference"
**Solutions:**
- Use smaller model
- Enable GPU if available
- Reduce `max_reflection_iterations`
- Enable caching

### Issue: "Redis connection failed"
**Solution:**
```bash
# The system will fall back to in-memory cache
# To fix Redis:
sudo systemctl start redis  # Linux
brew services start redis   # macOS
```

---

## 📚 Next Steps

1. ✅ **Setup complete!** Try the examples in `examples/`
2. 📖 Read `ARCHITECTURE_ANALYSIS.md` to understand the system
3. 🔧 Customize config for your use case
4. 🧪 Run tests: `pytest tests/`
5. 🚀 Deploy to production (see `DEPLOYMENT.md`)

---

## 💡 Tips

- **Start with `llama3.2:3b`** for development (fast downloads, quick responses)
- **Switch to `qwen2.5:7b`** for production (better reasoning)
- **Enable caching** to dramatically reduce latency for repeated queries
- **Use RAG** if you need factual verification
- **Monitor metrics** via `/metrics` endpoint

---

## 🆘 Getting Help

- 📖 Documentation: `README.md`, `ARCHITECTURE_ANALYSIS.md`
- 🐛 Issues: https://github.com/ashworks1706/KaelumAI/issues
- 💬 Ask me: I can help with any setup problems!
