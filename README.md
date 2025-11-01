
# **KaelumAI 🧠**

### *The All-in-One Reasoning Layer for Agentic LLMs*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io)

---

## 🎯 The Problem

Large Language Models *sound* intelligent but often **reason poorly**.
They hallucinate, contradict themselves, and produce logical or mathematical errors — all with convincing fluency.

“Chain-of-thought” and “self-reflection” improve *style*, not *truth*.
There has never been a **standard reasoning layer** that:

* verifies intermediate logic,
* cross-checks factual claims, and
* exposes *why* a model reached its conclusion.

---

## 💡 The Solution — **The Reasoning Layer**

**KaelumAI** is a **Modular Cognitive Processor (MCP)** — a reasoning middleware that can be plugged into any LLM runtime or agent framework.

It functions as a **logic co-processor**, verifying reasoning traces, refining them through multi-LLM reflection, and returning *auditable conclusions* with quantitative confidence.

> 🧠 *Think of it as a GPU for reasoning* — a plug-in layer that validates and accelerates thought inside any AI system.

---

## ✨ Core Features

### 1. **One-Line API** 🎯
```python
from kaelum import enhance

result = enhance("What is 25% of 80?")
```

### 2. **Symbolic Verification** 🧮
- Automatic math verification using SymPy
- Catches algebraic errors, equation inconsistencies
- No network calls, pure computation

### 3. **Pluggable RAG Verification** 🔍
```python
from kaelum import enhance
from kaelum.core.rag_adapter import ChromaAdapter

adapter = ChromaAdapter(collection=my_chroma_db)
result = enhance(
    "What is the speed of light?",
    rag_adapter=adapter,
    use_factual_verification=True
)
```

**Supported RAG Databases:**
- ChromaDB
- Qdrant
- Weaviate
- Custom (bring your own search function)

### 4. **Adaptive Reflection** 🔄
- Skips reflection when confidence > 0.85
- Automatically refines low-confidence answers
- Configurable max iterations (default: 2)

### 5. **Confidence Scoring** 📊
- Quantified reliability for every answer
- Combines symbolic verification + pattern matching
- Helps filter unreliable outputs

### 6. **LRU Caching** ⚡
- 1000-entry in-memory cache
- Optional Redis backend
- Massive speedup for repeated queries

### 7. **Multi-Provider Support** 🌐
- Ollama (local)
- OpenAI
- vLLM
- OpenRouter
- Any OpenAI-compatible API

---

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│          User Query                 │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│        enhance() - Public API       │
│  • Mode detection (auto/math/code)  │
│  • Config setup                     │
│  • Cache check                      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│     MCP Orchestrator (Single LLM)   │
├─────────────────────────────────────┤
│  1. ReasoningGenerator              │
│     └─> Generate reasoning trace    │
│                                     │
│  2. VerificationEngine              │
│     ├─> SymbolicVerifier (SymPy)   │
│     └─> FactualVerifier (RAG)      │
│                                     │
│  3. ReflectionEngine                │
│     └─> Self-correction loop        │
│                                     │
│  4. ConfidenceScorer                │
│     └─> Calculate reliability       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      Enhanced Response + Metadata   │
│  • Final answer                     │
│  • Confidence score                 │
│  • Reasoning trace                  │
│  • Verification results             │
└─────────────────────────────────────┘
```

**Key Design Principles:**
- **Single LLM** - Same model for all tasks (cost-efficient)
- **No fallbacks** - Fails loudly for proper debugging
- **Stateless** - Easy to scale horizontally
- **Modular** - Swap verification backends easily

---

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama** (for local models)
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:7b  # or llama3.2:3b
```

2. **Clone & Install**
```bash
git clone https://github.com/ashworks1706/KaelumAI.git
cd KaelumAI
pip install -r requirements.txt
```

### Basic Usage

**1. Simple Enhancement**
```python
from kaelum import enhance

# Math reasoning
result = enhance("If 3x + 5 = 20, what is x?", mode="math")
print(result)
# Output: "x = 5. First, subtract 5: 3x = 15. Then divide: x = 5."
```

**2. With Custom Model**
```python
result = enhance(
    "Explain quantum entanglement",
    model="llama3.2:3b",
    mode="auto"
)
```

**3. With RAG Verification**
```python
from kaelum import enhance
from kaelum.core.rag_adapter import ChromaAdapter
import chromadb

# Setup your knowledge base
client = chromadb.Client()
collection = client.get_or_create_collection("facts")
collection.add(
    documents=["The speed of light is 299,792,458 m/s"],
    ids=["fact1"]
)

# Use it for verification
adapter = ChromaAdapter(collection=collection)
result = enhance(
    "What is the speed of light?",
    rag_adapter=adapter,
    use_factual_verification=True
)
```

**4. Advanced Configuration**
```python
from kaelum import MCP, MCPConfig, LLMConfig

config = MCPConfig(
    llm=LLMConfig(
        model="qwen2.5:7b",
        provider="ollama",
        temperature=0.3,
        max_tokens=2048
    ),
    max_reflection_iterations=3,
    confidence_threshold=0.80,
    use_symbolic_verification=True,
    use_factual_verification=False
)

mcp = MCP(config)
result = mcp.infer("Complex reasoning query...")
print(f"Answer: {result.final}")
print(f"Confidence: {result.diagnostics['confidence']}")
```

### CLI Usage

```bash
# Simple query
kaelum "What is 15% of 240?"

# Math mode
kaelum "Solve x^2 + 5x + 6 = 0" --mode math

# Custom model
kaelum "Explain recursion" --model llama3.2:3b

# With streaming
kaelum "Write a poem about AI" --stream

# Disable cache
kaelum "Random question" --no-cache
```

---

## 📊 Benchmarking

KaelumAI includes a comprehensive benchmarking suite to measure improvements across 5 key areas:

### Available Benchmarks

| Benchmark | Tests | Metrics |
|-----------|-------|---------|
| **Speed** | 25 queries | Latency overhead, speedup factor |
| **Hallucination Detection** | 20 cases | Detection rate, false positives |
| **Tool Selection** | 25 scenarios | Accuracy, selection time |
| **Math Reasoning** | 20 problems | Answer correctness, step validity |
| **Agent Orchestration** | 10 workflows | Sequence accuracy, agent selection |

### Running Benchmarks

```bash
# Run all benchmarks
kaelum-benchmark all

# Run specific benchmark
kaelum-benchmark speed
kaelum-benchmark hallucination
kaelum-benchmark math

# Compare models
kaelum-benchmark all --models llama-3.2-3b qwen2.5:7b

# Visualize results
python benchmarks/visualize.py
```

### Example Results

```
Results saved to: benchmark_results/
├── speed_results.json
├── hallucination_results.json
├── tool_selection_results.json
├── math_results.json
├── orchestration_results.json
├── cost_analysis.json
└── summary.json
```

**Cost Tracking:**
- Total LLM calls
- Total tokens (input/output)
- Cost per query
- Latency per query
- Cache hit rate

See [`benchmarks/BENCHMARK_GUIDE.md`](benchmarks/BENCHMARK_GUIDE.md) for detailed documentation.

---

## 📂 Project Structure

```
KaelumAI/
├── kaelum/
│   ├── __init__.py                 # Public API (enhance function)
│   ├── cli.py                      # Command-line interface
│   ├── core/
│   │   ├── config.py              # LLMConfig, MCPConfig
│   │   ├── reasoning.py           # LLMClient, ReasoningGenerator
│   │   ├── verification.py        # SymbolicVerifier, FactualVerifier
│   │   ├── reflection.py          # ReflectionEngine
│   │   ├── confidence.py          # ConfidenceScorer
│   │   ├── cache.py               # LRU + Redis caching
│   │   └── rag_adapter.py         # RAG database adapters
│   └── runtime/
│       └── orchestrator.py        # MCP orchestration
├── benchmarks/
│   ├── benchmark_suite.py         # Core benchmark framework
│   ├── run_benchmarks.py          # Execution engine
│   ├── cost_tracker.py            # Cost/latency tracking
│   ├── visualize.py               # Results visualization
│   ├── test_cases.py              # Test dataset
│   ├── cli.py                     # Benchmark CLI
│   └── BENCHMARK_GUIDE.md         # Usage guide
├── examples/
│   ├── rag_chroma_example.py      # ChromaDB integration
│   ├── rag_qdrant_example.py      # Qdrant integration
│   └── rag_custom_example.py      # Custom RAG adapter
├── requirements.txt               # Core dependencies
└── README.md                      # This file
```

---

## 🔌 Integration Examples

### LangChain Integration

```python
from langchain.tools import Tool
from kaelum import enhance

def kaelum_reasoning_tool(query: str) -> str:
    """Enhanced reasoning with KaelumAI"""
    return enhance(query, mode="auto")

reasoning_tool = Tool(
    name="kaelum_reasoning",
    func=kaelum_reasoning_tool,
    description="Use this for complex reasoning, math problems, or when you need verified logic"
)

# Add to your agent
from langchain.agents import initialize_agent
agent = initialize_agent(
    tools=[reasoning_tool, ...],
    llm=your_llm,
    agent="zero-shot-react-description"
)
```

### LlamaIndex Integration

```python
from llama_index.tools import FunctionTool
from kaelum import enhance

reasoning_tool = FunctionTool.from_defaults(
    fn=lambda q: enhance(q, mode="auto"),
    name="kaelum_reasoning",
    description="Enhanced reasoning with verification"
)

# Add to your agent
from llama_index.agent import ReActAgent
agent = ReActAgent.from_tools(
    tools=[reasoning_tool, ...],
    llm=your_llm
)
```

### FastAPI Service

```python
from fastapi import FastAPI
from pydantic import BaseModel
from kaelum import enhance

app = FastAPI()

class Query(BaseModel):
    text: str
    mode: str = "auto"
    use_factual_verification: bool = False

@app.post("/reason")
async def reason(query: Query):
    result = enhance(
        query.text,
        mode=query.mode,
        use_factual_verification=query.use_factual_verification
    )
    return {"answer": result}
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# Optional: Set default API base
export OPENAI_API_BASE=http://localhost:11434/v1

# Optional: Set API key (not needed for Ollama)
export OPENAI_API_KEY=your-key-here
```

### Mode Templates

KaelumAI automatically adjusts prompts based on mode:

| Mode | Best For | Template |
|------|----------|----------|
| `auto` | General queries | Adaptive reasoning |
| `math` | Calculations, equations | Step-by-step math solving |
| `code` | Programming logic | Code-focused reasoning |
| `logic` | Formal reasoning | Logical deduction |
| `creative` | Open-ended tasks | Creative problem-solving |

### Performance Tuning

```python
config = MCPConfig(
    llm=LLMConfig(
        temperature=0.3,      # Lower = more deterministic
        max_tokens=2048,      # Increase for longer reasoning
    ),
    max_reflection_iterations=3,  # More iterations = better quality
    confidence_threshold=0.80,    # Higher = more strict
)
```

---

## 🎯 Roadmap

### Current (v0.2.0)
- ✅ One-line API
- ✅ Symbolic + RAG verification
- ✅ Adaptive reflection
- ✅ CLI interface
- ✅ Comprehensive benchmarks
- ✅ Multi-provider support

### Next (v0.3.0)
- 🔨 Streaming support (partially implemented)
- 🔨 Tool selection guardrails
- 🔨 Multi-agent orchestration
- 🔨 Prompt optimization
- 🔨 LangChain/LlamaIndex official plugins

### Future (v1.0.0)
- 📋 Self-consistency sampling
- 📋 Chain-of-Verification (CoVe)
- 📋 Tree-of-Thoughts (ToT)
- 📋 Program-Aided Language Models (PAL)
- 📋 Dashboard & analytics
- 📋 Distributed verification

---

## 🎯 Use Cases

### 1. **Cheap Model Enhancement**
Turn Llama 8B into a reliable reasoning engine without fine-tuning:
```python
result = enhance("Complex reasoning task", model="llama-3.2-3b")
```

### 2. **Math Education Platform**
Verify student explanations and catch errors:
```python
student_answer = "x = 5 because 3x + 5 = 20, so 3x = 15"
result = enhance(student_answer, mode="math")
# Checks symbolic correctness automatically
```

### 3. **RAG-Powered Fact-Checking**
Verify claims against your knowledge base:
```python
adapter = ChromaAdapter(company_knowledge_base)
result = enhance(
    "What's our refund policy?",
    rag_adapter=adapter,
    use_factual_verification=True
)
```

### 4. **Agent Tool Selection**
Improve agent tool choice accuracy:
```python
# Before KaelumAI: Agent might use wrong tool
# After KaelumAI: Enhanced reasoning leads to correct tool selection
result = enhance("Calculate 15% of 240", mode="auto")
# Correctly identifies this as math, not search
```

### 5. **Production LLM Guardrails**
Add verification layer to production systems:
```python
user_query = request.json['query']
verified_response = enhance(user_query, mode="auto")
# Only return if confidence > threshold
if verified_response.confidence > 0.8:
    return verified_response
```

---

## 📊 Target Metrics

| Priority | Metric | Target | Status |
|----------|--------|--------|--------|
| **Speed** | Latency overhead | <500ms | 🔨 Testing |
| **Hallucination** | Detection rate | >90% | 🔨 Testing |
| **Tool Selection** | Accuracy | >85% | 🔨 Testing |
| **Math** | Answer correctness | >95% | ✅ Achieved |
| **Orchestration** | Agent accuracy | >80% | 📋 Planned |
| **Cost** | $/1K queries | <$0.10 | ✅ Achieved |

---

## � Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_verification.py

# Run with coverage
pytest --cov=kaelum
```

### Code Quality

```bash
# Format code
black kaelum/

# Lint
ruff check kaelum/
```

### Adding New Features

1. **New Verification Backend**
   - Extend `VerificationEngine` in `kaelum/core/verification.py`
   - Add tests in `tests/test_verification.py`

2. **New RAG Adapter**
   - Implement `RAGAdapter` interface in `kaelum/core/rag_adapter.py`
   - Add example in `examples/`

3. **New Benchmark**
   - Add test cases in `benchmarks/test_cases.py`
   - Extend `BenchmarkSuite` in `benchmarks/benchmark_suite.py`

---

## 🤝 Contributing

We welcome contributions! Priority areas:

1. 🔥 **Implement research papers** - CoVe, ToT, PAL, etc.
2. 📊 **Expand benchmarks** - More test cases, metrics
3. 🔌 **New integrations** - More RAG databases, frameworks
4. 🐛 **Bug fixes** - Report issues on GitHub
5. 📖 **Documentation** - Examples, guides, tutorials

**Process:**
1. Fork the repo
2. Create branch: `feature/your-feature`
3. Make changes + add tests
4. Submit PR with clear description

---

## 📬 Contact & Community

- **GitHub:** [github.com/ashworks1706/KaelumAI](https://github.com/ashworks1706/KaelumAI)
- **Issues:** [Report bugs or request features](https://github.com/ashworks1706/KaelumAI/issues)
- **Email:** ashworks1706@gmail.com

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with:
- [OpenAI SDK](https://github.com/openai/openai-python) - LLM client
- [SymPy](https://www.sympy.org/) - Symbolic mathematics
- [FastAPI](https://fastapi.tiangolo.com/) - API framework
- [Pydantic](https://pydantic.dev/) - Data validation

Research inspiration:
- Chain-of-Verification (Meta)
- Self-Consistency (Google)
- ReAct (Princeton/Google)
- Program-Aided Language Models (CMU)
- Tree of Thoughts (Princeton)

---

**Made with ❤️ for the AI research community**
