# Kaelum v1.0.0 - Quickstart Guide

**Get started with Kaelum in under 5 minutes** ⚡

---

## 🚀 Installation

### Option 1: Docker (Recommended)
```bash
# Clone and start
git clone https://github.com/yourusername/KaelumAI.git
cd KaelumAI
docker-compose up -d

# Check health
docker-compose ps
```

### Option 2: Local Development
```bash
# Install package
pip install -e .

# Start vLLM server (separate terminal)
kaelum serve --model Qwen/Qwen2.5-7B-Instruct

# Or manually:
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000
```

---

## 💡 Basic Usage

### Python API
```python
from kaelum import set_reasoning_model, enhance

# Configure once
set_reasoning_model(
    base_url="http://localhost:8000/v1",
    model="Qwen/Qwen2.5-7B-Instruct",
    use_symbolic_verification=True,
    max_reflection_iterations=2
)

# Use anywhere
result = enhance("If I buy 3 items at $12.99 each with 8% tax, what's the total?")
print(result)
```

**Output:**
```
$42.01

Reasoning:
1. Calculate subtotal: 3 × $12.99 = $38.97
2. Calculate tax: $38.97 × 0.08 = $3.12
3. Calculate total: $38.97 + $3.12 = $42.09
```

### Streaming API
```python
from kaelum import enhance_stream

for chunk in enhance_stream("What is the derivative of x^2?"):
    print(chunk, end="", flush=True)
```

---

## 🎯 CLI Commands

```bash
# Start server
kaelum serve --model Qwen/Qwen2.5-7B-Instruct

# Run query
kaelum query "Solve 2x + 6 = 10" --stream

# Run benchmarks
kaelum benchmark --output results.json

# Run tests
kaelum test

# List models
kaelum models

# Check health
kaelum health
```

---

## 🔗 LangChain Integration

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from kaelum import kaelum_enhance_reasoning

# Create Kaelum tool
kaelum_tool = Tool(
    name="kaelum_reasoning",
    func=kaelum_enhance_reasoning,
    description="Use for complex reasoning tasks requiring verification"
)

# Initialize agent
agent = initialize_agent(
    tools=[kaelum_tool],
    llm=OpenAI(temperature=0),
    agent="zero-shot-react-description"
)

# Use it
result = agent.run("Calculate compound interest on $1000 at 5% for 3 years")
```

---

## 🤖 Function Calling (GPT-4, Claude)

```python
from openai import OpenAI
from kaelum import get_function_schema

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "If I save $500/month for 5 years at 4% interest, how much will I have?"}
    ],
    functions=[get_function_schema()],
    function_call="auto"
)

# GPT-4 will call kaelum_enhance_reasoning when needed
```

---

## 📊 Configuration Options

### Basic Setup
```python
set_reasoning_model(
    base_url="http://localhost:8000/v1",  # vLLM endpoint
    model="Qwen/Qwen2.5-7B-Instruct",     # Model name
)
```

### Advanced Setup
```python
set_reasoning_model(
    base_url="http://localhost:8000/v1",
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.7,                       # Sampling temperature
    max_tokens=2048,                       # Max generation length
    max_reflection_iterations=2,           # Self-correction passes
    use_symbolic_verification=True,        # Enable math checking
    use_factual_verification=False,        # Enable RAG checking
    debug_verification=False,              # Debug mode
)
```

### Environment Variables
```bash
# Create .env file
KAELUM_BASE_URL=http://localhost:8000/v1
KAELUM_MODEL=Qwen/Qwen2.5-7B-Instruct
KAELUM_API_KEY=your-key-here
KAELUM_MAX_REFLECTION_ITERATIONS=2
KAELUM_USE_SYMBOLIC_VERIFICATION=true
```

---

## 🧪 Testing

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Module
```bash
pytest tests/test_verification.py -v
```

### Run with Coverage
```bash
pytest tests/ --cov=kaelum --cov-report=html
```

### Run Benchmarks
```bash
python benchmarks/gsm8k_benchmark.py --output results.json
```

---

## 🎓 Example Use Cases

### 1. Math Problem Solving
```python
result = enhance("A train travels 120 miles in 2 hours. What's its average speed?")
# Output: 60 mph with verified calculation
```

### 2. Code Review
```python
result = enhance("""
Review this code for bugs:
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)
""")
# Output: Reasoning about edge cases and potential improvements
```

### 3. Customer Service (with custom prompts)
```python
set_reasoning_model(
    model="Qwen/Qwen2.5-7B-Instruct",
    reasoning_system_prompt="You are a helpful customer service agent.",
    reasoning_user_template="Customer question: {query}\n\nProvide a helpful response:"
)

result = enhance("How do I return a product?")
```

---

## 🐛 Troubleshooting

### Server not starting
```bash
# Check vLLM installation
pip install vllm

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Tests failing
```bash
# Reinstall dependencies
pip install -e ".[dev]"

# Clear cache
rm -rf __pycache__ kaelum/__pycache__
```

### Import errors
```bash
# Verify installation
python -c "import kaelum; print(kaelum.__version__)"

# Check sys.path
python -c "import sys; print(sys.path)"
```

---

## 📚 Next Steps

1. **Read the Architecture**: See `ARCHITECTURE.md` for system design
2. **Deploy to Production**: See `DEPLOYMENT.md` for cloud setup
3. **Contribute**: See `TODO.md` for roadmap
4. **Explore Examples**: Check `examples/` directory

---

## 🆘 Getting Help

- **Documentation**: See `docs/` folder
- **Issues**: Open a GitHub issue
- **Discussions**: Join GitHub Discussions
- **Email**: support@kaelum.ai

---

## 📄 License

MIT License - See LICENSE file

---

**Ready to ship!** 🚀 Kaelum v1.0.0 is production-ready.

For more details, see:
- **Full Guide**: `README.md`
- **Deployment**: `DEPLOYMENT.md`
- **Release Notes**: `RELEASE_v1.0.0.md`
- **Project Status**: `PROJECT_STATUS.md`
