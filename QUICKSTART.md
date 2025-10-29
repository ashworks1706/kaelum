# KaelumAI Quickstart Guide

Get started with KaelumAI in 5 minutes.

## Installation

```bash
# Clone the repository
git clone https://github.com/ashworks1706/KaelumAI.git
cd KaelumAI

# Install dependencies
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY="your-api-key-here"
```

## Basic Usage

### 1. Simple Reasoning

```python
from kaelum import MCP, MCPConfig

# Create MCP instance
mcp = MCP(MCPConfig())

# Run reasoning verification
result = mcp.infer("If 3x + 5 = 11, what is x?")

print(f"Answer: {result.final}")
print(f"Verified: {result.verified}")
print(f"Confidence: {result.confidence:.2f}")
```

### 2. Using ModelRuntime

```python
from kaelum import ModelRuntime, LLMClient, LLMConfig, ReasoningMCPTool, MCPConfig

# Create components
llm = LLMClient(LLMConfig(model="gpt-4o-mini"))
mcp_tool = ReasoningMCPTool(MCPConfig())

# Create runtime with MCP tool
runtime = ModelRuntime(llm).attach(mcp_tool)

# Generate verified content
response = runtime.generate_content(
    "Explain how reinforcement learning optimizes retrieval in RAG systems."
)

print(response)
```

### 3. Custom Configuration

```python
from kaelum import MCP, MCPConfig, LLMConfig

# Configure with custom settings
config = MCPConfig(
    llm=LLMConfig(model="gpt-4o", temperature=0.7),
    verifier_llm=LLMConfig(model="gpt-3.5-turbo", temperature=0.3),
    use_symbolic=True,
    confidence_threshold=0.8,
    max_reflection_iterations=3,
)

mcp = MCP(config)
result = mcp.infer("What is the square root of 144?")
```

## Running the API Server

### Start the server

```bash
uvicorn app.main:app --reload
```

### Test endpoints

```bash
# Health check
curl http://localhost:8000/health

# Verify reasoning
curl -X POST http://localhost:8000/verify_reasoning \
  -H "Content-Type: application/json" \
  -d '{"query": "If 2x + 3 = 11, what is x?"}'

# Get metrics
curl http://localhost:8000/metrics
```

## Docker Deployment

### Using Docker

```bash
# Build image
docker build -t kaelum .

# Run container
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  kaelum
```

### Using Docker Compose

```bash
# Create .env file
cp .env.example .env
# Edit .env with your API keys

# Start service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop service
docker-compose down
```

## LangChain Integration

```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from kaelum import MCPConfig
from kaelum.tools.mcp_tool import LangChainAdapter

# Create base LLM
llm = ChatOpenAI(model="gpt-4o")

# Create reasoning tool
reasoning_tool = LangChainAdapter.create_tool(MCPConfig())

# Initialize agent
agent = initialize_agent(
    [reasoning_tool],
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run with reasoning verification
response = agent.run("Calculate the compound interest on $1000 at 5% for 3 years")
```

## Configuration Options

### LLMConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"gpt-4o-mini"` | LLM model to use |
| `temperature` | `0.7` | Sampling temperature |
| `max_tokens` | `2048` | Maximum tokens to generate |
| `provider` | `"openai"` | Provider: "openai" or "anthropic" |

### MCPConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_symbolic` | `True` | Enable symbolic verification |
| `use_rag` | `False` | Enable RAG-based factual verification |
| `confidence_threshold` | `0.7` | Minimum confidence for verification |
| `max_reflection_iterations` | `2` | Maximum reflection cycles |
| `enable_policy_controller` | `True` | Enable adaptive policy |
| `log_traces` | `True` | Enable trace logging |

## Common Use Cases

### 1. Mathematical Problem Solving

```python
result = mcp.infer("Solve: 2x^2 + 5x - 3 = 0")
# Symbolic verification ensures correctness
```

### 2. Logical Reasoning

```python
result = mcp.infer(
    "If all mammals are warm-blooded and whales are mammals, "
    "what can we conclude about whales?"
)
# Multi-LLM verification checks logical consistency
```

### 3. Complex Analysis

```python
result = mcp.infer(
    "Analyze the time complexity of quicksort and explain "
    "when it performs best and worst."
)
# Reflection improves reasoning quality
```

## Monitoring and Metrics

```python
# Get metrics after multiple queries
metrics = mcp.get_metrics()

print(f"Total requests: {metrics['total_requests']}")
print(f"Verification rate: {metrics['verification_rate']:.1%}")
print(f"Avg confidence: {metrics['avg_confidence']:.2f}")
print(f"Avg iterations: {metrics['avg_iterations']:.1f}")
```

## Best Practices

### 1. Choose the Right Model
- Use `gpt-4o` or `gpt-4o-mini` for main reasoning
- Use `gpt-3.5-turbo` for verifier/reflector (cost-effective)
- Use `claude-3-haiku` for fast reflection

### 2. Adjust Confidence Threshold
- High stakes: `0.85` or higher
- General use: `0.7` (default)
- Exploratory: `0.5` or lower

### 3. Optimize Iterations
- Simple queries: `1-2` iterations
- Complex reasoning: `2-3` iterations
- Critical tasks: `3-5` iterations

### 4. Enable Appropriate Verification
- Math/logic problems: Enable symbolic verification
- Factual claims: Enable RAG verification
- General reasoning: Use multi-LLM verification

## Troubleshooting

### API Key Issues
```python
import os
os.environ["OPENAI_API_KEY"] = "your-key"
# Or pass directly
config = LLMConfig(model="gpt-4o", api_key="your-key")
```

### Low Confidence Scores
- Increase `max_reflection_iterations`
- Use stronger verifier model
- Enable symbolic verification for math

### High Latency
- Reduce `max_reflection_iterations`
- Use faster models (gpt-3.5-turbo)
- Disable unnecessary verification methods

### Memory Issues
- Reduce `max_tokens`
- Limit trace logging
- Process queries in batches

## Next Steps

1. **Explore Examples**: Check `examples/` directory for more use cases
2. **Read Documentation**: See `CONTRIBUTING.md` for architecture details
3. **Deploy to Production**: Follow `DEPLOYMENT.md` for cloud deployment
4. **Run Tests**: Execute `pytest` to verify installation
5. **Customize**: Extend with your own verification methods

## Support

- **Documentation**: Full docs in repository
- **Issues**: https://github.com/ashworks1706/KaelumAI/issues
- **Email**: ashworks1706@gmail.com

## License

MIT License - see LICENSE file for details
