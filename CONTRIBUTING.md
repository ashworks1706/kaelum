# Developer Guide for KaelumAI

This guide is for developers who want to contribute to or extend KaelumAI.

## Project Structure

```
KaelumAI/
├── kaelum/               # Main package
│   ├── core/             # Core reasoning components
│   │   ├── config.py     # Configuration models
│   │   ├── reasoning.py  # LLM client and reasoning generation
│   │   ├── verification.py # Symbolic and factual verification
│   │   ├── reflection.py # Multi-LLM verification and reflection
│   │   ├── scoring.py    # Confidence scoring
│   │   └── policy.py     # Adaptive policy controller
│   ├── runtime/          # Pipeline orchestration
│   │   └── orchestrator.py # MCP and ModelRuntime
│   └── tools/            # Integration tools
│       └── mcp_tool.py   # LangChain/LangGraph adapters
├── app/                  # FastAPI application
│   └── main.py           # API endpoints
├── mcp/                  # MCP protocol
│   ├── manifest.json     # MCP manifest
│   └── protocol.py       # Protocol handlers
├── tests/                # Test suite
├── examples/             # Example scripts
└── docs/                 # Documentation
```

## Development Setup

### 1. Clone and Install

```bash
git clone https://github.com/ashworks1706/KaelumAI.git
cd KaelumAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### 2. Set Up Environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=kaelum --cov-report=html

# Run specific test file
pytest tests/test_config.py

# Run with verbose output
pytest -v
```

## Architecture Deep Dive

### Core Components

#### 1. LLMClient (kaelum/core/reasoning.py)
- Abstraction layer for OpenAI and Anthropic APIs
- Handles message formatting and API calls
- Extensible to support more providers

#### 2. VerificationEngine (kaelum/core/verification.py)
- **SymbolicVerifier**: Uses SymPy to verify mathematical expressions
- **FactualVerifier**: RAG-based factual claim verification
- Modular design allows adding new verification methods

#### 3. ReflectionEngine (kaelum/core/reflection.py)
- **Verifier**: Independent LLM that critiques reasoning
- **Reflector**: Independent LLM that repairs reasoning
- Iterative improvement through multiple cycles

#### 4. ConfidenceScorer (kaelum/core/scoring.py)
- Weighted scoring from multiple sources
- Aggregates symbolic, factual, and LLM verification
- Provides trace-level and step-level confidence

#### 5. PolicyController (kaelum/core/policy.py)
- Adapts verification depth based on query complexity
- Learns from historical performance
- Optimizes latency-accuracy tradeoff

### Request Flow

```
User Query
    ↓
MCP.infer()
    ↓
1. ReasoningGenerator.generate_reasoning() → Initial trace
    ↓
2. VerificationEngine.verify_trace() → Symbolic + Factual checks
    ↓
3. ReflectionEngine.reflect_and_repair() → Multi-LLM verification
    ↓
4. ConfidenceScorer.compute_confidence() → Confidence score
    ↓
5. ReasoningGenerator.generate_answer() → Final answer
    ↓
ReasoningResult
```

## Adding New Features

### Adding a New LLM Provider

1. Extend `LLMClient` in `kaelum/core/reasoning.py`:

```python
def _generate_custom(self, messages: List[Message]) -> str:
    """Generate response using custom provider."""
    # Implement your provider logic
    pass
```

2. Update `LLMConfig` to support the new provider:

```python
provider: Literal["openai", "anthropic", "custom"] = Field(...)
```

### Adding a New Verification Method

1. Create a new verifier class in `kaelum/core/verification.py`:

```python
class CustomVerifier:
    def verify_step(self, step: str) -> Tuple[bool, Optional[str]]:
        # Implement verification logic
        pass
```

2. Integrate into `VerificationEngine`:

```python
self.custom_verifier = CustomVerifier() if use_custom else None
```

### Adding a New API Endpoint

1. Add endpoint in `app/main.py`:

```python
@app.post("/custom_endpoint")
async def custom_endpoint(request: CustomRequest):
    # Implement endpoint logic
    pass
```

2. Add request/response models:

```python
class CustomRequest(BaseModel):
    field: str = Field(description="...")
```

## Code Style

### Formatting
- Use `black` for code formatting: `black .`
- Line length: 100 characters
- Follow PEP 8

### Type Hints
- Always use type hints for function parameters and returns
- Use `Optional[T]` for nullable types
- Use `List[T]`, `Dict[K, V]` for collections

### Docstrings
- Use Google-style docstrings
- Document all public functions and classes
- Include Args, Returns, and Raises sections

Example:
```python
def verify_reasoning(query: str, context: Optional[str] = None) -> ReasoningResult:
    """
    Verify reasoning for a query.

    Args:
        query: The reasoning query to verify
        context: Optional context for the query

    Returns:
        ReasoningResult with verification details

    Raises:
        ValueError: If query is empty
    """
    pass
```

## Testing Guidelines

### Unit Tests
- Test each component in isolation
- Mock external dependencies (LLM APIs)
- Aim for >80% code coverage

### Integration Tests
- Test end-to-end workflows
- Can use real API calls (mark with `@pytest.mark.integration`)
- Test error handling and edge cases

### Test Naming
- Use descriptive test names: `test_<component>_<scenario>`
- Group related tests in classes
- Use fixtures for common setup

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspect Reasoning Traces

```python
mcp = MCP(MCPConfig(log_traces=True))
result = mcp.infer("query")
traces = mcp.get_traces()
print(traces)
```

### Profile Performance

```python
import cProfile
cProfile.run('mcp.infer("query")')
```

## Contributing

### 1. Fork and Branch

```bash
git checkout -b feature/my-feature
```

### 2. Make Changes

- Write code
- Add tests
- Update documentation

### 3. Run Quality Checks

```bash
# Format code
black .

# Run linter
ruff check .

# Run tests
pytest

# Type check
mypy kaelum/
```

### 4. Submit Pull Request

- Clear description of changes
- Reference any related issues
- Ensure CI passes

## Performance Optimization

### Caching
- Cache frequent queries using Redis
- Cache embeddings for RAG verification
- Implement request deduplication

### Batching
- Batch multiple verification requests
- Use async/await for parallel LLM calls
- Implement connection pooling

### Model Selection
- Use faster models for verification (gpt-3.5-turbo)
- Consider model quantization for local deployment
- Implement model fallback for reliability

## Security Considerations

### API Keys
- Never commit API keys
- Use environment variables or secrets management
- Rotate keys regularly

### Input Validation
- Validate all user inputs
- Sanitize queries before LLM calls
- Implement rate limiting

### Output Safety
- Filter potentially harmful outputs
- Implement content moderation
- Log and monitor for abuse

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure package is installed
pip install -e .
# Or add to PYTHONPATH
export PYTHONPATH=/path/to/KaelumAI:$PYTHONPATH
```

**API Key Errors**
```bash
# Check environment variables
echo $OPENAI_API_KEY
# Or set in code
os.environ["OPENAI_API_KEY"] = "..."
```

**Memory Issues**
- Reduce max_tokens in LLMConfig
- Limit max_reflection_iterations
- Implement result streaming

## Resources

- **OpenAI API Docs**: https://platform.openai.com/docs
- **Anthropic API Docs**: https://docs.anthropic.com
- **FastAPI Docs**: https://fastapi.tiangolo.com
- **Pydantic Docs**: https://docs.pydantic.dev

## Support

- GitHub Issues: https://github.com/ashworks1706/KaelumAI/issues
- Email: ashworks1706@gmail.com
