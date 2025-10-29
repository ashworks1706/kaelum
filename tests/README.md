# KaelumAI Tests

This directory contains the test suite for KaelumAI.

## Running Tests

```bash
# Install test dependencies
pip install -r ../requirements.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=kaelum --cov-report=html

# Run specific test file
pytest test_config.py

# Run with verbose output
pytest -v

# Run specific test
pytest test_config.py::test_llm_config_defaults
```

## Test Structure

- `conftest.py` - Pytest configuration and shared fixtures
- `test_config.py` - Tests for configuration module
- `test_verification.py` - Tests for verification engine
- `test_scoring.py` - Tests for confidence scoring
- `test_policy.py` - Tests for policy controller
- `test_mcp_tool.py` - Tests for MCP tool integration

## Test Coverage

The test suite aims for high coverage of:
- Configuration and validation
- Verification logic (symbolic and factual)
- Confidence scoring algorithms
- Policy adaptation
- Tool interfaces

## Notes

- Most tests are unit tests that don't require actual LLM API calls
- Integration tests that require API keys are marked and can be skipped
- Use `pytest -m "not integration"` to skip integration tests
