"""Pytest configuration and shared fixtures."""

import pytest
from unittest.mock import Mock, MagicMock
from kaelum.core.config import KaelumConfig, LLMConfig
from kaelum.core.reasoning import LLMClient


@pytest.fixture
def llm_config():
    """Create a test LLM configuration."""
    return LLMConfig(
        base_url="http://localhost:8000/v1",
        model="test-model",
        api_key="test-key",
        temperature=0.7,
        max_tokens=1024
    )


@pytest.fixture
def kaelum_config(llm_config):
    """Create a test Kaelum configuration."""
    return KaelumConfig(
        reasoning_llm=llm_config,
        max_reflection_iterations=2,
        use_symbolic_verification=True,
        use_factual_verification=False,
        debug_verification=False
    )


@pytest.fixture
def mock_llm_client(llm_config):
    """Create a mock LLM client."""
    client = Mock(spec=LLMClient)
    client.config = llm_config
    
    # Mock generate method
    def mock_generate(messages, stream=False):
        if stream:
            return iter(["Test ", "response ", "chunk"])
        return "Test response"
    
    client.generate = Mock(side_effect=mock_generate)
    return client


@pytest.fixture
def sample_reasoning_trace():
    """Sample reasoning trace for testing."""
    return [
        "Identify the problem: 2x + 6 = 10",
        "Subtract 6 from both sides: 2x = 4",
        "Divide both sides by 2: x = 2",
        "Verify: 2(2) + 6 = 4 + 6 = 10 ✓"
    ]


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "Solve for x: 2x + 6 = 10"
