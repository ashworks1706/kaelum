"""Tests for core configuration module."""

import pytest
from kaelum.core.config import LLMConfig, MCPConfig


def test_llm_config_defaults():
    """Test LLMConfig default values."""
    config = LLMConfig()
    assert config.model == "gemini-1.5-flash"
    assert config.temperature == 0.7
    assert config.max_tokens == 2048
    assert config.provider == "gemini"


def test_llm_config_custom():
    """Test LLMConfig with custom values."""
    config = LLMConfig(
        model="gemini-1.5-pro", temperature=0.5, max_tokens=1024, provider="gemini"
    )
    assert config.model == "gemini-1.5-pro"
    assert config.temperature == 0.5
    assert config.max_tokens == 1024
    assert config.provider == "gemini"


def test_mcp_config_defaults():
    """Test MCPConfig default values and initialization."""
    config = MCPConfig()

    # Check defaults
    assert config.use_symbolic is True
    assert config.use_rag is False
    assert config.confidence_threshold == 0.7
    assert config.max_reflection_iterations == 2
    assert config.enable_policy_controller is True
    assert config.log_traces is True

    # Check that LLM configs are initialized
    assert config.llm is not None
    assert config.llm.model == "gemini-1.5-flash"
    assert config.verifier_llm is not None
    assert config.verifier_llm.model == "gemini-1.5-flash"
    assert config.reflector_llm is not None
    assert config.reflector_llm.model == "gemini-1.5-flash"


def test_mcp_config_custom_llm():
    """Test MCPConfig with custom LLM configurations."""
    llm_config = LLMConfig(model="gemini-1.5-pro", temperature=0.8)
    config = MCPConfig(llm=llm_config)

    assert config.llm.model == "gemini-1.5-pro"
    assert config.llm.temperature == 0.8


def test_mcp_config_validation():
    """Test MCPConfig validation."""
    # Test valid confidence threshold
    config = MCPConfig(confidence_threshold=0.5)
    assert config.confidence_threshold == 0.5

    # Test boundary values
    config = MCPConfig(confidence_threshold=0.0)
    assert config.confidence_threshold == 0.0

    config = MCPConfig(confidence_threshold=1.0)
    assert config.confidence_threshold == 1.0


def test_mcp_config_iterations():
    """Test MCPConfig max_reflection_iterations validation."""
    config = MCPConfig(max_reflection_iterations=3)
    assert config.max_reflection_iterations == 3

    # Test boundaries
    config = MCPConfig(max_reflection_iterations=1)
    assert config.max_reflection_iterations == 1

    config = MCPConfig(max_reflection_iterations=5)
    assert config.max_reflection_iterations == 5
