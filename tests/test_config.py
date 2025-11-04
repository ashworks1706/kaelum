"""Tests for configuration management."""

import pytest
from kaelum.core.config import LLMConfig, KaelumConfig


class TestLLMConfig:
    """Test LLM configuration."""
    
    def test_default_config(self):
        """Test default LLM configuration."""
        config = LLMConfig()
        assert config.base_url == "http://localhost:11434/v1"
        assert config.model == "qwen2.5:7b"
        assert config.api_key is None
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
    
    def test_custom_config(self):
        """Test custom LLM configuration."""
        config = LLMConfig(
            base_url="http://custom:8000/v1",
            model="custom-model",
            api_key="test-key",
            temperature=0.5,
            max_tokens=4096
        )
        assert config.base_url == "http://custom:8000/v1"
        assert config.model == "custom-model"
        assert config.api_key == "test-key"
        assert config.temperature == 0.5
        assert config.max_tokens == 4096
    
    def test_temperature_validation(self):
        """Test temperature validation."""
        # Valid temperatures
        LLMConfig(temperature=0.0)
        LLMConfig(temperature=1.0)
        LLMConfig(temperature=2.0)
        
        # Invalid temperatures should raise ValueError
        with pytest.raises(ValueError):
            LLMConfig(temperature=-0.1)
        with pytest.raises(ValueError):
            LLMConfig(temperature=2.1)
    
    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        # Valid max_tokens
        LLMConfig(max_tokens=1)
        LLMConfig(max_tokens=128000)
        
        # Invalid max_tokens should raise ValueError
        with pytest.raises(ValueError):
            LLMConfig(max_tokens=0)


class TestKaelumConfig:
    """Test Kaelum orchestrator configuration."""
    
    def test_default_config(self):
        """Test default Kaelum configuration."""
        config = KaelumConfig()
        assert isinstance(config.reasoning_llm, LLMConfig)
        assert config.max_reflection_iterations == 2
        assert config.use_symbolic_verification is True
        assert config.use_factual_verification is False
        assert config.debug_verification is False
    
    def test_custom_config(self):
        """Test custom Kaelum configuration."""
        llm_config = LLMConfig(model="custom-model")
        config = KaelumConfig(
            reasoning_llm=llm_config,
            max_reflection_iterations=3,
            use_symbolic_verification=False,
            use_factual_verification=True,
            debug_verification=True
        )
        assert config.reasoning_llm.model == "custom-model"
        assert config.max_reflection_iterations == 3
        assert config.use_symbolic_verification is False
        assert config.use_factual_verification is True
        assert config.debug_verification is True
    
    def test_reflection_iterations_validation(self):
        """Test reflection iterations validation."""
        # Valid iterations
        KaelumConfig(max_reflection_iterations=0)
        KaelumConfig(max_reflection_iterations=5)
        
        # Invalid iterations should raise ValueError
        with pytest.raises(ValueError):
            KaelumConfig(max_reflection_iterations=-1)
        with pytest.raises(ValueError):
            KaelumConfig(max_reflection_iterations=6)
