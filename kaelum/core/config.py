"""Core configuration models for KaelumAI."""

from typing import Optional
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    
    provider: str = Field(
        default="ollama",
        description="LLM provider: ollama, vllm, or custom"
    )
    model: str = Field(
        default="llama3.2:3b",
        description="Model name"
    )
    base_url: Optional[str] = Field(
        default="http://localhost:11434/v1",
        description="API base URL"
    )
    api_key: Optional[str] = Field(
        default="ollama",
        description="API key (not needed for local)"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=128000)


class MCPConfig(BaseModel):
    """KaelumAI configuration."""
    
    # YOUR reasoning LLM
    reasoning_llm: LLMConfig = Field(default_factory=LLMConfig)
    
    # Reflection settings
    max_reflection_iterations: int = Field(default=2, ge=0, le=5)
    
    # Verification settings
    use_symbolic_verification: bool = Field(default=True)
    use_factual_verification: bool = Field(default=False)
    
    class Config:
        arbitrary_types_allowed = True

