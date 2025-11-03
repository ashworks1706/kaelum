"""Core configuration models for KaelumAI."""

from typing import Optional
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM configuration for any OpenAI-compatible endpoint."""
    
    base_url: str = Field(
        default="http://localhost:11434/v1",
        description="OpenAI-compatible API endpoint"
    )
    model: str = Field(
        default="qwen2.5:7b",
        description="Model name"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key (optional for local servers)"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=128000)


class KaelumConfig(BaseModel):
    """KaelumAI orchestrator configuration."""
    
    reasoning_llm: LLMConfig = Field(default_factory=LLMConfig)
    max_reflection_iterations: int = Field(default=2, ge=0, le=5)
    use_symbolic_verification: bool = Field(default=True)
    use_factual_verification: bool = Field(default=False)
    debug_verification: bool = Field(default=False, description="Enable debug logging for verification")
    
    class Config:
        arbitrary_types_allowed = True

