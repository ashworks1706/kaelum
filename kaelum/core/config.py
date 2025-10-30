"""Core configuration models for KaelumAI."""

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    
    provider: Literal["openai", "ollama", "vllm", "openrouter"] = Field(
        default="ollama",
        description="LLM provider (ollama for local, openai/vllm for API)"
    )
    model: str = Field(
        default="llama3.2:3b",
        description="Model name (e.g., llama3.2:3b, gpt-4o-mini, qwen2.5:7b)"
    )
    base_url: Optional[str] = Field(
        default="http://localhost:11434/v1",
        description="API base URL (None for OpenAI default)"
    )
    api_key: Optional[str] = Field(
        default="ollama",
        description="API key (not needed for local Ollama)"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=128000)


class MCPConfig(BaseModel):
    """Main KaelumAI configuration."""
    
    # Single LLM for all reasoning (cost-efficient)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    
    # Reflection settings
    max_reflection_iterations: int = Field(default=2, ge=1, le=5)
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    
    # Verification settings
    use_symbolic_verification: bool = Field(
        default=True,
        description="Enable SymPy-based math verification"
    )
    use_factual_verification: bool = Field(
        default=False,
        description="Enable factual verification (requires RAG adapter)"
    )
    
    # RAG adapter (not serialized by Pydantic)
    class Config:
        arbitrary_types_allowed = True

