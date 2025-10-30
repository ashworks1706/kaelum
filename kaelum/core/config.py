"""Core configuration models for KaelumAI."""

from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for KaelumAI reasoning acceleration."""

from typing import List, Literal, Optional

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


class ReasoningConfig(BaseModel):
    """Configuration for reasoning enhancement."""

    # Single LLM config (can use different temps for different roles)
    llm: Optional[LLMConfig] = Field(default=None, description="Main LLM configuration")
    
    # Fallback models for smart routing
    fallback_models: List[Tuple[str, float]] = Field(
        default_factory=list,
        description="List of (model_name, cost) for routing. Tries cheaper first."
    )
    
    # Enhancement features
    use_symbolic: bool = Field(default=True, description="Enable symbolic verification")
    use_reflection: bool = Field(default=True, description="Enable self-reflection")
    use_caching: bool = Field(default=True, description="Enable response caching")
    use_routing: bool = Field(default=False, description="Enable smart model routing")
    
    # Quality thresholds
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0, description="Min confidence")
    max_reflection_iterations: int = Field(default=2, ge=1, le=5, description="Max reflection cycles")
    
    # Performance
    enable_parallel: bool = Field(default=True, description="Parallel verification")
    log_traces: bool = Field(default=True, description="Log reasoning traces")

    def model_post_init(self, __context) -> None:
        """Initialize default config if not provided."""
        if self.llm is None:
            self.llm = LLMConfig(model="qwen2.5:7b", provider="ollama")
