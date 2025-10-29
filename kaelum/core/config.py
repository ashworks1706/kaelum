"""Core configuration models for KaelumAI."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for an LLM client."""

    model: str = Field(default="gemini-1.5-flash", description="LLM model identifier")
    api_key: Optional[str] = Field(default=None, description="API key for the LLM provider")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=2048, gt=0, description="Maximum tokens to generate")
    provider: Literal["gemini"] = Field(
        default="gemini", description="LLM provider"
    )


class MCPConfig(BaseModel):
    """Configuration for the MCP reasoning layer."""

    llm: Optional[LLMConfig] = Field(
        default=None, description="Main LLM configuration (defaults to gemini-1.5-flash)"
    )
    verifier_llm: Optional[LLMConfig] = Field(
        default=None, description="Verifier LLM configuration (defaults to gemini-1.5-flash)"
    )
    reflector_llm: Optional[LLMConfig] = Field(
        default=None, description="Reflector LLM configuration (defaults to gemini-1.5-flash)"
    )
    use_symbolic: bool = Field(default=True, description="Enable symbolic verification (SymPy)")
    use_rag: bool = Field(default=False, description="Enable RAG-based factual verification")
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum confidence for accepting reasoning"
    )
    max_reflection_iterations: int = Field(
        default=2, ge=1, le=5, description="Maximum reflection/repair iterations"
    )
    enable_policy_controller: bool = Field(
        default=True, description="Enable adaptive policy controller"
    )
    log_traces: bool = Field(default=True, description="Enable trace logging")

    def model_post_init(self, __context) -> None:
        """Initialize default configs if not provided."""
        if self.llm is None:
            self.llm = LLMConfig(model="gemini-1.5-flash")
        if self.verifier_llm is None:
            self.verifier_llm = LLMConfig(model="gemini-1.5-flash", temperature=0.3)
        if self.reflector_llm is None:
            self.reflector_llm = LLMConfig(model="gemini-1.5-flash", temperature=0.5)
