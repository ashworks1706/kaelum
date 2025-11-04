from typing import Optional
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    base_url: str = Field(default="http://localhost:8000/v1")
    model: str = Field(default="Qwen/Qwen2.5-3B-Instruct")
    api_key: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=128000)


class KaelumConfig(BaseModel):
    reasoning_llm: LLMConfig = Field(default_factory=LLMConfig)
    max_reflection_iterations: int = Field(default=2, ge=0, le=5)
    use_symbolic_verification: bool = Field(default=True)
    use_factual_verification: bool = Field(default=False)
    debug_verification: bool = Field(default=False)
    
    class Config:
        arbitrary_types_allowed = True

