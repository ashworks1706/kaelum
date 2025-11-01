"""KaelumAI - Make any LLM reason better."""

from typing import Optional
from kaelum.core.config import LLMConfig, MCPConfig
from kaelum.runtime.orchestrator import MCP

# YOUR internal reasoning model config (set once at startup)
_REASONING_MODEL_CONFIG: Optional[LLMConfig] = None
_mcp: Optional[MCP] = None


def set_reasoning_model(
    provider: str = "ollama",
    model: str = "llama3.2:3b", 
    base_url: Optional[str] = None,
):
    """
    Set YOUR reasoning model (internal, runs on your infrastructure).
    Call this once at startup before using enhance().
    
    Args:
        provider: "ollama", "vllm", or "custom"
        model: Your model name
        base_url: Your model endpoint (optional, has defaults)
    """
    global _REASONING_MODEL_CONFIG, _mcp
    
    _REASONING_MODEL_CONFIG = LLMConfig(
        provider=provider,
        model=model,
        base_url=base_url,
    )
    
    # Initialize MCP with your reasoning model
    config = MCPConfig(reasoning_llm=_REASONING_MODEL_CONFIG)
    _mcp = MCP(config)


def enhance(query: str, fast: bool = True) -> str:
    """
    Enhance reasoning for a query using YOUR reasoning model.
    
    Args:
        query: User's question
        fast: True = direct answer (1-2s), False = full reasoning trace (8-12s)
    
    Returns:
        Enhanced answer
    """
    global _mcp
    
    # Auto-initialize with default if not configured
    if _mcp is None:
        set_reasoning_model()
    
    if fast:
        from kaelum.core.reasoning import Message
        return _mcp.reasoning_llm.generate([Message(role="user", content=query)])
    
    # Full reasoning mode
    result = _mcp.infer(query)
    trace = "\n".join(f"{i+1}. {s}" for i, s in enumerate(result["trace"]))
    return f"{result['final']}\n\nReasoning:\n{trace}"


__all__ = ["enhance", "set_reasoning_model"]
