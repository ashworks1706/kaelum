"""KaelumAI - Make any LLM reason better."""

from typing import Dict
from kaelum.core.config import LLMConfig, MCPConfig
from kaelum.runtime.orchestrator import MCP

_mcp_cache: Dict[str, MCP] = {}


def enhance(query: str, model: str = "llama3.2:3b", fast: bool = True):
    """
    Enhance LLM reasoning.
    
    fast=True  -> Direct answer (1-2s)
    fast=False -> Full reasoning trace (8-12s)
    """
    global _mcp_cache
    
    if model not in _mcp_cache:
        config = MCPConfig(llm=LLMConfig(model=model))
        _mcp_cache[model] = MCP(config)
    
    mcp = _mcp_cache[model]
    
    if fast:
        from kaelum.core.reasoning import Message
        return mcp.llm.generate([Message(role="user", content=query)])
    
    result = mcp.infer(query)
    trace = "\n".join(f"{i+1}. {s}" for i, s in enumerate(result["trace"]))
    return f"{result['final']}\n\n{trace}"


__all__ = ["enhance"]
