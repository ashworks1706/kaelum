"""KaelumAI - One line to make any LLM reason better."""

__version__ = "0.2.0"

import os
from typing import Iterator, Literal, Optional, Union

from kaelum.core.config import LLMConfig, MCPConfig
from kaelum.runtime.orchestrator import MCP

# Cache for faster repeated calls
_default_mcp: Optional[MCP] = None


def enhance(
    query: str,
    model: str = "llama3.2:3b",
    mode: Literal["auto", "math", "logic"] = "auto",
    max_iterations: int = 1,
    temperature: float = 0.3,
    max_tokens: int = 512,
    stream: bool = False,
) -> Union[str, Iterator[str]]:
    """
    Make any LLM reason better - simplified!
    
    Args:
        query: Your question
        model: Model name (default: llama3.2:3b)
        mode: auto/math/logic
        max_iterations: Reflection cycles (1-2)
        temperature: 0.1=fast, 0.7=creative
        max_tokens: Response length
        stream: Stream output (beta)
    
    Returns:
        Enhanced answer as string
    
    Examples:
        >>> enhance("What is 15% of 240?", model="llama3.2:3b")
        >>> enhance("Solve: 2x + 5 = 13", mode="math", model="qwen2.5:7b")
    """
    global _default_mcp
    
    # Create new MCP or reuse cached one
    if _default_mcp is None:
        config = MCPConfig(
            llm=LLMConfig(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            max_reflection_iterations=max_iterations,
        )
        _default_mcp = MCP(config)
    
    # Apply mode template
    enhanced_query = _apply_mode(query, mode)
    
    # Run inference
    if stream:
        return _stream_reasoning(_default_mcp, enhanced_query)
    
    result = _default_mcp.infer(enhanced_query)
    return _format_result(result)


def _apply_mode(query: str, mode: str) -> str:
    """Apply mode template."""
    if mode == "math":
        return f"Solve step-by-step:\n{query}"
    elif mode == "logic":
        return f"Think logically:\n{query}"
    return query


def _format_result(result: dict) -> str:
    """Format result."""
    trace = "\n".join(f"{i+1}. {step}" for i, step in enumerate(result["trace"]))
    answer = result['final']
    confidence = result.get("diagnostics", {}).get("confidence", 0)
    
    output = f"{answer}\n\n💭 Reasoning:\n{trace}"
    
    if confidence >= 0.8:
        output += f"\n\n✅ {confidence:.0%}"
    
    return output


def _stream_reasoning(mcp: MCP, query: str) -> Iterator[str]:
    """Stream output."""
    trace = mcp.generator.generate_reasoning(query)
    
    yield "💭 Thinking:\n"
    for i, step in enumerate(trace, 1):
        yield f"{i}. {step}\n"
    
    yield "\n✅ Done\n"


# Export
__all__ = ["enhance", "MCP", "LLMConfig", "MCPConfig"]
