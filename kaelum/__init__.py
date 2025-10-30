"""KaelumAI - One line to make any LLM reason better."""

__version__ = "0.2.0"

import os
from typing import Iterator, Literal, Optional, Union

from kaelum.core.config import LLMConfig, MCPConfig
from kaelum.core.reasoning import LLMClient
from kaelum.runtime.orchestrator import MCP

# Simple one-line API
_default_mcp: Optional[MCP] = None


def enhance(
    query: str,
    mode: Literal["auto", "math", "code", "logic", "creative"] = "auto",
    model: Optional[str] = None,
    stream: bool = False,
    max_iterations: int = 1,
    cache: bool = True,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Union[str, Iterator[str]]:
    """
    Make any LLM reason better. One function.
    
    Args:
        query: Your question or problem
        mode: Reasoning mode (auto/math/code/logic/creative)
        model: Model name (auto-detects Ollama if None)
        stream: Stream reasoning steps in real-time
        max_iterations: Max reflection cycles (1-3)
        cache: Use caching for faster repeated queries
        api_base: Custom API endpoint
        api_key: API key (not needed for Ollama)
    
    Returns:
        Enhanced reasoning as string (or iterator if stream=True)
    
    Examples:
        >>> from kaelum import enhance
        >>> print(enhance("What is 15% of 240?"))
        >>> 
        >>> # With streaming
        >>> for step in enhance("Solve x^2 + 5x + 6 = 0", stream=True):
        ...     print(step)
    """
    global _default_mcp
    
    # Auto-detect Ollama if no model specified
    if model is None:
        model = _detect_model()
    
    # Auto-detect API base
    if base_url is None:
        base_url = _detect_api_base()
    
    # Auto-detect API key
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY", "ollama")
    
    # Create or reuse MCP instance
    if _default_mcp is None or not cache:
        config = MCPConfig(
            llm=LLMConfig(
                provider="ollama" if "localhost" in (base_url or "") else "openai",
                model=model,
                base_url=base_url,
                api_key=api_key,
            ),
            max_reflection_iterations=max_iterations,
        )
        if cache and _default_mcp is None:
            _default_mcp = MCP(config)
        else:
            mcp = MCP(config)
    else:
        mcp = _default_mcp
    
    # Enhance query with mode-specific template
    enhanced_query = _apply_mode_template(query, mode)
    
    # Run inference
    if stream:
        return _stream_reasoning(mcp or _default_mcp, enhanced_query)
    else:
        result = (mcp or _default_mcp).infer(enhanced_query, use_cache=cache)
        return _format_result(result)


def _detect_model() -> str:
    """Auto-detect best available model."""
    # Try Ollama first
    try:
        import subprocess
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and result.stdout:
            # Prefer these models in order
            for model in ["qwen2.5:7b", "llama3.2:3b", "llama3:8b", "mistral:7b"]:
                if model in result.stdout:
                    return model
            # Use first available model
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            if lines:
                return lines[0].split()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Default to common model
    return "qwen2.5:7b"


def _detect_api_base() -> str:
    """Auto-detect API base URL."""
    # Check environment
    if base := os.getenv("OPENAI_API_BASE"):
        return base
    
    # Try Ollama
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=1)
        if response.status_code == 200:
            return "http://localhost:11434/v1"
    except:
        pass
    
    # Default to Ollama
    return "http://localhost:11434/v1"


def _apply_mode_template(query: str, mode: str) -> str:
    """Apply mode-specific reasoning template."""
    templates = {
        "math": f"Solve this step-by-step, showing all mathematical work:\n{query}",
        "code": f"Analyze this code problem systematically:\n{query}",
        "logic": f"Think through this logically, examining each premise:\n{query}",
        "creative": f"Explore this creatively with multiple perspectives:\n{query}",
        "auto": query,
    }
    return templates.get(mode, query)


def _format_result(result: dict) -> str:
    """Format result for clean output."""
    trace_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(result["trace"]))
    
    output = f"💭 Reasoning:\n{trace_text}\n"
    
    if result.get("confidence", 0) >= 0.8:
        output += f"\n✅ Confidence: {result['confidence']:.0%}"
    else:
        output += f"\n⚠️  Confidence: {result['confidence']:.0%}"
    
    if result.get("cache_hit"):
        output += " (cached)"
    
    return output


def _stream_reasoning(mcp: MCP, query: str) -> Iterator[str]:
    """Stream reasoning steps in real-time."""
    # Generate initial reasoning
    initial = mcp.generator.generate_reasoning(query)
    
    yield "💭 Initial reasoning:\n"
    for i, step in enumerate(initial["trace"], 1):
        yield f"{i}. {step}\n"
    
    # Reflection
    if mcp.config.max_reflection_iterations > 0:
        yield "\n🔄 Reflecting...\n"
        reflection = mcp.reflection.enhance_reasoning(query, initial["trace"])
        
        if reflection["improved"]:
            yield "\n✨ Improved reasoning:\n"
            for i, step in enumerate(reflection["final_trace"], 1):
                yield f"{i}. {step}\n"
        else:
            yield "✅ No improvements needed\n"
        
        yield f"\n🎯 Confidence: {reflection['final_verification']['confidence']:.0%}\n"


# Export simple API
__all__ = ["enhance", "MCP", "LLMConfig", "MCPConfig"]
