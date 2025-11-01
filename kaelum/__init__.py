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
    rag_adapter = None,
    use_factual_verification: bool = False,
    temperature: float = 0.3,  # Lower = faster & more deterministic
    max_tokens: int = 512,  # Reduced for speed
) -> Union[str, Iterator[str]]:
    """
    Make any LLM reason better. One function.
    
    Args:
        query: Your question or problem
        mode: Reasoning mode (auto/math/code/logic/creative)
        model: Model name (auto-detects Ollama if None)
        stream: Stream reasoning steps in real-time
        max_iterations: Max reflection cycles (1-3, default=1 for speed)
        cache: Use caching for faster repeated queries
        base_url: Custom API endpoint
        api_key: API key (not needed for Ollama)
        rag_adapter: Optional RAG adapter for factual verification
        use_factual_verification: Enable factual verification with RAG
        temperature: LLM temperature (0.0-1.0, lower=faster/focused)
        max_tokens: Max response length (lower=faster)
    
    Returns:
        Enhanced reasoning as string (or iterator if stream=True)
    
    Examples:
        >>> from kaelum import enhance
        >>> # Fast mode (default)
        >>> print(enhance("What is 15% of 240?"))
        >>> 
        >>> # Quality mode
        >>> print(enhance("Explain relativity", model="qwen2.5:7b", 
        ...               max_iterations=2, max_tokens=1024))
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
    mcp = None
    if _default_mcp is None or not cache:
        config = MCPConfig(
            llm=LLMConfig(
                provider="ollama" if "localhost" in (base_url or "") else "openai",
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            max_reflection_iterations=max_iterations,
            use_factual_verification=use_factual_verification,
        )
        if cache and _default_mcp is None:
            _default_mcp = MCP(config, rag_adapter=rag_adapter)
            mcp = _default_mcp
        else:
            mcp = MCP(config, rag_adapter=rag_adapter)
    else:
        mcp = _default_mcp
    
    # Enhance query with mode-specific template
    enhanced_query = _apply_mode_template(query, mode)
    
    # Run inference
    if stream:
        return _stream_reasoning(mcp, enhanced_query)
    else:
        result = mcp.infer(enhanced_query)
        return _format_result(result)


def _detect_model() -> str:
    """Auto-detect best available model. Fails if Ollama not found."""
    import subprocess
    
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        timeout=5
    )
    
    if result.returncode != 0:
        raise RuntimeError(
            "Ollama not found or not running. Install Ollama from https://ollama.ai "
            "or explicitly specify model and base_url in LLMConfig."
        )
    
    if not result.stdout.strip():
        raise RuntimeError(
            "No Ollama models found. Run 'ollama pull qwen2.5:7b' or specify model explicitly."
        )
    
    # Prefer these models in order
    for model in ["qwen2.5:7b", "llama3.2:3b", "llama3:8b", "mistral:7b"]:
        if model in result.stdout:
            return model
    
    # Use first available model
    lines = result.stdout.strip().split("\n")[1:]  # Skip header
    if lines:
        return lines[0].split()[0]
    
    raise RuntimeError("Could not parse Ollama model list")


def _detect_api_base() -> str:
    """Auto-detect API base URL. Fails if Ollama not accessible."""
    # Check environment first
    if base := os.getenv("OPENAI_API_BASE"):
        return base
    
    # Check if Ollama is running
    import httpx
    response = httpx.get("http://localhost:11434/api/tags", timeout=2)
    
    if response.status_code != 200:
        raise RuntimeError(
            "Ollama not accessible at http://localhost:11434. "
            "Make sure Ollama is running or set OPENAI_API_BASE environment variable."
        )
    
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
    output += f"\n📝 Answer: {result['final']}\n"
    
    confidence = result.get("diagnostics", {}).get("confidence", 0)
    if confidence >= 0.8:
        output += f"\n✅ Confidence: {confidence:.0%}"
    else:
        output += f"\n⚠️  Confidence: {confidence:.0%}"
    
    iterations = result.get("diagnostics", {}).get("iterations", 0)
    if iterations > 0:
        output += f" (after {iterations} reflection{'s' if iterations > 1 else ''})"
    
    return output


def _stream_reasoning(mcp: MCP, query: str) -> Iterator[str]:
    """Stream reasoning steps in real-time."""
    # Generate initial reasoning trace
    trace = mcp.generator.generate_reasoning(query)
    
    yield "💭 Initial reasoning:\n"
    for i, step in enumerate(trace, 1):
        yield f"{i}. {step}\n"
    
    # Reflection
    if mcp.config.max_reflection_iterations > 0:
        yield "\n🔄 Reflecting...\n"
        reflection = mcp.reflection.enhance_reasoning(query, trace)
        
        if reflection.get("improved"):
            yield "\n✨ Improved reasoning:\n"
            for i, step in enumerate(reflection["final_trace"], 1):
                yield f"{i}. {step}\n"
                yield f"{i}. {step}\n"
        else:
            yield "✅ No improvements needed\n"
        
        yield f"\n🎯 Confidence: {reflection['final_verification']['confidence']:.0%}\n"


# Export simple API
__all__ = ["enhance", "MCP", "LLMConfig", "MCPConfig"]

# Export RAG adapters for convenience
from kaelum.core import rag_adapter
__all__.extend(["rag_adapter"])
