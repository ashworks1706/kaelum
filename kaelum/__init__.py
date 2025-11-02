"""KaelumAI - Make any LLM reason better."""

from typing import Optional
from kaelum.core.config import LLMConfig, MCPConfig
from kaelum.runtime.orchestrator import MCP

# YOUR reasoning model
_mcp: Optional[MCP] = None


def set_reasoning_model(
    provider: str = "ollama",
    model: str = "llama3.2:3b", 
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    max_reflection_iterations: int = 2,
    use_symbolic_verification: bool = True,
    use_factual_verification: bool = False,
    rag_adapter = None,
):
    """
    Set YOUR reasoning model with all tweakable parameters.
    
    Args:
        provider: "ollama", "vllm", or "custom"
        model: Your model name
        base_url: Your model endpoint (optional, has defaults)
        temperature: Sampling temperature (0.0-2.0, higher = more creative)
        max_tokens: Max tokens to generate (1-128000)
        max_reflection_iterations: Number of self-correction iterations (1-5)
        use_symbolic_verification: Enable math verification with SymPy
        use_factual_verification: Enable RAG-based fact checking
        rag_adapter: RAG adapter instance (required if use_factual_verification=True)
    """
    global _mcp
    
    config = MCPConfig(
        reasoning_llm=LLMConfig(
            provider=provider,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        max_reflection_iterations=max_reflection_iterations,
        use_symbolic_verification=use_symbolic_verification,
        use_factual_verification=use_factual_verification,
    )
    _mcp = MCP(config, rag_adapter=rag_adapter)


def enhance(query: str) -> str:
    """
    Enhance reasoning for a query.
    
    Args:
        query: User's question
    
    Returns:
        Answer with reasoning trace
    """
    global _mcp
    
    if _mcp is None:
        set_reasoning_model()
    
    result = _mcp.infer(query)
    
    # Clean up the output
    final = result["final"].strip()
    trace = result["trace"]
    
    # Format output cleanly
    output = f"{final}\n\nReasoning:"
    for i, step in enumerate(trace, 1):
        # Clean each step
        step_clean = step.strip().replace('\n', ' ')
        output += f"\n{i}. {step_clean}"
    
    return output


def enhance_stream(query: str):
    """
    Enhance reasoning for a query with streaming output.
    
    Args:
        query: User's question
    
    Yields:
        Chunks of the response as they're generated
    """
    global _mcp
    
    if _mcp is None:
        set_reasoning_model()
    
    # Stream the response
    for chunk in _mcp.infer(query, stream=True):
        yield chunk


__all__ = ["enhance", "enhance_stream", "set_reasoning_model"]
