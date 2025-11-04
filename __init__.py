"""KaelumAI - Local reasoning models as cognitive middleware for commercial LLMs."""

__version__ = "2.0.0"

from typing import Optional, Dict, Any
from core.config import KaelumConfig, LLMConfig
from kaelum.runtime.orchestrator import KaelumOrchestrator

# Core Infrastructure
from core.metrics import CostTracker
from core.router import Router, QueryType, ReasoningStrategy

# Global orchestrator with complete pipeline:
# Router → Worker (LATS + Cache) → Verification → Reflection
_orchestrator: Optional[KaelumOrchestrator] = None


def set_reasoning_model(
    base_url: str = "http://localhost:11434/v1",
    model: str = "qwen2.5:3b",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    max_reflection_iterations: int = 2,
    use_symbolic_verification: bool = True,
    use_factual_verification: bool = False,
    enable_routing: bool = True,
    debug_verification: bool = False,
):
    """Configure reasoning model with verification and reflection.
    
    Kaelum Architecture:
    1. Router: Intelligently routes query to expert worker (math/logic/code/factual/creative)
    2. Worker: Uses LATS (tree search) + caching for multi-step reasoning
    3. Verification: Checks correctness (symbolic math, logic, etc.)
    4. Reflection: If verification fails, improves reasoning and retries (max iterations)
    
    Args:
        base_url: API endpoint (Ollama: localhost:11434/v1, vLLM: localhost:8000/v1)
        model: Model name (e.g., 'qwen2.5:3b', 'Qwen/Qwen2.5-3B-Instruct')
        api_key: API key if required (optional for local servers)
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Max tokens to generate
        max_reflection_iterations: Max self-correction iterations (0-5)
        use_symbolic_verification: Enable math verification with SymPy
        use_factual_verification: Enable factual checking
        enable_routing: Enable intelligent worker selection
        debug_verification: Enable detailed debug logging for verification
    """
    global _orchestrator

    print(f"Initializing Kaelum v{__version__}")
    print(f"Model: {model}")
    print(f"Architecture: Router → Worker (LATS) → Verify → Reflect")
    
    config = KaelumConfig(
        reasoning_llm=LLMConfig(
            base_url=base_url,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        max_reflection_iterations=max_reflection_iterations,
        use_symbolic_verification=use_symbolic_verification,
        use_factual_verification=use_factual_verification,
        debug_verification=debug_verification,
    )
    
    _orchestrator = KaelumOrchestrator(
        config,
        enable_routing=enable_routing,
    )


def enhance(query: str) -> str:
    """Enhance reasoning for a query using the complete Kaelum pipeline.
    
    Architecture:
    1. Router → Classifies query and selects expert worker
    2. Worker → Uses LATS (MCTS tree search) + caching for multi-step reasoning
    3. Verification → Checks reasoning correctness (symbolic math, logic, etc.)
    4. Reflection → If verification fails, improves reasoning and retries
    5. Loop until verification passes or max iterations reached
    
    Args:
        query: User's question
    
    Returns:
        Answer with reasoning trace and verification status
    """
    global _orchestrator
    
    if _orchestrator is None:
        set_reasoning_model()
    
    # Run through complete pipeline
    result = _orchestrator.infer(query, stream=False)
    
    # Format output cleanly
    answer = result.get("answer", "").strip()
    trace = result.get("reasoning_trace", [])
    worker = result.get("worker", "unknown")
    confidence = result.get("confidence", 0.0)
    verification = result.get("verification_passed", False)
    iterations = result.get("iterations", 1)
    cache_hit = result.get("cache_hit", False)
    
    output = f"{answer}\n\n"
    output += f"Worker: {worker} | Confidence: {confidence:.2f} | "
    output += f"Verification: {'✓ PASSED' if verification else '✗ FAILED'}"
    if iterations > 1:
        output += f" | Iterations: {iterations}"
    if cache_hit:
        output += " | 🎯 Cache Hit"
    output += "\n\nReasoning:"
    
    for i, step in enumerate(trace, 1):
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
    global _orchestrator
    
    if _orchestrator is None:
        set_reasoning_model()
    
    # Stream the response through the pipeline
    for chunk in _orchestrator.infer(query, stream=True):
        yield chunk


def kaelum_enhance_reasoning(query: str, domain: str = "general") -> Dict[str, Any]:
    """Function for commercial LLMs to call for reasoning enhancement.
    
    Uses complete Kaelum pipeline:
    Router → Worker (LATS + Cache) → Verification → Reflection
    
    Args:
        query: The question or problem that needs reasoning enhancement
        domain: Optional domain hint (math, logic, code, science, general)
    
    Returns:
        Dictionary with reasoning trace, answer, verification status
    """
    global _orchestrator
    
    if _orchestrator is None:
        set_reasoning_model()
    
    # Run through complete pipeline
    result = _orchestrator.infer(query, stream=False)
    
    # Format for function calling response
    return {
        "reasoning_steps": result.get("reasoning_trace", []),
        "reasoning_count": len(result.get("reasoning_trace", [])),
        "suggested_approach": result.get("answer", ""),
        "worker_used": result.get("worker", "unknown"),
        "confidence": result.get("confidence", 0.0),
        "verification_passed": result.get("verification_passed", False),
        "iterations": result.get("iterations", 1),
        "cache_hit": result.get("cache_hit", False),
        "domain": domain,
        "note": "Reasoning generated using Kaelum: Router → Worker (LATS) → Verification → Reflection"
    }


__all__ = [
    # Core functions
    "enhance", 
    "enhance_stream", 
    "set_reasoning_model",
    "kaelum_enhance_reasoning",
    
    # Infrastructure
    "CostTracker",
    
    # Routing
    "Router",
    "QueryType",
    "ReasoningStrategy",
]
