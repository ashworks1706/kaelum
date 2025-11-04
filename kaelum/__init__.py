"""KaelumAI - Local reasoning models as cognitive middleware for commercial LLMs."""

__version__ = "1.5.0"

from typing import Optional, Dict, Any
from kaelum.core.config import KaelumConfig, LLMConfig
from kaelum.runtime.orchestrator import KaelumOrchestrator
from kaelum.core.tools import (
    get_kaelum_function_schema, 
    get_all_kaelum_schemas,
    kaelum_verify_math,
    kaelum_compute_math
)

# Infrastructure
from kaelum.core.metrics import CostTracker
from kaelum.core.registry import (
    ModelRegistry, 
    ModelSpec, 
    MathCapabilities,
    get_registry,
    register_default_math_models
)
from kaelum.core.router import Router, QueryType, ReasoningStrategy

# Global orchestrator with verification + reflection
_orchestrator: Optional[KaelumOrchestrator] = None


def set_reasoning_model(
    base_url: str = "http://localhost:11434/v1",
    model: str = "TinyLlama/TinyLlama-1.1B-Chat-v0.3",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    max_reflection_iterations: int = 2,
    use_symbolic_verification: bool = True,
    use_factual_verification: bool = False,
    enable_routing: bool = True,
    debug_verification: bool = False,
    strict_math_format: bool = False,
    rag_adapter = None,
    reasoning_system_prompt: Optional[str] = None,
    reasoning_user_template: Optional[str] = None,
):
    """
    Configure reasoning model with verification and reflection.
    
    Args:
        base_url: API endpoint (default: vLLM at localhost:8000/v1)
        model: Model name (full HuggingFace path for vLLM)
        api_key: API key if required (optional for local servers)
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Max tokens to generate
        max_reflection_iterations: Self-correction iterations (0-5)
        use_symbolic_verification: Enable math verification with SymPy
        use_factual_verification: Enable RAG-based fact checking
        enable_routing: Enable adaptive strategy selection (Phase 2)
        debug_verification: Enable detailed debug logging for verification
        rag_adapter: RAG adapter instance (required if use_factual_verification=True)
        reasoning_system_prompt: Custom system prompt for reasoning model
        reasoning_user_template: Custom user prompt template. Use {query} placeholder.
    """
    global _orchestrator

    print("Setting reasoning model to:", model)
    
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
        strict_math_format=strict_math_format,
    )
    
    _orchestrator = KaelumOrchestrator(
        config, 
        rag_adapter=rag_adapter,
        reasoning_system_prompt=reasoning_system_prompt,
        reasoning_user_template=reasoning_user_template,
        enable_routing=enable_routing,
    )


def enhance(query: str) -> str:
    """
    Enhance reasoning for a query with verification and reflection.
    
    Args:
        query: User's question
    
    Returns:
        Answer with reasoning trace
    """
    global _orchestrator
    
    if _orchestrator is None:
        set_reasoning_model()
    
    # Ensure orchestrator is not None
    assert _orchestrator is not None, "Orchestrator failed to initialize"
    
    # Run through Generate → Verify → Reflect → Answer pipeline
    result = _orchestrator.infer(query, stream=False)
    
    # Ensure result is a dictionary
    assert isinstance(result, dict), f"Expected dict result, got {type(result)}"
    
    # Format output cleanly
    final = result["answer"].strip()
    trace = result["reasoning_trace"]
    
    output = f"{final}\n\nReasoning:"
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
    
    # Ensure orchestrator is not None
    assert _orchestrator is not None, "Orchestrator failed to initialize"
    
    # Stream the response through the pipeline
    for chunk in _orchestrator.infer(query, stream=True):
        yield chunk


def kaelum_enhance_reasoning(query: str, domain: str = "general") -> Dict[str, Any]:
    """
    Function for commercial LLMs to call for reasoning enhancement.
    Uses full pipeline: Generate → Verify → Reflect → Answer
    
    Args:
        query: The question or problem that needs reasoning enhancement
        domain: Optional domain hint (math, logic, code, science, general)
    
    Returns:
        Dictionary with reasoning trace and suggested answer structure
    """
    global _orchestrator
    
    if _orchestrator is None:
        set_reasoning_model()
    
    # Ensure orchestrator is not None
    assert _orchestrator is not None, "Orchestrator failed to initialize"
    
    # Run through full verification + reflection pipeline
    result = _orchestrator.infer(query, stream=False)
    
    # Ensure result is a dictionary
    assert isinstance(result, dict), f"Expected dict result, got {type(result)}"
    
    # Format for function calling response
    return {
        "reasoning_steps": result["reasoning_trace"],
        "reasoning_count": len(result["reasoning_trace"]),
        "suggested_approach": result["answer"],
        "domain": domain,
        "note": "Use these verified reasoning steps to formulate your comprehensive answer"
    }


# Export function schemas for LLM integration
def get_function_schema(format: str = "openai") -> Dict[str, Any]:
    """
    Get the function schema for integrating Kaelum with commercial LLMs.
    
    Args:
        format: "openai" (default, works for GPT-4, Claude) or "gemini"
    
    Returns:
        Function schema that can be passed to the LLM
    """
    return get_kaelum_function_schema()


__all__ = [
    # Core functions
    "enhance", 
    "enhance_stream", 
    "set_reasoning_model",
    "kaelum_enhance_reasoning",
    "get_function_schema",
    
    # Math functions
    "kaelum_verify_math",
    "kaelum_compute_math", 
    "get_all_kaelum_schemas",
    
    # Infrastructure
    "CostTracker",
    "ModelRegistry",
    "ModelSpec",
    "MathCapabilities",
    "get_registry",
    "register_default_math_models",
    
    # Routing (Phase 2)
    "Router",
    "QueryType",
    "ReasoningStrategy",
]


# Initialize default math models on import
register_default_math_models()
