"""KaelumAI - Local reasoning models as cognitive middleware for commercial LLMs."""

__version__ = "1.5.0"

from typing import Optional, Dict, Any
from core.config import KaelumConfig, LLMConfig
from kaelum.runtime.orchestrator import KaelumOrchestrator
from core.tools import get_kaelum_function_schema

# Infrastructure
from core.metrics import CostTracker
from core.registry import ModelRegistry, ModelSpec, get_registry
from core.router import Router, QueryType, ReasoningStrategy
from core.neural_router import NeuralRouter
from core.neural_router_trainer import NeuralRouterTrainer

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
    Enhance reasoning for a query using LATS-based worker agents.
    
    Architecture:
    1. Router classifies query and selects specialist worker
    2. Worker uses LATS (MCTS) tree search for multi-step reasoning
    3. Returns verified answer with reasoning trace
    
    Args:
        query: User's question
    
    Returns:
        Answer with reasoning trace
    """
    global _orchestrator
    
    if _orchestrator is None:
        set_reasoning_model()
    
    # Run through Router → Worker → LATS pipeline
    result = _orchestrator.infer(query, stream=False)
    
    # Format output cleanly
    answer = result.get("answer", "").strip()
    trace = result.get("reasoning_trace", [])
    worker = result.get("worker", "unknown")
    confidence = result.get("confidence", 0.0)
    cache_hit = result.get("cache_hit", False)
    
    output = f"{answer}\n\n"
    output += f"Worker: {worker} | Confidence: {confidence:.2f}"
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
    """
    Function for commercial LLMs to call for reasoning enhancement.
    Uses Router → Worker → LATS pipeline for verified multi-step reasoning.
    
    Args:
        query: The question or problem that needs reasoning enhancement
        domain: Optional domain hint (math, logic, code, science, general)
    
    Returns:
        Dictionary with reasoning trace and suggested answer structure
    """
    global _orchestrator
    
    if _orchestrator is None:
        set_reasoning_model()
    
    # Run through worker-based LATS reasoning
    result = _orchestrator.infer(query, stream=False)
    
    # Format for function calling response
    return {
        "reasoning_steps": result.get("reasoning_trace", []),
        "reasoning_count": len(result.get("reasoning_trace", [])),
        "suggested_approach": result.get("answer", ""),
        "worker_used": result.get("worker", "unknown"),
        "confidence": result.get("confidence", 0.0),
        "cache_hit": result.get("cache_hit", False),
        "domain": domain,
        "note": "These reasoning steps were generated using LATS (Language Agent Tree Search) with MCTS exploration"
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
    
    # Infrastructure
    "CostTracker",
    "ModelRegistry",
    "ModelSpec",
    "get_registry",
    
    # Routing (Phase 2)
    "Router",
    "QueryType",
    "ReasoningStrategy",
    
    # Neural Routing (Kaelum Brain)
    "NeuralRouter",
    "NeuralRouterTrainer",
]
