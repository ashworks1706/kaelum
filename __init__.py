__version__ = "2.0.0"

from typing import Optional, Dict, Any
from core.config import KaelumConfig, LLMConfig
from runtime.orchestrator import KaelumOrchestrator
from core.metrics import CostTracker
from core.router import Router, QueryType, ReasoningStrategy

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
    global _orchestrator
    
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
    
    _orchestrator = KaelumOrchestrator(config, enable_routing=enable_routing)


def enhance(query: str) -> str:
    global _orchestrator
    
    if _orchestrator is None:
        set_reasoning_model()
    
    result = _orchestrator.infer(query, stream=False)
    
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
    global _orchestrator
    
    if _orchestrator is None:
        set_reasoning_model()
    
    for chunk in _orchestrator.infer(query, stream=True):
        yield chunk


def kaelum_enhance_reasoning(query: str, domain: str = "general") -> Dict[str, Any]:
    global _orchestrator
    
    if _orchestrator is None:
        set_reasoning_model()
    
    result = _orchestrator.infer(query, stream=False)
    
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
    }


__all__ = [
    "enhance", 
    "enhance_stream", 
    "set_reasoning_model",
    "kaelum_enhance_reasoning",
    "CostTracker",
    "Router",
    "QueryType",
    "ReasoningStrategy",
]
