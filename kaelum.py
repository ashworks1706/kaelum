__version__ = "2.0.0"

from typing import Optional, Dict, Any, List
from core.config import KaelumConfig, LLMConfig
from runtime.orchestrator import KaelumOrchestrator
from core.learning import CostTracker, TokenCounter, AnalyticsDashboard
from core.search import Router, QueryType, ReasoningStrategy
from core.learning import ActiveLearningEngine, QuerySelector
from core.paths import DEFAULT_CACHE_DIR, DEFAULT_ROUTER_DIR

_orchestrator: Optional[KaelumOrchestrator] = None
_embedding_model: str = "all-MiniLM-L6-v2"

def set_reasoning_model(
    base_url: str = "http://localhost:11434/v1",
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    api_key: Optional[str] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    max_reflection_iterations: int = 2,
    use_symbolic_verification: bool = True,
    use_factual_verification: bool = False,
    enable_routing: bool = True,
    enable_active_learning: bool = True,
    debug_verification: bool = False,
    cache_dir: str = DEFAULT_CACHE_DIR,
    router_data_dir: str = DEFAULT_ROUTER_DIR,
    parallel: bool = False,
    max_workers: int = 4,
    max_tree_depth: Optional[int] = None,
    num_simulations: Optional[int] = None,
    router_learning_rate: float = 0.001,
    router_buffer_size: int = 32,
    router_exploration_rate: float = 0.1,
):
    global _orchestrator, _embedding_model
    
    _embedding_model = embedding_model
    
    config = KaelumConfig(
        reasoning_llm=LLMConfig(
            base_url=base_url,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        embedding_model=embedding_model,
        max_reflection_iterations=max_reflection_iterations,
        use_symbolic_verification=use_symbolic_verification,
        use_factual_verification=use_factual_verification,
        debug_verification=debug_verification,
    )
    
    _orchestrator = KaelumOrchestrator(
        config,
        enable_routing=enable_routing,
        enable_active_learning=enable_active_learning,
        cache_dir=cache_dir,
        router_data_dir=router_data_dir,
        parallel=parallel,
        max_workers=max_workers,
        max_tree_depth=max_tree_depth,
        num_simulations=num_simulations,
        router_learning_rate=router_learning_rate,
        router_buffer_size=router_buffer_size,
        router_exploration_rate=router_exploration_rate,
    )

def get_embedding_model() -> str:
    """Get the currently configured embedding model name."""
    return _embedding_model

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

def get_metrics() -> Dict[str, Any]:
    """Get comprehensive metrics and analytics."""
    global _orchestrator
    
    if _orchestrator is None:
        return {"error": "Orchestrator not initialized"}
    
    return _orchestrator.get_metrics_summary()

def get_active_learning_stats() -> Dict[str, Any]:
    """Get active learning statistics."""
    global _orchestrator
    
    if _orchestrator is None:
        return {"error": "Orchestrator not initialized"}
    
    return _orchestrator.get_active_learning_stats()

def generate_training_batch(
    strategy: str = "mixed",
    batch_size: int = 20
) -> List[Dict[str, Any]]:
    """Generate training batch using active learning.
    
    Args:
        strategy: Selection strategy (uncertainty, diversity, error, complexity, mixed)
        batch_size: Number of queries to select
    
    Returns:
        List of training examples
    """
    global _orchestrator
    
    if _orchestrator is None:
        set_reasoning_model()
    
    return _orchestrator.generate_training_batch(strategy, batch_size)

def export_training_data(output_path: str) -> int:
    """Export collected training data.
    
    Args:
        output_path: Path to save training dataset
    
    Returns:
        Number of examples exported
    """
    global _orchestrator
    
    if _orchestrator is None:
        return 0
    
    return _orchestrator.export_training_dataset(output_path)

__all__ = [
    "enhance", 
    "enhance_stream", 
    "set_reasoning_model",
    "kaelum_enhance_reasoning",
    "get_metrics",
    "get_active_learning_stats",
    "generate_training_batch",
    "export_training_data",
    "CostTracker",
    "TokenCounter",
    "AnalyticsDashboard",
    "Router",
    "QueryType",
    "ReasoningStrategy",
    "ActiveLearningEngine",
    "QuerySelector",
]
