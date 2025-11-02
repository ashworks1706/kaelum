"""Example: Using Kaelum's plugin system."""

import asyncio
from kaelum.plugins import ReasoningPlugin
from kaelum.core.metrics import CostTracker
from kaelum.core.registry import ModelRegistry, ModelSpec, get_registry


async def main():
    # Initialize cost tracker
    tracker = CostTracker()
    tracker.start_session("demo_session", metadata={"user": "demo"})
    
    # Register models
    registry = get_registry()
    registry.register(ModelSpec(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        model_type="reasoning",
        base_url="http://localhost:8000/v1",
        description="General-purpose reasoning model",
        domain="general",
        context_length=32768,
        quantization="4bit",
        vram_gb=5.5
    ))
    
    # Set as default
    registry.set_default("reasoning", "Qwen/Qwen2.5-7B-Instruct")
    
    # Create reasoning plugin
    reasoning = ReasoningPlugin(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:8000/v1"
    )
    
    # Process query
    print("🧠 Kaelum Reasoning Plugin Demo\n")
    query = "If a train travels 120 km in 2 hours, how far will it travel in 5 hours at the same speed?"
    
    print(f"Query: {query}\n")
    print("Processing...\n")
    
    result = await reasoning.process(query, max_tokens=1000, temperature=0.7)
    
    print(f"Result:\n{result}\n")
    
    # Get plugin metrics
    metrics = reasoning.get_metrics()
    print("\n📊 Plugin Metrics:")
    print(f"  Inferences: {metrics['inferences']}")
    print(f"  Avg Latency: {metrics['avg_latency_ms']:.2f}ms")
    print(f"  Total Cost: ${metrics['total_cost']:.6f}")
    
    savings = reasoning.get_cost_savings()
    print(f"  Estimated Savings: ${savings:.4f} vs Gemini 2.0 Flash")
    
    # Show registered models
    print("\n📚 Registered Models:")
    for model in registry.list_all():
        print(f"  - {model.model_id} ({model.model_type})")
        print(f"    Domain: {model.domain}, VRAM: {model.vram_gb}GB")


if __name__ == "__main__":
    asyncio.run(main())
