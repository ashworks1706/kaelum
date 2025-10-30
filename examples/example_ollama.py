"""
Example: Using KaelumAI with Ollama and Open-Source Models

This example demonstrates:
1. Using Ollama instead of Gemini
2. Enabling RAG for factual verification
3. Using Redis caching
4. Comparing different models
"""

import os
import time
from kaelum.core.config import LLMConfig, MCPConfig
from kaelum.runtime.orchestrator import MCP


def example_basic_ollama():
    """Basic example using Ollama with Qwen model."""
    print("=" * 60)
    print("Example 1: Basic Reasoning with Ollama")
    print("=" * 60)
    
    # Configure to use Ollama with Qwen 2.5
    config = MCPConfig(
        llm=LLMConfig(
            model="qwen2.5:7b",
            provider="ollama",
            temperature=0.7
        ),
        use_symbolic=True,
        use_cache=True,
        use_rag=False
    )
    
    mcp = MCP(config)
    
    # Test reasoning
    query = "If 3x + 5 = 11, what is x?"
    print(f"\nQuery: {query}")
    
    result = mcp.infer(query)
    
    print(f"\nFinal Answer: {result.final}")
    print(f"Verified: {result.verified}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Latency: {result.diagnostics['latency']:.2f}s")
    print(f"\nReasoning Steps:")
    for i, step in enumerate(result.trace, 1):
        print(f"  {i}. {step}")


def example_with_rag():
    """Example using RAG for factual verification."""
    print("\n" + "=" * 60)
    print("Example 2: Reasoning with RAG Knowledge Base")
    print("=" * 60)
    
    # Configure with RAG enabled
    config = MCPConfig(
        llm=LLMConfig(model="qwen2.5:7b", provider="ollama"),
        use_symbolic=True,
        use_rag=True,
        use_cache=False  # Disable cache for this demo
    )
    
    mcp = MCP(config)
    
    # Add knowledge to RAG
    print("\nAdding knowledge to RAG...")
    verifier = mcp.verification_engine.factual_verifier
    verifier.add_to_knowledge_base(
        texts=[
            "Paris is the capital of France.",
            "The Eiffel Tower is located in Paris.",
            "France is a country in Western Europe.",
            "Python was created by Guido van Rossum in 1991.",
            "Python is an interpreted, high-level programming language.",
        ],
        metadatas=[
            {"topic": "geography", "country": "France"},
            {"topic": "landmarks", "country": "France"},
            {"topic": "geography", "country": "France"},
            {"topic": "programming", "language": "Python"},
            {"topic": "programming", "language": "Python"},
        ]
    )
    
    print(f"Knowledge base size: {verifier.get_knowledge_base_size()} documents")
    
    # Test with factual query
    query = "What is the capital of France and where is the Eiffel Tower?"
    print(f"\nQuery: {query}")
    
    result = mcp.infer(query)
    
    print(f"\nFinal Answer: {result.final}")
    print(f"Verified: {result.verified}")
    print(f"Confidence: {result.confidence:.2f}")


def example_caching():
    """Example demonstrating caching benefits."""
    print("\n" + "=" * 60)
    print("Example 3: Caching Performance")
    print("=" * 60)
    
    config = MCPConfig(
        llm=LLMConfig(model="qwen2.5:7b", provider="ollama"),
        use_cache=True
    )
    
    mcp = MCP(config)
    
    query = "Calculate the area of a circle with radius 5."
    
    # First call - no cache
    print(f"\nQuery: {query}")
    print("\n1st call (no cache):")
    start = time.time()
    result1 = mcp.infer(query)
    latency1 = time.time() - start
    print(f"   Latency: {latency1:.3f}s")
    print(f"   Cached: {result1.diagnostics.get('cached', False)}")
    
    # Second call - should hit cache
    print("\n2nd call (with cache):")
    start = time.time()
    result2 = mcp.infer(query)
    latency2 = time.time() - start
    print(f"   Latency: {latency2:.3f}s")
    print(f"   Cached: {result2.diagnostics.get('cached', False)}")
    
    speedup = latency1 / latency2 if latency2 > 0 else 0
    print(f"\nSpeedup: {speedup:.1f}x faster!")
    
    # Check cache stats
    print("\nCache Statistics:")
    stats = mcp.cache.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")


def example_model_comparison():
    """Compare different models."""
    print("\n" + "=" * 60)
    print("Example 4: Model Comparison")
    print("=" * 60)
    
    models = [
        ("llama3.2:3b", "Llama 3.2 (3B) - Fast"),
        ("qwen2.5:7b", "Qwen 2.5 (7B) - Balanced"),
    ]
    
    query = "If a train travels 120 km in 2 hours, what is its average speed?"
    print(f"\nQuery: {query}\n")
    
    for model, description in models:
        print(f"\n{description}:")
        print("-" * 40)
        
        config = MCPConfig(
            llm=LLMConfig(model=model, provider="ollama"),
            use_cache=False  # Disable cache for fair comparison
        )
        
        try:
            mcp = MCP(config)
            start = time.time()
            result = mcp.infer(query)
            latency = time.time() - start
            
            print(f"Answer: {result.final}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Latency: {latency:.2f}s")
            print(f"Verified: {result.verified}")
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure the model is installed: ollama pull " + model)


def example_custom_config():
    """Example with custom configuration."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Configuration")
    print("=" * 60)
    
    # Highly customized config
    config = MCPConfig(
        llm=LLMConfig(
            model="qwen2.5:7b",
            provider="ollama",
            temperature=0.7,
            max_tokens=1024
        ),
        verifier_llm=LLMConfig(
            model="qwen2.5:7b",
            provider="ollama",
            temperature=0.2,  # Lower temp for stricter verification
            max_tokens=512
        ),
        reflector_llm=LLMConfig(
            model="qwen2.5:7b",
            provider="ollama",
            temperature=0.5,
            max_tokens=512
        ),
        use_symbolic=True,
        use_rag=False,
        use_cache=True,
        confidence_threshold=0.8,  # Higher threshold
        max_reflection_iterations=3,  # More iterations
        enable_policy_controller=True,
        log_traces=True
    )
    
    mcp = MCP(config)
    
    query = "Solve: 2x^2 + 5x - 3 = 0"
    print(f"\nQuery: {query}")
    
    result = mcp.infer(query)
    
    print(f"\nFinal Answer: {result.final}")
    print(f"Verified: {result.verified}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Reflection Iterations: {len(result.diagnostics['reflection']['iterations'])}")
    
    # Show metrics
    print("\n System Metrics:")
    metrics = mcp.get_metrics()
    print(f"   Total Requests: {metrics['total_requests']}")
    print(f"   Verification Rate: {metrics['verification_rate']:.2%}")
    print(f"   Avg Confidence: {metrics['avg_confidence']:.2f}")
    print(f"   Cache Backend: {metrics['cache_stats']['backend']}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("KaelumAI - Ollama Examples")
    print("=" * 60)
    print("\nMake sure Ollama is running: ollama serve")
    print("And you have the models: ollama pull qwen2.5:7b")
    print()
    
    try:
        # Run examples
        example_basic_ollama()
        example_with_rag()
        example_caching()
        example_model_comparison()
        example_custom_config()
        
        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Pull the model: ollama pull qwen2.5:7b")
        print("3. Check if Ollama is accessible: curl http://localhost:11434/api/tags")
