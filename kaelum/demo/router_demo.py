"""Demo of Intelligent Router - No keyword matching!

This demonstrates the embedding-based router that understands queries semantically.
"""

from kaelum.core.intelligent_router import IntelligentRouter

def main():
    print("\n" + "="*70)
    print("🧠 INTELLIGENT ROUTER DEMO - Semantic Understanding")
    print("="*70 + "\n")
    
    # Initialize router
    print("Initializing intelligent router...")
    router = IntelligentRouter()
    print(f"✓ Loaded with {len(router.training_examples)} training examples\n")
    
    # Test queries - notice these DON'T use obvious keywords
    test_queries = [
        # Math - but phrased differently
        "If I have 17 apples and give away 5, how many remain?",
        "What's 25% of 80?",
        "Find the slope of y = 3x + 2",
        
        # Logic - subtle phrasing
        "Given that it's raining, and rain makes streets wet, what can we say about the streets?",
        "If the statement 'All cats are animals' is true, and Fluffy is a cat, what follows?",
        
        # Code - not using word "code"
        "I need a function that reverses a string",
        "My loop isn't terminating, can you help?",
        
        # Factual - questions about knowledge
        "Tell me about Marie Curie",
        "What happened during the French Revolution?",
        
        # Creative - imaginative tasks
        "Come up with names for a coffee shop",
        "Describe a futuristic city",
        
        # Mixed/ambiguous
        "How do neural networks learn?",  # Could be factual or analysis
        "Build a calculator",  # Could be code or creative
    ]
    
    print("Testing queries (WITHOUT keyword matching):\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─'*70}")
        print(f"Query {i}: {query}")
        print(f"{'─'*70}")
        
        decision = router.route(query)
        
        print(f"\n✓ Classification:")
        print(f"  Primary Type: {decision.query_type.value}")
        print(f"  Confidence: {decision.confidence:.2f}")
        print(f"  Secondary: {[t.value for t in decision.secondary_types] if decision.secondary_types else 'None'}")
        print(f"  Complexity: {decision.complexity_score:.2f}")
        print(f"  Strategy: {decision.strategy.value}")
        print(f"  Reflection Depth: {decision.max_reflection_iterations}")
    
    print("\n" + "="*70)
    print("✅ Demo Complete - Router uses semantic understanding, not keywords!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
