"""Demo: Router selecting optimal strategies for different query types."""

from kaelum.core.router import Router, QueryType, ReasoningStrategy

# Initialize router (loads learned performance data)
router = Router(learning_enabled=True)

# Test queries of different types
test_cases = [
    "Calculate the compound interest on $5000 at 6% for 3 years",
    "If all humans are mortal and Socrates is human, what can we conclude?",
    "What year did the Apollo 11 mission land on the moon?",
    "Analyze the time complexity of bubble sort algorithm",
]

print("="*70)
print("KAELUM ROUTER DEMO")
print("="*70)
print("\nShowing how the router adapts strategy per query type:\n")

for i, query in enumerate(test_cases, 1):
    print(f"Query {i}: {query}\n")
    
    # Route the query
    decision = router.route(query)
    
    print(f"  🏷️  Type: {decision.query_type.value}")
    print(f"  🎯 Strategy: {decision.strategy.value}")
    print(f"  ⚙️  Config:")
    print(f"      - Reflection iterations: {decision.max_reflection_iterations}")
    print(f"      - Symbolic verification: {decision.use_symbolic_verification}")
    print(f"      - Factual verification: {decision.use_factual_verification}")
    print(f"      - Confidence threshold: {decision.confidence_threshold}")
    print(f"  💭 Reasoning: {decision.reasoning}\n")
    print("-" * 70 + "\n")

# Show performance summary
print("="*70)
print("LEARNED PERFORMANCE DATA")
print("="*70)

summary = router.get_performance_summary()
print(f"\nTotal queries learned from: {summary['total_queries']}")
print(f"Outcomes logged: {summary['outcomes_logged']}\n")

if summary['by_strategy']:
    print("Top strategies by query type:\n")
    for query_type, stats in summary['by_strategy'].items():
        strategies = sorted(
            stats['strategies'].items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        
        if strategies:
            best = strategies[0]
            print(f"  {query_type.upper()}")
            print(f"    → {best[0]}: {best[1]['accuracy']:.1%} accuracy, "
                  f"{best[1]['avg_latency']:.0f}ms avg latency")

print("\n" + "="*70)
print("✓ Router learns and adapts from every inference!")
print("="*70)
