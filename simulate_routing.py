"""Bootstrap router with simulated routing outcomes.

This generates synthetic training data to help the router learn
optimal strategy selection before real-world usage.
"""

from kaelum.core.router import Router

# Test queries covering different types
TEST_QUERIES = [
    {
        "query": "Calculate the total cost of 15 items at $12.99 each with 8.5% tax",
        "expected_type": "MATH",
        "ground_truth": "$179.66"
    },
    {
        "query": "If all cats are mammals and Fluffy is a cat, what can we conclude?",
        "expected_type": "LOGIC",
        "ground_truth": "Fluffy is a mammal"
    },
    {
        "query": "Debug this Python function that's supposed to find the maximum value in a list",
        "expected_type": "CODE",
        "ground_truth": "Code analysis needed"
    },
    {
        "query": "Who was the first president of the United States?",
        "expected_type": "FACTUAL",
        "ground_truth": "George Washington"
    },
    {
        "query": "A train leaves station A at 60 mph. Another train leaves station B at 80 mph. They're 350 miles apart. When do they meet?",
        "expected_type": "MATH",
        "ground_truth": "2.5 hours"
    },
    {
        "query": "Analyze the pros and cons of remote work versus office work",
        "expected_type": "ANALYSIS",
        "ground_truth": "Comparison analysis"
    },
    {
        "query": "Solve for x: 3x^2 - 12x + 9 = 0",
        "expected_type": "MATH",
        "ground_truth": "x = 1 or x = 3"
    },
    {
        "query": "If it rains, then the ground is wet. The ground is wet. Does that mean it rained?",
        "expected_type": "LOGIC",
        "ground_truth": "Not necessarily (affirming the consequent fallacy)"
    },
    {
        "query": "What is the capital of France?",
        "expected_type": "FACTUAL",
        "ground_truth": "Paris"
    },
    {
        "query": "A store offers 20% off, then an additional 10% off the discounted price. Is this the same as 30% off?",
        "expected_type": "MATH",
        "ground_truth": "No, it's 28% off total"
    }
]


def main():
    print("="*70)
    print("KAELUM ROUTING SIMULATION")
    print("="*70)
    print("\nThis will generate synthetic training data to bootstrap the router.")
    print("The router learns which strategies work best for different query types.\n")
    
    # Initialize router with learning enabled
    router = Router(learning_enabled=True)
    
    # Run simulation
    results = router.simulate_queries(TEST_QUERIES, num_runs=5)
    
    # Show performance summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    
    summary = router.get_performance_summary()
    print(f"\nTotal queries simulated: {summary['total_queries']}")
    print(f"Outcomes logged: {summary['outcomes_logged']}")
    print(f"\nPerformance by query type:\n")
    
    for query_type, stats in summary['by_strategy'].items():
        print(f"  {query_type.upper()}")
        print(f"    Queries: {stats['count']}")
        
        # Sort strategies by accuracy
        strategies = sorted(
            stats['strategies'].items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        
        print(f"    Best strategies:")
        for strategy, perf in strategies[:3]:
            print(f"      • {strategy}: "
                  f"acc={perf['accuracy']:.2%}, "
                  f"latency={perf['avg_latency']:.0f}ms, "
                  f"cost=${perf['avg_cost']:.6f}")
        print()
    
    print("="*70)
    print("✓ Simulation complete! Router is ready to learn from real usage.")
    print("="*70)


if __name__ == "__main__":
    main()
