"""
Kaelum AI - Simple Example
Demonstrates basic usage of Kaelum for verified reasoning
"""

from kaelum import set_reasoning_model, enhance

# Configure reasoning model (using local vLLM server)
set_reasoning_model(
    base_url="http://localhost:8000/v1",
    model="TinyLlama/TinyLlama-1.1B-Chat-v0.3",
    temperature=0.7,
    max_tokens=1024,
    use_symbolic_verification=True,
    max_reflection_iterations=2
)

# Example queries
queries = [
    "Calculate 15% of $89.97",
    "If I buy 3 items at $12.99 each with 8% tax, what's the total?",
    "Solve for x: 2x + 6 = 10",
]

print("=" * 80)
print("Kaelum AI - Verified Reasoning Examples")
print("=" * 80)

for i, query in enumerate(queries, 1):
    print(f"\n\n[Query {i}] {query}")
    print("-" * 80)
    
    result = enhance(query)
    print(result)
    print("-" * 80)

print("\n\n" + "=" * 80)
print("All examples complete!")
print("=" * 80)
