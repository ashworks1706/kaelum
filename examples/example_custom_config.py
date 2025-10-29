"""
Example 3: Custom MCP configuration

This example shows advanced configuration options.
"""

import os
from kaelum import MCP, MCPConfig, LLMConfig

# Set API keys
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Create custom configuration
config = MCPConfig(
    llm=LLMConfig(model="gpt-4o", temperature=0.7),
    verifier_llm=LLMConfig(model="gpt-3.5-turbo", temperature=0.3),
    reflector_llm=LLMConfig(model="gpt-4o-mini", temperature=0.5),
    use_symbolic=True,
    use_rag=False,
    confidence_threshold=0.8,
    max_reflection_iterations=3,
    enable_policy_controller=True,
    log_traces=True,
)

# Create MCP with custom config
mcp = MCP(config)

# Run multiple queries
queries = [
    "What is the derivative of x^2 + 3x + 5?",
    "If I have 100 apples and give away 30%, how many do I have left?",
    "Explain the relationship between entropy and information theory.",
]

print("=" * 60)
print("RUNNING MULTIPLE QUERIES WITH CUSTOM CONFIG")
print("=" * 60)

for i, query in enumerate(queries, 1):
    print(f"\n[Query {i}] {query}")
    result = mcp.infer(query)
    print(f"Answer: {result.final}")
    print(f"Verified: {result.verified} | Confidence: {result.confidence:.2f}")
    print("-" * 60)

# Get final metrics
metrics = mcp.get_metrics()
print("\nFINAL METRICS:")
print(f"  Total requests: {metrics['total_requests']}")
print(f"  Verified: {metrics['verified_count']}")
print(f"  Failed: {metrics['failed_count']}")
print(f"  Verification rate: {metrics['verification_rate']:.2%}")
print(f"  Avg confidence: {metrics['avg_confidence']:.2f}")
print(f"  Avg iterations: {metrics['avg_iterations']:.1f}")
print("=" * 60)
