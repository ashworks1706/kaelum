"""Test script to demonstrate debug logging for math verification."""

from kaelum import set_reasoning_model, enhance

# Configure with debug mode enabled
set_reasoning_model(
    base_url="http://localhost:11434/v1",
    model="qwen3:4b",
    use_symbolic_verification=True,
    debug_verification=True  # Enable detailed debug output
)

# Test with a multivariate calculus problem
query = """
Given f(x,y) = x**2 * y + sin(x*y), calculate:
1. The partial derivative ∂f/∂x
2. The second partial derivative ∂²f/∂x∂y

Show all steps with the derivative calculations.
"""

print("="*80)
print("Testing Math Verification with Detailed Debug Logging")
print("="*80)
print(f"\nQuery: {query.strip()}\n")
print("="*80)
print("\nWaiting for LLM reasoning with verification debug output:\n")
print("="*80)

result = enhance(query)
print("\n" + "="*80)
print("FINAL RESULT:")
print("="*80)
print(result)
