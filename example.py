"""
Quick start example for KaelumAI
"""

from kaelum import enhance

# Example 1: Simple math question
print("=" * 60)
print("Example 1: Math reasoning")
print("=" * 60)

result = enhance("What is 25% of 80?")
print(f"Result: {result}")
print()

# Example 2: With mode specification
print("=" * 60)
print("Example 2: Math mode")
print("=" * 60)

result = enhance("Solve: 3x + 5 = 20", mode="math")
print(f"Result: {result}")
print()

# Example 3: Complex reasoning
print("=" * 60)
print("Example 3: Logic reasoning")
print("=" * 60)

result = enhance(
    "If all birds can fly, and penguins are birds, can penguins fly?",
    mode="logic"
)
print(f"Result: {result}")
print()

print("✅ All examples completed!")
