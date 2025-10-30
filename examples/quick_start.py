"""Quick example of KaelumAI usage."""

from kaelum import enhance

# Example 1: Basic usage
print("=" * 60)
print("Example 1: Basic Math")
print("=" * 60)
result = enhance("What is 15% of 240?")
print(result)
print()

# Example 2: With mode
print("=" * 60)
print("Example 2: Math Mode")
print("=" * 60)
result = enhance("Solve x^2 + 5x + 6 = 0", mode="math")
print(result)
print()

# Example 3: Code reasoning
print("=" * 60)
print("Example 3: Code Mode")
print("=" * 60)
result = enhance("Explain how binary search works", mode="code")
print(result)
print()

# Example 4: Streaming
print("=" * 60)
print("Example 4: Streaming")
print("=" * 60)
for chunk in enhance("What are the prime factors of 84?", stream=True, mode="math"):
    print(chunk, end="", flush=True)
print("\n")
