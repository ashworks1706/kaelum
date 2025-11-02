"""
KaelumAI Demo - Test YOUR reasoning model with different settings
"""

import sys
from kaelum import enhance_stream, set_reasoning_model

print("🧠 KaelumAI Demo")
print("=" * 60)

# Set YOUR reasoning model with tweakable parameters
set_reasoning_model(
    provider="ollama",
    model="llama3.2:3b",           # Use a better model
    temperature=0.3,               # Lower temp for more focused output
    max_tokens=512,                # Shorter responses
    max_reflection_iterations=0,   # No reflection for faster testing
    use_symbolic_verification=True,   # Math checking
    use_factual_verification=False,   # RAG checking (needs rag_adapter)
)

print("✓ Model: llama3.2:3b")
print("✓ Temperature: 0.3 (focused)")
print("✓ Max tokens: 512")
print("✓ Reflection: OFF (faster)")
print("✓ Verification: Symbolic only")
print("=" * 60)

# Test query
query = "What is 25% of 80?"
print(f"\nQuery: {query}\n")

try:
    # Stream the response to console
    for chunk in enhance_stream(query):
        print(chunk, end='', flush=True)
    print()  # Final newline
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
