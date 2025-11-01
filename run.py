"""
KaelumAI Demo - Test YOUR reasoning model with different settings
"""

from kaelum import enhance, set_reasoning_model

print("🧠 KaelumAI Demo")
print("=" * 60)

# Set YOUR reasoning model with tweakable parameters
set_reasoning_model(
    provider="ollama",
    model="llama3.2:3b",
    temperature=0.7,              # 0.0 = deterministic, 2.0 = creative
    max_tokens=2048,              # Max response length
    max_reflection_iterations=2,  # Self-correction iterations
    use_symbolic_verification=True,  # Math checking
    use_factual_verification=False,  # RAG checking (needs rag_adapter)
)

print("=" * 60)

# Test query
query = "What is 25% of 80?"
print(f"\nQuery: {query}\n")

result = enhance(query)
print(result)

