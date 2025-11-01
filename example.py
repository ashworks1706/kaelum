"""
KaelumAI Demo - Reasoning enhancement using YOUR local model
"""

from kaelum import enhance, set_reasoning_model

print("🧠 KaelumAI Demo")
print("=" * 60)

# Set YOUR reasoning model (optional - defaults to Ollama llama3.2:3b)
set_reasoning_model(
    provider="ollama",
    model="llama3.2:3b",
)

print("✓ Reasoning model: llama3.2:3b (Ollama)")
print("=" * 60)

# User query
query = "What is 25% of 80?"
print(f"\nQuery: {query}\n")

# Get enhanced answer
result = enhance(query)
print(result)

