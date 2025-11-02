"""KaelumAI - Standalone Streaming Demo"""

from kaelum import enhance_stream, set_reasoning_model

set_reasoning_model(
    provider="ollama",
    model="tinyllama",
    temperature=0.3,
    max_tokens=512,
    max_reflection_iterations=0,
    use_symbolic_verification=True,
    use_factual_verification=False,
    rag_adapter=None,
)

query = "What is 25% of 80?"
print(f"Query: {query}\n")

try:
    for chunk in enhance_stream(query):
        print(chunk, end='', flush=True)
    print()
except Exception as e:
    print(f"Error: {e}")

