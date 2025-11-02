"""KaelumAI - Standalone Streaming Demo"""

from kaelum import enhance_stream, set_reasoning_model

set_reasoning_model(
    base_url="http://localhost:8000/v1",  # vLLM OpenAI-compatible endpoint
    model="TinyLlama/TinyLlama-1.1B-Chat-v0.3",
    temperature=0.3,
    max_tokens=512,
)

query = "What is 25% of 80?"
print(f"Query: {query}\n")

try:
    for chunk in enhance_stream(query):
        print(chunk, end='', flush=True)
    print()
except Exception as e:
    print(f"Error: {e}")

