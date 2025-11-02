"""KaelumAI - Standalone Streaming Demo

Start vLLM server first (for 6GB GPU):
    python -m vllm.entrypoints.openai.api_server \
        --model TinyLlama/TinyLlama-1.1B-Chat-v0.3 \
        --port 8000 \
        --gpu-memory-utilization 0.7 \
        --max-num-seqs 32 \
        --max-model-len 1024 \
        --chat-template "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}assistant: "

For larger GPUs (8GB+), you can remove the memory limits.
"""
import os
from kaelum import enhance_stream, set_reasoning_model

set_reasoning_model(
    base_url="http://localhost:8000/v1",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    # model="Qwen/Qwen2.5-1.5B-Instruct",
    # model="TinyLlama/TinyLlama-1.1B-Chat-v0.3",
    temperature=0.7,
    max_tokens=512,
)

query = "What is 25% of 80?"

# get txt file

with open("demo_query.txt", "r") as f:
    query = "Summarize this: " + f.read().strip()

print(f"Query: {query}\n")

try:
    for chunk in enhance_stream(query):
        print(chunk, end='', flush=True)
    print()
except Exception as e:
    print(f"Error: {e}")

