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


query = "What is 25% of 80?"

# get txt file

with open("demo_query.txt", "r") as f:
    query = "Summarize this: " + f.read().strip()


reasoning_system = """You are a reasoning assistant. Break down problems into clear, logical steps.
Present your reasoning as a numbered list."""

#     # Experiment 1: More structured reasoning
reasoning_system = """You are a precise reasoning engine. For each problem:
1. Identify what is being asked
2. List known information
3. Break down the solution into logical steps
4. Verify each step before proceeding
Present your reasoning as a numbered list."""

#     # Experiment 2: Chain-of-thought emphasis
reasoning_system = """You are an expert at step-by-step reasoning. Think through problems carefully:
- State assumptions explicitly
- Show your work for calculations
- Explain why each step follows from the previous
- Double-check your logic
Format your response as a numbered list of reasoning steps."""

#     # Experiment 3: Domain-specific (math focus)
reasoning_system = """You are a mathematical reasoning assistant. When solving problems:
- Define all variables clearly
- Show all algebraic steps
- Verify calculations
- State the final answer clearly
Present reasoning as numbered steps."""
    
# User template experiments
reasoning_template = "{query}"
reasoning_template = "Think step-by-step about this problem: {query}"
reasoning_template = "Break down and solve: {query}"

set_reasoning_model(
    base_url="http://localhost:8000/v1",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    # model="Qwen/Qwen2.5-1.5B-Instruct",
    # model="TinyLlama/TinyLlama-1.1B-Chat-v0.3",
    temperature=0.7,
    max_tokens=512,
        use_symbolic_verification=True,
    use_factual_verification=False,
    max_reflection_iterations=2,
            reasoning_system_prompt=reasoning_system,
        reasoning_user_template=reasoning_template,
    

)

print(f"Query: {query}\n")

try:
    for chunk in enhance_stream(query):
        print(chunk, end='', flush=True)
    print()
except Exception as e:
    print(f"Error: {e}")

