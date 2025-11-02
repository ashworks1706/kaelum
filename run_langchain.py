"""KaelumAI + LangChain - Simple Integration Demo

Start vLLM server first (for 6GB GPU):
    python -m vllm.entrypoints.openai.api_server \
        --model TinyLlama/TinyLlama-1.1B-Chat-v0.3 \
        --port 8000 \
        --gpu-memory-utilization 0.7 \
        --max-num-seqs 32 \
        --max-model-len 1024 \
        --chat-template "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}assistant: "
"""

from kaelum import set_reasoning_model, kaelum_enhance_reasoning

try:
    from langchain_community.llms import VLLM
except ImportError:
    print("❌ Install: pip install langchain-community")
    exit(1)

set_reasoning_model(
    base_url="http://localhost:8000/v1",
    model="TinyLlama/TinyLlama-1.1B-Chat-v0.3",
    temperature=0.3,
    max_tokens=512,
    max_reflection_iterations=0,
    use_symbolic_verification=True,
    use_factual_verification=False,
    rag_adapter=None,
)

# Use OpenAI-compatible endpoint with LangChain
from langchain_community.llms import OpenAI
llm = OpenAI(
    base_url="http://localhost:8000/v1",
    model="TinyLlama/TinyLlama-1.1B-Chat-v0.3",
    temperature=0.7,
    api_key="EMPTY"
)

query = "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?"
print(f"Question: {query}\n")

print("🧠 Step 1: Kaelum generates reasoning steps...")
reasoning = kaelum_enhance_reasoning(query, domain="logic")

print("\n✓ Reasoning steps:")
for i, step in enumerate(reasoning["reasoning_steps"], 1):
    print(f"   {i}. {step}")

prompt = f"""Based on this step-by-step reasoning:

{chr(10).join(f"{i}. {step}" for i, step in enumerate(reasoning["reasoning_steps"], 1))}

Question: {query}

Provide a clear, comprehensive final answer:"""

print("\n📝 Step 2: LangChain LLM formulates final answer...\n")
print("=" * 70)

try:
    response = llm.invoke(prompt)
    print(response)
except Exception as e:
    print(f"Error: {e}")





