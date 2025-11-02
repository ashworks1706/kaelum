"""KaelumAI + LangChain - Simple Integration Demo"""

from kaelum import set_reasoning_model, kaelum_enhance_reasoning

try:
    from langchain_community.llms import Ollama
except ImportError:
    print("❌ Install: pip install langchain-community")
    exit(1)

set_reasoning_model(
    base_url="http://localhost:11434/v1",
    model="qwen2.5:7b",
    temperature=0.3,
    max_tokens=512,
    max_reflection_iterations=0,
    use_symbolic_verification=True,
    use_factual_verification=False,
    rag_adapter=None,
)


llm = Ollama(model="qwen2.5:7b", temperature=0.7)

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





