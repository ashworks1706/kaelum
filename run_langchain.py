"""KaelumAI + LangChain - Works with Gemini or Ollama"""

import os
from kaelum import set_reasoning_model
from kaelum.integrations import create_kaelum_tool

try:
    from langchain.agents import AgentType, initialize_agent
    from langchain.memory import ConversationBufferMemory
except ImportError:
    print("❌ Install: pip install langchain langchain-google-genai langchain-community")
    exit(1)

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
print("✓ Kaelum ready (qwen2.5:7b)")

USE_GEMINI = True

if USE_GEMINI:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Set GOOGLE_API_KEY or GEMINI_API_KEY")
        print("Or set USE_GEMINI = False to use local Ollama")
        exit(1)
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.7)
        print("✓ Using Gemini 1.5 Flash")
    except ImportError:
        print("❌ Install: pip install langchain-google-genai")
        exit(1)
else:
    try:
        from langchain_community.llms import Ollama
        llm = Ollama(model="llama3.2:3b")
        print("✓ Using Ollama (llama3.2:3b)")
    except ImportError:
        print("❌ Install: pip install langchain-community")
        exit(1)

kaelum_tool = create_kaelum_tool()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=[kaelum_tool],
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
)

print("\n" + "=" * 70)
query = "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?"
print(f"Question: {query}\n")

try:
    response = agent.run(query)
    print("\n" + "=" * 70)
    print("Answer:", response)
except Exception as e:
    print(f"Error: {e}")

