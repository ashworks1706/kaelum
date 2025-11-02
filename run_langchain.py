"""KaelumAI + LangChain - Open Source Local Models"""

import os
from kaelum import set_reasoning_model
from kaelum.integrations import create_kaelum_tool

from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama

set_reasoning_model(
    provider="ollama",
    model="qwen2.5:7b",
    temperature=0.3,
    max_tokens=512,
    max_reflection_iterations=0,
)
print("✓ Kaelum ready (qwen2.5:7b)")

llm = Ollama(model="qwen2.5:7b")
print("✓ LangChain using Ollama (qwen2.5:7b)")

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


