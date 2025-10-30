"""
Example 4: LangChain Integration

This example demonstrates using KaelumAI with LangChain.
"""

import os

# Uncomment when LangChain is installed

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from kaelum import MCPConfig
from kaelum.tools.mcp_tool import LangChainAdapter

# Set API key
os.environ["GEMINI_API_KEY"] = "your-api-key-here"

# Create base LLM
base_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# Create KaelumAI tool
config = MCPConfig(use_symbolic=True, confidence_threshold=0.75)
reasoning_tool = LangChainAdapter.create_tool(config)

# Initialize agent with reasoning tool
agent = initialize_agent(
    [reasoning_tool],
    base_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run agent
query = "If a train travels 120 km in 2 hours, then speeds up and travels 180 km in 2 hours, what is the average speed?"
response = agent.run(query)

print("=" * 60)
print("AGENT RESPONSE:")
print(response)
print("=" * 60)


# Demo placeholder
print("LangChain Integration Example")
print("=" * 60)
print("To run this example, install LangChain:")
print("  pip install langchain langchain-google-genai")
print("\nThen uncomment the code in this file.")
print("=" * 60)
