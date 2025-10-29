"""
Example 2: Using KaelumAI with ModelRuntime

This example shows how to use the ModelRuntime interface.
"""

import os
from kaelum import ModelRuntime, LLMClient, LLMConfig, ReasoningMCPTool, MCPConfig

# Set API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Create LLM client
llm = LLMClient(LLMConfig(model="gpt-4o-mini"))

# Create MCP tool
mcp_tool = ReasoningMCPTool(MCPConfig())

# Create runtime and attach MCP tool
runtime = ModelRuntime(llm).attach(mcp_tool)

# Generate content with reasoning verification
query = "Explain how reinforcement learning can optimize a RAG retriever."
response = runtime.generate_content(query)

print("=" * 50)
print("QUERY:", query)
print("=" * 50)
print(f"\nRESPONSE:\n{response}\n")

# Get metrics
metrics = runtime.get_metrics()
if metrics:
    print("METRICS:")
    print(f"  - Total requests: {metrics['total_requests']}")
    print(f"  - Verification rate: {metrics['verification_rate']:.2%}")
    print(f"  - Avg confidence: {metrics['avg_confidence']:.2f}")
print("=" * 50)
