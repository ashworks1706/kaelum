"""
Example 1: Basic reasoning with KaelumAI

This example demonstrates the simplest usage of KaelumAI.
"""

import os
from kaelum import MCP, MCPConfig

# Set API key (or use environment variable)
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Create MCP instance with default configuration
mcp = MCP(MCPConfig())

# Run reasoning on a query
result = mcp.infer("If 3x + 5 = 11, what is x?")

print("=" * 50)
print("QUERY: If 3x + 5 = 11, what is x?")
print("=" * 50)
print(f"\nFINAL ANSWER:\n{result.final}\n")
print(f"VERIFIED: {result.verified}")
print(f"CONFIDENCE: {result.confidence:.2f}\n")
print("REASONING TRACE:")
for i, step in enumerate(result.trace, 1):
    print(f"{i}. {step}")
print("\n" + "=" * 50)
