"""ReasoningMCPTool for LangChain and framework integration."""

from typing import Any, Dict, List, Optional

from kaelum.core.config import MCPConfig
from kaelum.core.reasoning import Message
from kaelum.runtime.orchestrator import MCP


class ReasoningMCPTool:
    """MCP tool adapter for LangChain and other agent frameworks."""

    name: str = "kaelum_reasoning"
    description: str = """Verifies and corrects reasoning traces using multi-LLM verification.
    Use this tool when you need to ensure logical correctness and consistency in complex reasoning.
    Input should be a query or reasoning trace to verify."""

    def __init__(self, config: MCPConfig):
        """Initialize ReasoningMCPTool with MCP configuration."""
        self.config = config
        self.mcp = MCP(config)

    def run(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Run reasoning verification on messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Dictionary with verification results
        """
        # Extract query from messages
        user_messages = [m for m in messages if m.get("role") == "user"]

        if not user_messages:
            return {
                "error": "No user message found",
                "verified": False,
                "confidence": 0.0,
            }

        query = user_messages[-1]["content"]

        # Run MCP inference
        result = self.mcp.infer(query)

        # Return structured result
        return {
            "final": result.final,
            "verified": result.verified,
            "confidence": result.confidence,
            "trace": result.trace,
            "diagnostics": result.diagnostics,
        }

    def __call__(self, query: str) -> str:
        """
        Callable interface for simple usage.

        Args:
            query: Query string

        Returns:
            Final answer
        """
        messages = [{"role": "user", "content": query}]
        result = self.run(messages)
        return result.get("final", "")

    def get_metrics(self) -> Dict:
        """Get reasoning metrics."""
        return self.mcp.get_metrics()


class LangChainAdapter:
    """Adapter for LangChain Tool integration."""

    @staticmethod
    def create_tool(config: MCPConfig):
        """
        Create a LangChain Tool from ReasoningMCPTool.

        Args:
            config: MCP configuration

        Returns:
            LangChain Tool instance
        """
        reasoning_mcp = ReasoningMCPTool(config)

        try:
            from langchain.tools import Tool

            return Tool(
                name=reasoning_mcp.name,
                func=lambda q: reasoning_mcp(q),
                description=reasoning_mcp.description,
            )
        except ImportError:
            raise ImportError(
                "LangChain not installed. Install with: pip install langchain"
            )


class LangGraphAdapter:
    """Adapter for LangGraph node integration."""

    @staticmethod
    def create_node(config: MCPConfig):
        """
        Create a LangGraph node from ReasoningMCPTool.

        Args:
            config: MCP configuration

        Returns:
            Function that can be used as a LangGraph node
        """
        reasoning_mcp = ReasoningMCPTool(config)

        def reasoning_node(state: Dict) -> Dict:
            """LangGraph node for reasoning verification."""
            query = state.get("query", "")
            if not query:
                return state

            result = reasoning_mcp(query)

            # Update state with verification results
            return {
                **state,
                "reasoning_result": result,
                "verified": reasoning_mcp.mcp.infer(query).verified,
            }

        return reasoning_node
