"""Tests for MCP tool integration."""

import pytest
from kaelum.core.config import MCPConfig
from kaelum.tools.mcp_tool import ReasoningMCPTool


class TestReasoningMCPTool:
    """Tests for ReasoningMCPTool."""

    def test_initialization(self):
        """Test tool initialization."""
        config = MCPConfig()
        tool = ReasoningMCPTool(config)

        assert tool.name == "kaelum_reasoning"
        assert tool.description is not None
        assert tool.mcp is not None

    def test_tool_attributes(self):
        """Test tool has required attributes."""
        tool = ReasoningMCPTool(MCPConfig())

        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert hasattr(tool, "run")
        assert callable(tool.run)

    def test_run_with_valid_message(self):
        """Test running tool with valid message."""
        # This is a unit test that doesn't require actual LLM calls
        # We're testing the interface, not the implementation
        tool = ReasoningMCPTool(MCPConfig())

        messages = [{"role": "user", "content": "What is 2 + 2?"}]

        # The run method should return a dict
        # We won't test the actual result since it requires API keys
        assert callable(tool.run)

    def test_run_with_empty_messages(self):
        """Test running tool with empty messages."""
        tool = ReasoningMCPTool(MCPConfig())

        messages = []
        result = tool.run(messages)

        assert "error" in result
        assert result["verified"] is False

    def test_run_with_no_user_message(self):
        """Test running tool with no user message."""
        tool = ReasoningMCPTool(MCPConfig())

        messages = [{"role": "system", "content": "System message"}]
        result = tool.run(messages)

        assert "error" in result
        assert result["verified"] is False

    def test_callable_interface(self):
        """Test tool is callable."""
        tool = ReasoningMCPTool(MCPConfig())

        # Tool should be callable
        assert callable(tool)

    def test_get_metrics(self):
        """Test getting metrics from tool."""
        tool = ReasoningMCPTool(MCPConfig())

        metrics = tool.get_metrics()

        assert isinstance(metrics, dict)
        assert "total_requests" in metrics
        assert "verification_rate" in metrics


class TestLangChainAdapter:
    """Tests for LangChain adapter."""

    def test_adapter_exists(self):
        """Test that LangChain adapter is available."""
        from kaelum.tools.mcp_tool import LangChainAdapter

        assert LangChainAdapter is not None
        assert hasattr(LangChainAdapter, "create_tool")

    def test_create_tool_method(self):
        """Test create_tool method exists."""
        from kaelum.tools.mcp_tool import LangChainAdapter

        assert callable(LangChainAdapter.create_tool)


class TestLangGraphAdapter:
    """Tests for LangGraph adapter."""

    def test_adapter_exists(self):
        """Test that LangGraph adapter is available."""
        from kaelum.tools.mcp_tool import LangGraphAdapter

        assert LangGraphAdapter is not None
        assert hasattr(LangGraphAdapter, "create_node")

    def test_create_node_method(self):
        """Test create_node method exists."""
        from kaelum.tools.mcp_tool import LangGraphAdapter

        assert callable(LangGraphAdapter.create_node)
