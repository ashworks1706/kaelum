"""MCP protocol handlers and adapters."""

from typing import Any, Dict


class MCPProtocolHandler:
    """Handler for MCP protocol messages."""

    PROTOCOL_VERSION = "0.1"

    @staticmethod
    def create_request(method: str, params: Dict[str, Any]) -> Dict:
        """Create an MCP protocol request."""
        return {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1,
        }

    @staticmethod
    def create_response(result: Any, request_id: int = 1) -> Dict:
        """Create an MCP protocol response."""
        return {
            "jsonrpc": "2.0",
            "result": result,
            "id": request_id,
        }

    @staticmethod
    def create_error(code: int, message: str, request_id: int = 1) -> Dict:
        """Create an MCP protocol error response."""
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": code,
                "message": message,
            },
            "id": request_id,
        }


class MCPAdapter:
    """Adapter for integrating KaelumAI with MCP-compatible systems."""

    def __init__(self, mcp_instance):
        """Initialize MCP adapter."""
        self.mcp = mcp_instance
        self.handler = MCPProtocolHandler()

    def handle_request(self, request: Dict) -> Dict:
        """
        Handle an MCP protocol request.

        Args:
            request: MCP protocol request

        Returns:
            MCP protocol response
        """
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id", 1)

        try:
            if method == "verify_reasoning":
                result = self._verify_reasoning(params)
                return self.handler.create_response(result, request_id)

            elif method == "get_metrics":
                result = self._get_metrics()
                return self.handler.create_response(result, request_id)

            else:
                return self.handler.create_error(
                    -32601, f"Method not found: {method}", request_id
                )

        except Exception as e:
            return self.handler.create_error(-32603, f"Internal error: {str(e)}", request_id)

    def _verify_reasoning(self, params: Dict) -> Dict:
        """Handle verify_reasoning method."""
        query = params.get("query")
        if not query:
            raise ValueError("Missing required parameter: query")

        context = params.get("context")
        result = self.mcp.infer(query, context)

        return {
            "verified": result.verified,
            "confidence": result.confidence,
            "final_answer": result.final,
            "trace": result.trace,
            "diagnostics": result.diagnostics,
        }

    def _get_metrics(self) -> Dict:
        """Handle get_metrics method."""
        return self.mcp.get_metrics()
