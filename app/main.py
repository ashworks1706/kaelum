"""FastAPI application for KaelumAI reasoning service."""

from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kaelum.core.config import LLMConfig, MCPConfig
from kaelum.runtime.orchestrator import MCP

# Initialize FastAPI app
app = FastAPI(
    title="KaelumAI Reasoning API",
    description="The All-in-One Reasoning Layer for Agentic LLMs",
    version="0.1.0",
)

# Global MCP instance (will be initialized on first request or at startup)
mcp_instance: Optional[MCP] = None


def get_mcp() -> MCP:
    """Get or create MCP instance."""
    global mcp_instance
    if mcp_instance is None:
        # Initialize with default config
        config = MCPConfig()
        mcp_instance = MCP(config)
    return mcp_instance


# Request/Response models
class VerifyReasoningRequest(BaseModel):
    """Request model for reasoning verification."""

    query: str = Field(description="The reasoning query to verify")
    reasoning_trace: Optional[List[str]] = Field(
        default=None, description="Optional pre-generated reasoning trace"
    )
    context: Optional[str] = Field(default=None, description="Optional context")
    config: Optional[Dict] = Field(
        default=None, description="Optional MCP configuration overrides"
    )


class VerifyReasoningResponse(BaseModel):
    """Response model for reasoning verification."""

    verified: bool = Field(description="Whether reasoning was verified")
    confidence: float = Field(description="Confidence score (0-1)")
    final_answer: str = Field(description="Final reasoning answer")
    trace: List[str] = Field(description="Reasoning trace steps")
    diagnostics: Dict = Field(default_factory=dict, description="Diagnostic information")


class MetricsResponse(BaseModel):
    """Response model for metrics."""

    total_requests: int
    verified_count: int
    failed_count: int
    verification_rate: float
    avg_confidence: float
    avg_iterations: float
    policy_state: Optional[Dict] = None


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "KaelumAI Reasoning API",
        "version": "0.1.0",
        "endpoints": {
            "verify_reasoning": "/verify_reasoning",
            "metrics": "/metrics",
            "health": "/health",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "kaelum-reasoning"}


@app.post("/verify_reasoning", response_model=VerifyReasoningResponse)
async def verify_reasoning(request: VerifyReasoningRequest):
    """
    Verify and improve reasoning for a query.

    This endpoint:
    1. Generates or uses provided reasoning trace
    2. Performs symbolic and factual verification
    3. Uses multi-LLM verification and reflection
    4. Computes confidence scores
    5. Returns verified reasoning with diagnostics
    """
    try:
        mcp = get_mcp()

        # Run MCP inference
        result = mcp.infer(request.query, context=request.context)

        # Build response
        return VerifyReasoningResponse(
            verified=result.verified,
            confidence=result.confidence,
            final_answer=result.final,
            trace=result.trace,
            diagnostics=result.diagnostics,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reasoning verification failed: {str(e)}")


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get reasoning quality metrics.

    Returns aggregated metrics including:
    - Total requests processed
    - Verification success rate
    - Average confidence scores
    - Policy controller state (if enabled)
    """
    try:
        mcp = get_mcp()
        metrics = mcp.get_metrics()

        return MetricsResponse(**metrics)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics: {str(e)}")


@app.get("/traces")
async def get_traces(limit: int = 10):
    """
    Get recent reasoning traces.

    Args:
        limit: Maximum number of traces to return
    """
    try:
        mcp = get_mcp()
        traces = mcp.get_traces()

        # Return most recent traces
        return {"traces": traces[-limit:], "total": len(traces)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve traces: {str(e)}")


@app.post("/configure")
async def configure_mcp(config: Dict):
    """
    Update MCP configuration.

    Note: This creates a new MCP instance with the new configuration.
    """
    try:
        global mcp_instance

        # Create new config from dict
        mcp_config = MCPConfig(**config)

        # Create new MCP instance
        mcp_instance = MCP(mcp_config)

        return {"status": "configured", "config": mcp_config.model_dump()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
