"""Runtime orchestration and pipeline execution."""

import time
from typing import Dict, List, Optional

from kaelum.core.config import LLMConfig, MCPConfig
from kaelum.core.policy import PolicyController
from kaelum.core.reasoning import LLMClient, Message, ReasoningGenerator, ReasoningResult
from kaelum.core.reflection import Reflector, ReflectionEngine, Verifier
from kaelum.core.scoring import ConfidenceScorer, QualityMetrics
from kaelum.core.verification import VerificationEngine


class MCP:
    """Main MCP reasoning pipeline orchestrator."""

    def __init__(self, config: MCPConfig):
        """Initialize MCP with configuration."""
        self.config = config

        # Initialize LLM clients
        self.main_llm = LLMClient(config.llm)
        self.verifier_llm = LLMClient(config.verifier_llm)
        self.reflector_llm = LLMClient(config.reflector_llm)

        # Initialize pipeline components
        self.reasoning_generator = ReasoningGenerator(self.main_llm)
        self.verification_engine = VerificationEngine(
            use_symbolic=config.use_symbolic, use_rag=config.use_rag
        )
        self.verifier = Verifier(self.verifier_llm)
        self.reflector = Reflector(self.reflector_llm)
        self.reflection_engine = ReflectionEngine(
            self.verifier, self.reflector, max_iterations=config.max_reflection_iterations
        )
        self.confidence_scorer = ConfidenceScorer()
        self.quality_metrics = QualityMetrics()

        # Initialize policy controller
        self.policy_controller = (
            PolicyController(enable_learning=True)
            if config.enable_policy_controller
            else None
        )

        # Trace storage
        self.traces = []

    def infer(self, query: str, context: Optional[str] = None) -> ReasoningResult:
        """
        Run complete reasoning pipeline on a query.

        Args:
            query: The reasoning query
            context: Optional context

        Returns:
            ReasoningResult with verified reasoning
        """
        start_time = time.time()

        # Get policy for this query
        policy = (
            self.policy_controller.get_policy_for_query(query)
            if self.policy_controller
            else {}
        )

        # Step 1: Generate initial reasoning trace
        reasoning_trace = self.reasoning_generator.generate_reasoning(query, context)

        # Step 2: Symbolic and factual verification
        verification_results = self.verification_engine.verify_trace(reasoning_trace)

        # Step 3: Multi-LLM verification and reflection
        reflection_results = self.reflection_engine.reflect_and_repair(query, reasoning_trace)

        final_trace = reflection_results["final_trace"]
        final_verification = reflection_results["final_verification"]

        # Step 4: Generate final answer
        final_answer = self.reasoning_generator.generate_answer(query, final_trace)

        # Step 5: Compute confidence
        confidence = self.confidence_scorer.compute_confidence(
            verification_results, verification_results, final_verification
        )

        # Check against threshold
        verified = (
            confidence >= self.config.confidence_threshold
            and final_verification.get("valid", True)
        )

        # Build result
        result = ReasoningResult(
            final=final_answer,
            trace=final_trace,
            verified=verified,
            confidence=confidence,
            diagnostics={
                "initial_trace": reasoning_trace,
                "verification": verification_results,
                "reflection": reflection_results,
                "confidence": confidence,
                "policy": policy,
                "latency": time.time() - start_time,
            },
        )

        # Log trace if enabled
        if self.config.log_traces:
            self.traces.append(
                {
                    "query": query,
                    "result": result.model_dump(),
                    "timestamp": time.time(),
                }
            )

        # Update metrics
        self.quality_metrics.record_result(
            verified, confidence, len(reflection_results["iterations"])
        )

        # Update policy
        if self.policy_controller:
            self.policy_controller.update_policy(result.model_dump())

        return result

    def get_metrics(self) -> Dict:
        """Get quality metrics."""
        metrics = self.quality_metrics.get_metrics()

        if self.policy_controller:
            metrics["policy_state"] = self.policy_controller.get_state()

        return metrics

    def get_traces(self) -> List[Dict]:
        """Get logged reasoning traces."""
        return self.traces


class ModelRuntime:
    """Runtime that integrates MCP as a tool layer."""

    def __init__(self, llm: LLMClient):
        """Initialize model runtime with base LLM."""
        self.llm = llm
        self.tools = []
        self.mcp = None

    def attach(self, tool):
        """Attach a tool (like ReasoningMCPTool) to the runtime."""
        self.tools.append(tool)

        # If it's an MCP tool, store reference
        if hasattr(tool, "mcp"):
            self.mcp = tool.mcp

        return self

    def generate_content(self, query: str) -> str:
        """
        Generate content with optional tool usage.

        Args:
            query: User query

        Returns:
            Generated response
        """
        # If MCP is attached, use it for reasoning
        if self.mcp:
            result = self.mcp.infer(query)
            return f"{result.final}\n\n[Confidence: {result.confidence:.2f}]"

        # Otherwise, use base LLM
        messages = [Message(role="user", content=query)]
        return self.llm.generate(messages)

    def get_metrics(self) -> Optional[Dict]:
        """Get metrics from attached MCP."""
        if self.mcp:
            return self.mcp.get_metrics()
        return None
