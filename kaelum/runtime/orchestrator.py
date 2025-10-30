"""Runtime orchestration and pipeline execution."""

import time
import hashlib
from typing import Dict, List, Optional

from kaelum.core.config import LLMConfig, MCPConfig
from kaelum.core.policy import PolicyController
from kaelum.core.reasoning import LLMClient, Message, ReasoningGenerator, ReasoningResult
from kaelum.core.reflection import Reflector, ReflectionEngine, Verifier
from kaelum.core.scoring import ConfidenceScorer, QualityMetrics
from kaelum.core.verification import VerificationEngine
from kaelum.core.cache import ReasoningCache


class MCP:
    """MCP Runtime Orchestrator - Streamlined reasoning acceleration."""

from typing import Dict, List, Optional

from kaelum.core.config import MCPConfig
from kaelum.core.reasoning import LLMClient, ReasoningGenerator
from kaelum.core.reflection import ReflectionEngine
from kaelum.core.scoring import ConfidenceScorer
from kaelum.core.verification import VerificationEngine
from kaelum.core.cache import ReasoningCache


class MCP:
    """Modular Cognitive Processor - Accelerates LLM reasoning."""

    def __init__(self, config: MCPConfig):
        """Initialize MCP with configuration."""
        self.config = config

        # Single LLM for all reasoning tasks (cost-efficient)
        self.llm = LLMClient(config.llm)
        
        # Core components
        self.generator = ReasoningGenerator(self.llm)
        self.reflection = ReflectionEngine(self.llm, max_iterations=config.max_reflection_iterations)
        self.verification = VerificationEngine(
            symbolic_enabled=config.use_symbolic_verification,
            factual_enabled=True,  # Lightweight pattern matching
        )
        self.scorer = ConfidenceScorer(
            symbolic_weight=0.3,
            factual_weight=0.3,
            verifier_weight=0.4,
        )
        self.cache = ReasoningCache()

    def infer(self, query: str, use_cache: bool = True) -> Dict:
        """
        Run reasoning acceleration pipeline.

        Args:
            query: Input query
            use_cache: Whether to use caching (default: True)

        Returns:
            Enhanced reasoning with confidence score
        """
        # Check cache
        if use_cache:
            cached = self.cache.get(query, str(self.config))
            if cached:
                cached["cache_hit"] = True
                return cached

        # Generate initial reasoning with CoT
        initial = self.generator.generate_reasoning(query)

        # Enhance through reflection cycles
        reflection = self.reflection.enhance_reasoning(query, initial["trace"])

        # Verify final trace
        verification = self.verification.verify(query, reflection["final_trace"])

        # Compute confidence
        confidence = self.scorer.compute_confidence(
            symbolic_score=verification.get("symbolic_score", 0.5),
            factual_score=verification.get("factual_score", 0.5),
            verifier_confidence=reflection["final_verification"]["confidence"],
        )

        result = {
            "query": query,
            "trace": reflection["final_trace"],
            "confidence": confidence,
            "verification": verification,
            "iterations": len(reflection["iterations"]),
            "improved": reflection["improved"],
            "cache_hit": False,
        }
        
        # Cache high-confidence results
        if use_cache and confidence >= self.config.confidence_threshold:
            self.cache.set(query, str(self.config), result)

        return result


class ModelRuntime:
    """Runtime wrapper for MCP integration with agent frameworks."""

    def __init__(self, mcp: MCP):
        """Initialize runtime with MCP instance."""
        self.mcp = mcp

    def __call__(self, query: str) -> str:
        """Execute MCP and return formatted result."""
        result = self.mcp.infer(query)

        trace_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(result["trace"]))

        return f"""Reasoning:
{trace_text}

Confidence: {result['confidence']:.2f}
Iterations: {result['iterations']}
"""

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

        # Check cache first
        cached = self.cache.get(query, self.config_hash)
        if cached:
            print(f"Cache hit for query (latency: {(time.time() - start_time)*1000:.0f}ms)")
            result = ReasoningResult(**cached)
            result.diagnostics["cached"] = True
            result.diagnostics["latency"] = time.time() - start_time
            return result

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
                "cached": False,
            },
        )

        # Cache result
        self.cache.set(query, result.model_dump(), self.config_hash)

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
    
    def _generate_config_hash(self) -> str:
        """Generate a hash of the config for cache keys."""
        config_str = f"{self.config.llm.model}_{self.config.llm.temperature}_{self.config.confidence_threshold}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def get_metrics(self) -> Dict:
        """Get quality metrics."""
        metrics = self.quality_metrics.get_metrics()

        if self.policy_controller:
            metrics["policy_state"] = self.policy_controller.get_state()
        
        # Add cache stats
        metrics["cache_stats"] = self.cache.get_stats()

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
