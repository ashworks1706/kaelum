"""Multi-LLM Verifier and Reflector architecture."""

from typing import Dict, List, Optional

from kaelum.core.reasoning import LLMClient, Message


class Verifier:
    """Independent LLM that verifies reasoning traces."""

    def __init__(self, llm_client: LLMClient):
        """Initialize verifier with an LLM client."""
        self.llm = llm_client

    def verify_trace(self, query: str, trace: List[str]) -> Dict[str, any]:
        """
        Verify a reasoning trace for logical consistency and correctness.

        Returns:
            Dictionary with verification results
        """
        trace_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(trace))

        system_prompt = """You are a critical reasoning verifier. Your job is to:
1. Check logical consistency between steps
2. Identify errors, contradictions, or unsupported claims
3. Rate the overall quality of the reasoning

Be thorough and critical. Point out even minor issues."""

        user_prompt = f"""Query: {query}

Reasoning trace to verify:
{trace_text}

Please analyze this reasoning and provide:
1. Is the reasoning logically sound? (Yes/No)
2. List any errors or issues found
3. Overall confidence score (0-1)

Format your response as:
VALID: [Yes/No]
ISSUES: [List any issues, or "None"]
CONFIDENCE: [0.0-1.0]"""

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        response = self.llm.generate(messages)

        # Parse response
        return self._parse_verification_response(response)

    def _parse_verification_response(self, response: str) -> Dict[str, any]:
        """Parse the verifier's response."""
        result = {"valid": True, "issues": [], "confidence": 0.8}

        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("VALID:"):
                valid_str = line.split(":", 1)[1].strip().lower()
                result["valid"] = valid_str in ["yes", "true"]
            elif line.startswith("ISSUES:"):
                issues_str = line.split(":", 1)[1].strip()
                if issues_str.lower() not in ["none", "no issues"]:
                    result["issues"].append(issues_str)
            elif line.startswith("CONFIDENCE:"):
                try:
                    conf_str = line.split(":", 1)[1].strip()
                    result["confidence"] = float(conf_str)
                except ValueError:
                    pass

        return result


class Reflector:
    """Independent LLM that repairs and improves reasoning."""

    def __init__(self, llm_client: LLMClient):
        """Initialize reflector with an LLM client."""
        self.llm = llm_client

    def repair_trace(
        self, query: str, trace: List[str], issues: List[str]
    ) -> List[str]:
        """
        Repair a reasoning trace based on identified issues.

        Returns:
            Repaired reasoning trace
        """
        if not issues:
            return trace

        trace_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(trace))
        issues_text = "\n".join(f"- {issue}" for issue in issues)

        system_prompt = """You are a reasoning repair specialist. Your job is to:
1. Fix errors and inconsistencies in reasoning
2. Improve logical flow
3. Ensure all steps are well-justified

Maintain the overall approach but fix the specific issues."""

        user_prompt = f"""Query: {query}

Current reasoning trace:
{trace_text}

Issues identified:
{issues_text}

Please provide an improved reasoning trace that addresses these issues.
Format as a numbered list of steps."""

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        response = self.llm.generate(messages)

        # Parse repaired trace
        repaired_trace = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
                step = line.lstrip("0123456789.-•) ").strip()
                if step:
                    repaired_trace.append(step)

        return repaired_trace if repaired_trace else trace


class ReflectionEngine:
    """Orchestrates verification and reflection cycles."""

    def __init__(self, verifier: Verifier, reflector: Reflector, max_iterations: int = 2):
        """Initialize reflection engine."""
        self.verifier = verifier
        self.reflector = reflector
        self.max_iterations = max_iterations

    def reflect_and_repair(
        self, query: str, initial_trace: List[str]
    ) -> Dict[str, any]:
        """
        Run verification and reflection cycles.

        Returns:
            Dictionary with final trace and diagnostics
        """
        current_trace = initial_trace
        iterations = []

        for iteration in range(self.max_iterations):
            # Verify current trace
            verification = self.verifier.verify_trace(query, current_trace)

            iterations.append(
                {
                    "iteration": iteration + 1,
                    "trace": current_trace.copy(),
                    "verification": verification,
                }
            )

            # If valid and high confidence, we're done
            if verification["valid"] and verification["confidence"] > 0.85:
                break

            # If we have issues and more iterations, repair
            if verification["issues"] and iteration < self.max_iterations - 1:
                current_trace = self.reflector.repair_trace(
                    query, current_trace, verification["issues"]
                )

        # Get final verification
        final_verification = self.verifier.verify_trace(query, current_trace)

        return {
            "final_trace": current_trace,
            "iterations": iterations,
            "final_verification": final_verification,
            "improved": len(iterations) > 1,
        }
