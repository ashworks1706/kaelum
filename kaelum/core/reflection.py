"""Self-reflection and reasoning improvement engine."""

from typing import Dict, List, Optional

from kaelum.core.reasoning import LLMClient, Message


class ReflectionEngine:
    """Enhances reasoning through self-reflection and iterative improvement."""

    def __init__(self, llm_client: LLMClient, max_iterations: int = 2):
        """Initialize reflection engine."""
        self.llm = llm_client
        self.max_iterations = max_iterations

    def enhance_reasoning(self, query: str, initial_trace: List[str]) -> Dict:
        """
        Enhance reasoning through reflection cycles.
        
        Returns:
            Dictionary with improved trace and diagnostics
        """
        current_trace = initial_trace
        iterations = []
        
        for iteration in range(self.max_iterations):
            # Verify current trace
            verification = self._verify_trace(query, current_trace)
            
            iterations.append({
                "iteration": iteration + 1,
                "trace": current_trace.copy(),
                "verification": verification,
            })
            
            # Stop if valid and high confidence
            if verification["valid"] and verification["confidence"] > 0.85:
                break
            
            # Improve trace if issues found
            if verification["issues"] and iteration < self.max_iterations - 1:
                current_trace = self._improve_trace(query, current_trace, verification["issues"])
        
        # Final verification
        final_verification = self._verify_trace(query, current_trace)
        
        return {
            "final_trace": current_trace,
            "iterations": iterations,
            "final_verification": final_verification,
            "improved": len(iterations) > 1,
        }
    
    def _verify_trace(self, query: str, trace: List[str]) -> Dict:
        """Verify reasoning trace for logical consistency."""
        trace_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(trace))
        
        system_prompt = """You are a critical reasoning verifier. Analyze the reasoning for:
1. Logical consistency between steps
2. Correctness of conclusions
3. Any errors or gaps

Be thorough but concise."""
        
        user_prompt = f"""Query: {query}

Reasoning trace:
{trace_text}

Provide:
VALID: [Yes/No]
ISSUES: [List issues or "None"]
CONFIDENCE: [0.0-1.0]"""
        
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]
        
        response = self.llm.generate(messages)
        return self._parse_verification(response)
    
    def _improve_trace(self, query: str, trace: List[str], issues: List[str]) -> List[str]:
        """Improve reasoning trace based on identified issues."""
        if not issues:
            return trace
        
        trace_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(trace))
        issues_text = "\n".join(f"- {issue}" for issue in issues)
        
        system_prompt = """You are a reasoning improvement specialist. Fix errors and improve logical flow while maintaining the overall approach."""
        
        user_prompt = f"""Query: {query}

Current reasoning:
{trace_text}

Issues to fix:
{issues_text}

Provide an improved reasoning trace as a numbered list."""
        
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]
        
        response = self.llm.generate(messages)
        
        # Parse improved trace
        improved_trace = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                step = line.lstrip("0123456789.-•) ").strip()
                if step:
                    improved_trace.append(step)
        
        return improved_trace if improved_trace else trace
    
    def _parse_verification(self, response: str) -> Dict:
        """Parse verification response."""
        result = {"valid": True, "issues": [], "confidence": 0.8}
        
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("VALID:"):
                result["valid"] = "yes" in line.lower() or "true" in line.lower()
            elif line.startswith("ISSUES:"):
                issues_str = line.split(":", 1)[1].strip()
                if issues_str.lower() not in ["none", "no issues"]:
                    result["issues"].append(issues_str)
            elif line.startswith("CONFIDENCE:"):
                try:
                    result["confidence"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
        
        return result
