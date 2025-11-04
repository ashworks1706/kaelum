from typing import List
from core.reasoning import LLMClient, Message


class ReflectionEngine:
    def __init__(self, llm_client: LLMClient, max_iterations: int = 2):
        self.llm = llm_client
        self.max_iterations = max_iterations

    def enhance_reasoning(self, query: str, initial_trace: List[str], verification_issues: List[str] = None) -> List[str]:
        current_trace = initial_trace
        
        # If specific verification issues provided, use them for targeted improvement
        if verification_issues and len(verification_issues) > 0:
            current_trace = self._improve_trace(query, current_trace, verification_issues)
        else:
            # Otherwise do general reflection
            for iteration in range(self.max_iterations):
                issues = self._verify_trace(query, current_trace)
                
                if not issues:
                    break
                
                if iteration < self.max_iterations - 1:
                    current_trace = self._improve_trace(query, current_trace, issues)
        
        return current_trace
    
    def _verify_trace(self, query: str, trace: List[str]) -> List[str]:
        trace_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(trace))
        
        messages = [
            Message(role="system", content="You are a critical reasoning verifier. List any logical errors or gaps."),
            Message(role="user", content=f"Query: {query}\n\nReasoning:\n{trace_text}\n\nList issues (or 'None'):"),
        ]
        
        response = self.llm.generate(messages)
        
        if "none" in response.lower() or not response.strip():
            return []
        
        return [response.strip()]
    
    def _improve_trace(self, query: str, trace: List[str], issues: List[str]) -> List[str]:
        trace_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(trace))
        issues_text = "\n".join(f"- {issue}" for issue in issues)
        
        messages = [
            Message(role="system", content="Fix errors in reasoning."),
            Message(role="user", content=f"Query: {query}\n\nReasoning:\n{trace_text}\n\nIssues:\n{issues_text}\n\nImproved reasoning:"),
        ]
        
        response = self.llm.generate(messages)
        
        # Parse improved trace
        improved = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                step = line.lstrip("0123456789.-•) ").strip()
                if step:
                    improved.append(step)
        
        return improved if improved else trace
