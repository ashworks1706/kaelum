import re
from typing import List, Optional
from transformers import pipeline

from ..reasoning import LLMClient, Message


class ReflectionEngine:
    def __init__(self, llm_client: LLMClient, verification_engine=None, max_iterations: int = 2):
        self.llm = llm_client
        self.verification_engine = verification_engine
        self.max_iterations = max_iterations

    def enhance_reasoning(self, query: str, initial_trace: List[str], 
                         worker_type: Optional[str] = None,
                         verification_issues: List[str] = None) -> List[str]:
        import logging
        logger = logging.getLogger("kaelum.reflection")
        
        current_trace = initial_trace
        
        if verification_issues and len(verification_issues) > 0:
            logger.info(f"\n{'─' * 80}")
            logger.info(f"REFLECTION: Starting improvement with {len(verification_issues)} known issues")
            for i, issue in enumerate(verification_issues[:3], 1):
                logger.info(f"  Issue {i}: {issue}")
            if len(verification_issues) > 3:
                logger.info(f"  ... and {len(verification_issues) - 3} more")
            
            current_trace = self._improve_trace(query, current_trace, verification_issues)
            logger.info(f"REFLECTION: Trace improved ({len(initial_trace)} → {len(current_trace)} steps)")
            logger.info(f"{'─' * 80}\n")
        else:
            logger.info(f"\n{'─' * 80}")
            logger.info(f"REFLECTION: Starting iterative improvement (max {self.max_iterations} iterations)")
            
            for iteration in range(self.max_iterations):
                logger.info(f"\nREFLECTION: Iteration {iteration + 1}/{self.max_iterations}")
                
                if self.verification_engine and worker_type:
                    fake_answer = " ".join(current_trace[-2:]) if len(current_trace) >= 2 else ""
                    
                    result = self.verification_engine.verify(
                        query=query,
                        reasoning_steps=current_trace,
                        answer=fake_answer,
                        worker_type=worker_type
                    )
                    
                    if result["passed"]:
                        logger.info(f"REFLECTION: ✓ Verification passed on iteration {iteration + 1}")
                        break
                    
                    issues = result.get("issues", [])
                    if issues and iteration < self.max_iterations - 1:
                        logger.info(f"REFLECTION: Found {len(issues)} issues, attempting improvement...")
                        current_trace = self._improve_trace(query, current_trace, issues)
                        logger.info(f"REFLECTION: Trace updated ({len(current_trace)} steps)")
                else:
                    issues = self._verify_trace(query, current_trace)
                    
                    if not issues:
                        logger.info(f"REFLECTION: ✓ No issues found on iteration {iteration + 1}")
                        break
                    
                    if iteration < self.max_iterations - 1:
                        logger.info(f"REFLECTION: Found issues, attempting improvement...")
                        current_trace = self._improve_trace(query, current_trace, issues)
            
            logger.info(f"REFLECTION: Complete ({len(initial_trace)} → {len(current_trace)} steps)")
            logger.info(f"{'─' * 80}\n")
        
        return current_trace
    
    def _verify_trace(self, query: str, trace: List[str]) -> List[str]:
        trace_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(trace))
        
        messages = [
            Message(role="system", content="You are a critical reasoning verifier. List any logical errors or gaps."),
            Message(role="user", content=f"Query: {query}\n\nReasoning:\n{trace_text}\n\nList issues (or 'None'):"),
        ]
        
        
        response = self.llm_client.generate(messages).strip()
        
        if not response or not response.strip():
            return None
        
        try:
            nli = pipeline("text-classification", model="facebook/bart-large-mnli")
            no_improvement_labels = [
                "No improvements are needed",
                "The answer is correct as is",
                "Everything looks good"
            ]
            
            for label in no_improvement_labels:
                result = nli(f"{response}", f"hypothesis: {label}")
                if result and result[0]['label'] == 'ENTAILMENT' and result[0]['score'] > 0.75:
                    return None
        except:
            if "none" in response.lower() or "no improvement" in response.lower():
                words_before = response.lower().split("none")[0] if "none" in response.lower() else ""
                if not any(neg in words_before for neg in ["not", "no", "isn't", "aren't"]):
                    return None
        
        improvements = self._parse_structured_list(response)
        
        return improvements if improvements else None
    
    def _parse_structured_list(self, text: str) -> List[str]:
        patterns = [
            r'^\d+[\.\)]\s+(.+)',
            r'^[-•*]\s+(.+)',
            r'^[a-zA-Z][\.\)]\s+(.+)',
            r'^(?:Step\s+\d+:)\s*(.+)'
        ]
        
        items = []
        current_item = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            is_new_item = False
            matched_content = None
            
            for pattern in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    is_new_item = True
                    matched_content = match.group(1)
                    break
            
            if is_new_item:
                if current_item:
                    items.append(' '.join(current_item))
                current_item = [matched_content]
            else:
                current_item.append(line)
        
        if current_item:
            items.append(' '.join(current_item))
        
        return items
    
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
