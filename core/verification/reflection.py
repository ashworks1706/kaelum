import re
from typing import List, Optional

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
        
        response = self.llm.generate(messages).strip()
        
        if not response:
            return None
        
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
        if not issues:
            return trace
        
        import logging
        logger = logging.getLogger("kaelum.reflection")
        
        problematic_indices = []
        for issue in issues:
            for match in re.finditer(r'\b(?:step|line)\s*(\d+)', issue.lower()):
                idx = int(match.group(1)) - 1
                if 0 <= idx < len(trace):
                    problematic_indices.append(idx)
        
        problematic_indices = sorted(set(problematic_indices))
        
        if not problematic_indices:
            max_issues_to_show = 2
            problematic_indices = list(range(max(0, len(trace) - max_issues_to_show), len(trace)))
        
        if not problematic_indices:
            return trace
        
        logger.info(f"REFLECTION: Fixing {len(problematic_indices)} problematic steps: {[i+1 for i in problematic_indices]}")
        
        improved_trace = trace.copy()
        
        for idx in problematic_indices:
            context_start = max(0, idx - 2)
            context_end = min(len(trace), idx + 2)
            context = trace[context_start:context_end]
            
            context_text = "\n".join(f"{context_start + i + 1}. {step}" 
                                    for i, step in enumerate(context))
            relevant_issues = [issue for issue in issues 
                             if str(idx + 1) in issue or "step" not in issue.lower()]
            issues_text = "\n".join(f"- {issue}" for issue in relevant_issues[:2])
            
            messages = [
                Message(role="system", content="Fix the problematic step. Return only the fixed step, no numbering."),
                Message(role="user", content=f"Query: {query}\n\nContext:\n{context_text}\n\nIssues with step {idx+1}:\n{issues_text}\n\nFixed step {idx+1}:"),
            ]
            
            response = self.llm.generate(messages).strip()
            
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', response)
            cleaned = re.sub(r'^[-•*]\s*', '', cleaned)
            cleaned = cleaned.strip()
            
            if cleaned and len(cleaned) > 10:
                improved_trace[idx] = cleaned
                logger.debug(f"REFLECTION: Fixed step {idx+1}")
        
        return improved_trace
