
import re
import ast
import time
from typing import Optional, Dict, List, Any

from core.workers import WorkerAgent, WorkerResult, WorkerSpecialty
from core.tree_cache import TreeCache
from core.config import KaelumConfig
from core.reasoning import Message
from core.lats import LATS, LATSNode
from core.reward_model import RewardModel
from core.language_detector import LanguageDetector
from core.task_classifier import TaskClassifier
from core.code_extractor import CodeExtractor
from core.adaptive_penalty import AdaptivePenalty


class CodeWorker(WorkerAgent):
    
    def __init__(self, config: Optional[KaelumConfig] = None, tree_cache: Optional[TreeCache] = None):
        super().__init__(config, tree_cache)
        self.language_detector = LanguageDetector()
        self.task_classifier = TaskClassifier()
        self.code_extractor = CodeExtractor()
        self.supported_languages = {
            'python', 'javascript', 'typescript', 'java', 'cpp', 'c',
            'go', 'rust', 'ruby', 'php', 'swift', 'kotlin'
        }
    
    def get_specialty(self) -> WorkerSpecialty:
        return WorkerSpecialty.CODE
    
    def can_handle(self, query: str, context: Optional[Dict] = None) -> float:
        return 1.0
    
    def solve(self, query: str, context: Optional[Dict] = None,
              use_cache: bool = True, max_tree_depth: int = 5,
              num_simulations: int = 10, parallel: bool = False) -> WorkerResult:
        start_time = time.time()
        
        if use_cache:
            cached_result = self._check_cache(query)
            if cached_result:
                return cached_result
        
        language = self._detect_language(query)
        task_type = self._classify_task(query)
        
        root_state = {
            "query": query,
            "step": f"Analyzing {task_type} task for {language or 'unspecified'} language",
            "depth": 0,
            "language": language,
            "task_type": task_type,
            "code_parts": []
        }
        
        def simulate_code_step(node: LATSNode) -> float:
            state = node.state
            depth = state.get("depth", 0)
            
            if "code" in state:
                code = state["code"]
                if code and language == 'python':
                    syntax_valid = self._validate_python_syntax(code)
                    has_answer = len(code) > 20
                    return RewardModel.get_reward("code", state, depth, 
                                                 has_answer=has_answer, syntax_valid=syntax_valid)
                elif code and len(code) > 20:
                    return RewardModel.get_reward("code", state, depth, has_answer=True)
                return RewardModel.get_reward("code", state, depth)
            
            has_partial = len(state.get("code_parts", [])) > 0
            return RewardModel.get_reward("code", state, depth, has_partial=has_partial)
        
        def expand_code_step(parent_node: LATSNode) -> Dict[str, Any]:
            parent_state = parent_node.state
            depth = parent_state.get("depth", 0)
            
            history = []
            node = parent_node
            while node.parent is not None:
                if "step" in node.state:
                    history.insert(0, node.state["step"])
                node = node.parent
            
            prompt = self._build_prompt(query, language, task_type, context)
            if history:
                prompt += "\n\nPrevious steps:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(history))
                prompt += "\n\nWhat is the next implementation step?"
            
            try:
                messages = [
                    Message(role="system", content=self.get_system_prompt()),
                    Message(role="user", content=prompt)
                ]
                response = self.llm_client.generate(messages)
                next_step = response.strip()
                
                code = self._extract_code(next_step)
                has_code = code is not None and len(code) > 10
                is_final = depth >= max_tree_depth - 1 or has_code
                
                return {
                    "query": query,
                    "step": next_step,
                    "depth": depth + 1,
                    "language": language,
                    "task_type": task_type,
                    "code_parts": parent_state.get("code_parts", []) + ([next_step] if not is_final else []),
                    "code": code if is_final else None
                }
            except:
                return {
                    "query": query,
                    "step": f"Implementation step {depth + 1}",
                    "depth": depth + 1,
                    "language": language,
                    "task_type": task_type,
                    "code_parts": parent_state.get("code_parts", [])
                }
        
        tree = LATS(root_state, simulator=simulate_code_step, expand_fn=expand_code_step)
        
        tree.run_simulations(num_simulations, max_tree_depth, parallel=parallel)
        
        best_node = tree.best_child()
        if best_node is None:
            best_node = tree.root
        
        reasoning_steps = []
        node = best_node
        while node is not None:
            if node.state.get("step") and node != tree.root:
                reasoning_steps.insert(0, node.state["step"])
            node = node.parent
        
        code = best_node.state.get("code")
        if not code and reasoning_steps:
            code = self._extract_code(reasoning_steps[-1])
        
        answer = reasoning_steps[-1] if reasoning_steps else ""
        syntax_valid = True
        if language == 'python' and code:
            syntax_valid = self._validate_python_syntax(code)
        
        confidence = RewardModel.compute_confidence(
            best_node.value if best_node else 0.0,
            best_node.visits if best_node else 0,
            tree.root.visits
        )
        execution_time = time.time() - start_time
        
        if use_cache:
            self.tree_cache.store(query, tree, self.get_specialty().value,
                                 False, confidence)
        
        return WorkerResult(
            answer=answer,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            verification_passed=False,
            specialty=self.get_specialty(),
            execution_time=execution_time,
            metadata={
                'language': language,
                'task_type': task_type,
                'syntax_valid': syntax_valid,
                'num_simulations': num_simulations,
                'tree_depth': best_node.state.get("depth", 0),
                'cache_hit': False
            }
        )
    
    def _detect_language(self, query: str) -> Optional[str]:
        result = self.language_detector.detect(query, "")
        return result['language'] if result['confidence'] > 0.3 else None
    
    
    def _classify_task(self, query: str) -> str:
        result = self.task_classifier.classify_single(query, 'code')
        return result['task']
    
    def _build_prompt(
        self,
        query: str,
        language: Optional[str],
        task_type: str,
        context: Optional[Dict]
    ) -> str:
        prompt_parts = []
        
        # Task-specific instructions
        if task_type == 'debugging':
            prompt_parts.append("You are debugging code. Identify the issue and provide a fix.")
        elif task_type == 'optimization':
            prompt_parts.append("You are optimizing code for better performance and readability.")
        elif task_type == 'review':
            prompt_parts.append("You are reviewing code. Provide constructive feedback.")
        elif task_type == 'testing':
            prompt_parts.append("You are writing test cases. Include edge cases and assertions.")
        elif task_type == 'algorithm':
            prompt_parts.append("You are implementing an algorithm. Focus on correctness and efficiency.")
        else:
            prompt_parts.append("You are generating code. Write clean, idiomatic, well-documented code.")
        
        # Language-specific guidelines
        if language:
            prompt_parts.append(f"\nLanguage: {language}")
            if language == 'python':
                prompt_parts.append("Follow PEP 8 style guide. Use type hints.")
            elif language == 'javascript':
                prompt_parts.append("Use modern ES6+ syntax. Prefer const/let over var.")
            elif language == 'java':
                prompt_parts.append("Follow Java naming conventions. Use proper access modifiers.")
        
        # Best practices
        prompt_parts.append("\nBest practices:")
        prompt_parts.append("- Write clear, self-documenting code")
        prompt_parts.append("- Include comments for complex logic")
        prompt_parts.append("- Handle edge cases and errors")
        prompt_parts.append("- Use descriptive variable/function names")
        
        # Original query
        prompt_parts.append(f"\nTask: {query}")
        
        # Context if provided
        if context:
            prompt_parts.append(f"\nContext: {context}")
        
        prompt_parts.append("\nProvide the complete code solution:")
        
        return "\n".join(prompt_parts)
    
    def _extract_code(self, response: str) -> Optional[str]:
        # Try to extract from markdown code blocks
        code_block_pattern = r'```(?:\w+)?\n(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # Try to find indented code blocks
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.startswith('    ') or line.startswith('\t'):
                in_code = True
                code_lines.append(line)
            elif in_code and line.strip():
                # Check if it's a continuation
                if any(line.startswith(kw) for kw in ['def ', 'class ', 'if ', 'for ', 'while ']):
                    code_lines.append(line)
                else:
                    break
        
        if code_lines:
            return '\n'.join(code_lines)
        
        return None
    
    def _validate_python_syntax(self, code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _calculate_confidence(
        self,
        code: Optional[str],
        syntax_valid: bool,
        task_type: str,
        language: Optional[str]
    ) -> float:
        confidence = 0.5  # Base confidence
        
        # Adjust for code extraction
        if code:
            confidence += 0.2
        
        # Adjust for syntax validation
        if syntax_valid:
            confidence += 0.15
        
        # Adjust for language specificity
        if language:
            confidence += 0.1
        
        # Adjust for task complexity
        if task_type in ['debugging', 'optimization']:
            confidence -= 0.05  # More complex
        elif task_type == 'generation':
            confidence += 0.05  # More straightforward
        
        return min(max(confidence, 0.0), 1.0)
