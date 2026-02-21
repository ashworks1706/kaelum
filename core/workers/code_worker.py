import re
import ast
import time
import logging
from typing import Optional, Dict, List, Any

from .workers import WorkerAgent, WorkerResult, WorkerSpecialty
from ..search import TreeCache
from ..config import KaelumConfig
from ..reasoning import Message
from ..search import LATS, LATSNode
from ..search import RewardModel
from ..detectors import TaskClassifier

logger = logging.getLogger("kaelum.code_worker")

class CodeWorker(WorkerAgent):
    
    def __init__(self, config: Optional[KaelumConfig] = None, tree_cache: Optional[TreeCache] = None):
        super().__init__(config, tree_cache)
        self.task_classifier = TaskClassifier(embedding_model=self.config.embedding_model)
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
              num_simulations: int = 10, parallel: bool = False,
              max_workers: int = 4,
              existing_tree=None,
              extra_sims: int = 0,
              verification_issues: Optional[List[str]] = None) -> WorkerResult:
        start_time = time.time()
        
        if use_cache and existing_tree is None:
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
                prompt += "\n\nProvide ONLY the next implementation step. Keep it concise (describe the step or provide a small code snippet). Do not generate the entire solution."
            
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
        if existing_tree is not None:
            tree = existing_tree
            tree.simulator = simulate_code_step
            tree.expand_fn = expand_code_step
            tree.coherence_checker = self._lightweight_coherence_check
            self._penalize_failed_path(tree, verification_issues or [])
            sims = extra_sims if extra_sims > 0 else max(3, num_simulations // 2)
            logger.info(f"TREE-REUSE: Continuing code search ({sims} additional simulations)")
        else:
            tree = LATS(root_state, simulator=simulate_code_step, expand_fn=expand_code_step)
            sims = num_simulations
        
        tree.run_simulations(sims, max_tree_depth, parallel=parallel, max_workers=max_workers)
        
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
        
        avg_reward = tree.get_avg_reward()
        
        return WorkerResult(
            answer=answer,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            verification_passed=False,
            specialty=self.get_specialty(),
            execution_time=execution_time,            lats_tree=tree,            metadata={
                'language': language,
                'task_type': task_type,
                'syntax_valid': syntax_valid,
                'num_simulations': num_simulations,
                'tree_depth': best_node.state.get("depth", 0),
                'cache_hit': False,
                'avg_reward': avg_reward
            }
        )
    
    def _detect_language(self, query: str) -> Optional[str]:
        """Simple heuristic language detection from query keywords."""
        query_lower = query.lower()
        
        lang_keywords = {
            'python': ['python', '.py', 'def ', 'import ', 'class '],
            'javascript': ['javascript', 'js', '.js', 'function', 'const ', 'let ', '=>'],
            'typescript': ['typescript', 'ts', '.ts', 'interface', 'type '],
            'java': ['java', '.java', 'public class', 'private ', 'System.out'],
            'cpp': ['c++', 'cpp', '.cpp', '#include', 'std::', 'namespace'],
            'go': ['golang', 'go', '.go', 'func ', 'package '],
            'rust': ['rust', '.rs', 'fn ', 'let mut', 'impl ']
        }
        
        scores = {}
        for lang, keywords in lang_keywords.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[lang] = score
        
        if scores:
            return max(scores, key=scores.get)
        return None
    
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
        
        if language:
            prompt_parts.append(f"\nLanguage: {language}")
            if language == 'python':
                prompt_parts.append("Follow PEP 8 style guide. Use type hints.")
            elif language == 'javascript':
                prompt_parts.append("Use modern ES6+ syntax. Prefer const/let over var.")
            elif language == 'java':
                prompt_parts.append("Follow Java naming conventions. Use proper access modifiers.")
        
        prompt_parts.append("\nBest practices:")
        prompt_parts.append("- Write clear, self-documenting code")
        prompt_parts.append("- Include comments for complex logic")
        prompt_parts.append("- Handle edge cases and errors")
        prompt_parts.append("- Use descriptive variable/function names")
        
        prompt_parts.append(f"\nTask: {query}")
        
        if context:
            prompt_parts.append(f"\nContext: {context}")
        
        prompt_parts.append("\nProvide the complete code solution:")
        
        return "\n".join(prompt_parts)
    
    def _extract_code(self, response: str) -> Optional[str]:

        code_block_pattern = r'```(?:\w+)?\n(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.startswith('    ') or line.startswith('\t'):
                in_code = True
                code_lines.append(line)
            elif in_code and line.strip():

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
        base_confidence = 0.5
        
        task_features = {
            'code_present': code is not None,
            'syntax_valid': syntax_valid,
            'language_detected': language is not None,
            'task_simple': task_type == 'generation',
            'task_complex': task_type in ['debugging', 'optimization']
        }
        
        return base_confidence
