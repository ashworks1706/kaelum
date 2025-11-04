
import re
import ast
import asyncio
import time
from typing import Optional, Dict, List, Any

from core.workers import WorkerAgent, WorkerResult, WorkerSpecialty
from core.config import KaelumConfig
from core.reasoning import LLMClient, Message


class CodeWorker(WorkerAgent):
    
    def __init__(self, config: Optional[KaelumConfig] = None):
        super().__init__(config)
        self.supported_languages = {
            'python', 'javascript', 'typescript', 'java', 'cpp', 'c',
            'go', 'rust', 'ruby', 'php', 'swift', 'kotlin'
        }
    
    def get_specialty(self) -> WorkerSpecialty:
        return WorkerSpecialty.CODE
    
    def can_handle(self, query: str, context: Optional[Dict] = None) -> float:
        from sentence_transformers import SentenceTransformer, util
        
        if not hasattr(self, '_encoder'):
            self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
            
            code_exemplars = [
                "Write a Python function to sort a list",
                "Implement a binary search algorithm",
                "Debug this JavaScript code",
                "Create a class for user authentication",
                "Optimize this SQL query",
                "Refactor this code to use async/await",
                "Write unit tests for this function",
                "Fix the syntax error in this code"
            ]
            self._code_embeddings = self._encoder.encode(code_exemplars, convert_to_tensor=True)
        
        query_embedding = self._encoder.encode(query, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, self._code_embeddings)[0]
        max_similarity = float(similarities.max())
        
        code_chars = sum(c in query for c in '{}[]();')
        if code_chars > 5:
            max_similarity = min(max_similarity + 0.15, 1.0)
        
        return max_similarity
    
    def solve(self, query: str, context: Optional[Dict] = None) -> WorkerResult:
        return asyncio.run(self._solve_async(query, context))
    
    async def solve_async(self, query: str, context: Optional[Dict] = None) -> WorkerResult:
        return await self._solve_async(query, context)
    
    async def _solve_async(self, query: str, context: Optional[Dict] = None) -> WorkerResult:
        start_time = time.time()
        reasoning_steps = []
        
        # Detect language
        language = self._detect_language(query)
        reasoning_steps.append(f"Detected language: {language or 'unspecified'}")
        
        # Determine task type
        task_type = self._classify_task(query)
        reasoning_steps.append(f"Task type: {task_type}")
        
        # Generate specialized prompt
        prompt = self._build_prompt(query, language, task_type, context)
        reasoning_steps.append("Built specialized code generation prompt")
        
        # Generate code
        messages = [Message(role="user", content=prompt)]
        response = self.llm_client.generate(messages)
        reasoning_steps.append("Generated code solution")
        
        # Extract code from response
        code = self._extract_code(response)
        
        # Validate syntax if Python
        syntax_valid = True
        if language == 'python' and code:
            syntax_valid = self._validate_python_syntax(code)
            reasoning_steps.append(f"Python syntax validation: {'passed' if syntax_valid else 'failed'}")
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            code, syntax_valid, task_type, language
        )
        
        execution_time = time.time() - start_time
        
        return WorkerResult(
            answer=response,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            verification_passed=syntax_valid,
            specialty=WorkerSpecialty.CODE,
            execution_time=execution_time,
            metadata={
                'language': language,
                'task_type': task_type,
                'code_extracted': code is not None,
                'syntax_valid': syntax_valid
            }
        )
    
    def _detect_language(self, query: str) -> Optional[str]:
        query_lower = query.lower()
        
        # First check for explicit language mentions (most reliable)
        # Check longer names first to avoid partial matches
        language_patterns = {
            'javascript': [r'\bjavascript\b', r'\bjs\b', r'\.js\b'],
            'typescript': [r'\btypescript\b', r'\bts\b', r'\.ts\b'],
            'python': [r'\bpython\b', r'\.py\b'],
            'java': [r'\bjava\b', r'\.java\b'],
            'c++': [r'\bc\+\+', r'\bcpp\b', r'\.cpp\b', r'\.hpp\b'],
            'go': [r'\bgolang\b', r'\bgo\s', r'\.go\b'],
            'rust': [r'\brust\b', r'\.rs\b'],
            'ruby': [r'\bruby\b', r'\.rb\b'],
            'php': [r'\bphp\b', r'\.php\b'],
            'swift': [r'\bswift\b', r'\.swift\b'],
            'kotlin': [r'\bkotlin\b', r'\.kt\b'],
            'c': [r'\bc\s', r'\.c\b', r'\.h\b']  # C last, most likely to match accidentally
        }
        
        for lang, patterns in language_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return lang
        
        # Then check for language-specific code patterns
        if 'def ' in query or 'import ' in query or 'self.' in query:
            return 'python'
        elif 'function ' in query or 'const ' in query or 'let ' in query or 'var ' in query:
            return 'javascript'
        elif 'public class' in query or 'private void' in query or 'System.out' in query:
            return 'java'
        
        return None
    
    def _classify_task(self, query: str) -> str:
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ['debug', 'fix', 'error', 'bug']):
            return 'debugging'
        elif any(kw in query_lower for kw in ['optimize', 'improve', 'refactor']):
            return 'optimization'
        elif any(kw in query_lower for kw in ['review', 'analyze', 'explain']):
            return 'review'
        elif any(kw in query_lower for kw in ['test', 'unittest', 'pytest']):
            return 'testing'
        elif any(kw in query_lower for kw in ['algorithm', 'data structure']):
            return 'algorithm'
        else:
            return 'generation'
    
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
    
    async def verify(self, query: str, answer: str, context: Optional[Dict] = None) -> bool:
        # Extract code
        code = self._extract_code(answer)
        if not code:
            return False
        
        # Validate Python syntax
        language = self._detect_language(query)
        if language == 'python':
            return self._validate_python_syntax(code)
        
        # For other languages, basic checks
        if len(code) < 10:  # Too short
            return False
        
        # Check for common error indicators
        error_indicators = ['error', 'exception', 'undefined', 'null pointer']
        if any(indicator in answer.lower() for indicator in error_indicators):
            return False
        
        return True
