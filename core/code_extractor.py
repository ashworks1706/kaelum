import re
import ast
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np


class CodeBlock:
    def __init__(self, code: str, language: str, confidence: float, method: str):
        self.code = code
        self.language = language
        self.confidence = confidence
        self.method = method


class CodeExtractor:
    
    FENCE_PATTERN = r'```(\w+)?\n(.*?)```'
    
    def __init__(self):
        self.encoder = SentenceTransformer('all-mpnet-base-v2')
        
        self.language_signatures = {
            'python': {
                'exemplars': [
                    "def function():\n    pass",
                    "import module\nclass MyClass:\n    def __init__(self):\n        pass"
                ],
                'strong_patterns': [
                    (r'^\s*def\s+\w+\s*\([^)]*\)\s*:', 0.9, True),
                    (r'^\s*class\s+\w+.*:', 0.9, True),
                    (r'^\s*import\s+\w+', 0.85, True),
                    (r'^\s*from\s+\w+\s+import', 0.85, True),
                    (r'@\w+\s*\n\s*def', 0.80, True),
                    (r'async\s+def\s+\w+', 0.85, True)
                ],
                'ast_validator': self._validate_python_ast
            },
            'javascript': {
                'exemplars': [
                    "function test() { return true; }",
                    "const data = async () => await fetch();"
                ],
                'strong_patterns': [
                    (r'\bfunction\s+\w+\s*\([^)]*\)\s*\{', 0.85, True),
                    (r'\b(const|let|var)\s+\w+\s*=', 0.75, False),
                    (r'=>\s*\{', 0.80, False),
                    (r'\basync\s+function|\bawait\s+', 0.80, True),
                    (r'console\.log\(', 0.70, False)
                ],
                'ast_validator': None
            },
            'java': {
                'exemplars': [
                    "public class Test {\n    public static void main(String[] args) {}\n}"
                ],
                'strong_patterns': [
                    (r'\b(public|private)\s+class\s+\w+', 0.90),
                    (r'\b(public|private|protected)\s+\w+\s+\w+\s*\(', 0.80),
                    (r'System\.out\.print', 0.85)
                ],
                'ast_validator': None
            },
            'cpp': {
                'exemplars': [
                    "#include <iostream>\nint main() { std::cout << \"test\"; }"
                ],
                'strong_patterns': [
                    (r'#include\s*<[^>]+>', 0.90),
                    (r'std::\w+', 0.85),
                    (r'\btemplate\s*<', 0.90)
                ],
                'ast_validator': None
            },
            'go': {
                'exemplars': [
                    "package main\nfunc main() { fmt.Println(\"test\") }"
                ],
                'strong_patterns': [
                    (r'^\s*package\s+\w+', 0.90),
                    (r'\bfunc\s+\w+\s*\(', 0.85),
                    (r':=', 0.70)
                ],
                'ast_validator': None
            },
            'rust': {
                'exemplars': [
                    "fn main() { let x = 5; println!(\"{}\", x); }"
                ],
                'strong_patterns': [
                    (r'\bfn\s+\w+\s*\(', 0.90),
                    (r'\blet\s+mut\s+', 0.85),
                    (r'\bimpl\s+\w+', 0.85)
                ],
                'ast_validator': None
            }
        }
        
        self._cache_embeddings()
    
    def _cache_embeddings(self):
        self.lang_embeddings = {}
        for lang, sig in self.language_signatures.items():
            embeddings = self.encoder.encode(sig['exemplars'], convert_to_tensor=False)
            self.lang_embeddings[lang] = embeddings
    
    def extract(self, text: str, expected_language: Optional[str] = None) -> List[CodeBlock]:
        blocks = []
        
        fenced_blocks = self._extract_fenced(text)
        blocks.extend(fenced_blocks)
        
        if not blocks:
            indented_blocks = self._extract_indented(text, expected_language)
            blocks.extend(indented_blocks)
        
        if not blocks:
            inline_blocks = self._extract_inline(text, expected_language)
            blocks.extend(inline_blocks)
        
        return blocks
    
    def _extract_fenced(self, text: str) -> List[CodeBlock]:
        blocks = []
        
        matches = re.finditer(self.FENCE_PATTERN, text, re.DOTALL)
        
        for match in matches:
            language = match.group(1) or 'unknown'
            code = match.group(2).strip()
            
            if not code:
                continue
            
            detected_lang = self._detect_language(code)
            if language == 'unknown' and detected_lang:
                language = detected_lang
                confidence = 0.7
            else:
                confidence = 0.95
            
            blocks.append(CodeBlock(code, language, confidence, 'fenced'))
        
        return blocks
    
    def _extract_indented(self, text: str, expected_language: Optional[str]) -> List[CodeBlock]:
        lines = text.split('\n')
        code_blocks = []
        current_block = []
        in_code = False
        base_indent = 0
        
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            
            if not in_code:
                if indent >= 4 or line.startswith('\t'):
                    in_code = True
                    base_indent = indent
                    current_block = [stripped]
                elif self._looks_like_code(line, expected_language):
                    in_code = True
                    base_indent = 0
                    current_block = [line]
            else:
                if not line.strip():
                    current_block.append('')
                elif indent >= base_indent or self._is_continuation(line, current_block):
                    current_block.append(line[base_indent:] if base_indent > 0 else line)
                else:
                    if current_block:
                        code = '\n'.join(current_block).strip()
                        if len(code) > 10:
                            language = self._detect_language(code) or expected_language or 'unknown'
                            confidence = 0.6 if self._validate_syntax(code, language) else 0.4
                            code_blocks.append(CodeBlock(code, language, confidence, 'indented'))
                    
                    in_code = False
                    current_block = []
                    
                    if indent >= 4 or line.startswith('\t'):
                        in_code = True
                        base_indent = indent
                        current_block = [stripped]
        
        if current_block:
            code = '\n'.join(current_block).strip()
            if len(code) > 10:
                language = self._detect_language(code) or expected_language or 'unknown'
                confidence = 0.6 if self._validate_syntax(code, language) else 0.4
                code_blocks.append(CodeBlock(code, language, confidence, 'indented'))
        
        return code_blocks
    
    def _extract_inline(self, text: str, expected_language: Optional[str]) -> List[CodeBlock]:
        inline_pattern = r'`([^`]+)`'
        matches = re.findall(inline_pattern, text)
        
        blocks = []
        for code in matches:
            if len(code) > 15 and self._looks_like_code(code, expected_language):
                language = self._detect_language(code) or expected_language or 'unknown'
                blocks.append(CodeBlock(code, language, 0.5, 'inline'))
        
        return blocks
    
    def _detect_language(self, code: str) -> Optional[str]:
        if not code.strip():
            return None
        
        scores = {}
        
        for lang, sig in self.language_signatures.items():
            score = 0.0
            
            for pattern, weight in sig['strong_patterns']:
                if re.search(pattern, code, re.MULTILINE):
                    score += weight
            
            if sig['ast_validator']:
                if sig['ast_validator'](code):
                    score += 0.5
            
            if score > 0:
                scores[lang] = score
        
        if len(code) > 30:
            semantic_scores = self._semantic_language_match(code)
            for lang, sem_score in semantic_scores.items():
                scores[lang] = scores.get(lang, 0.0) * 0.6 + sem_score * 0.4
        
        if not scores:
            return None
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _semantic_language_match(self, code: str) -> Dict[str, float]:
        code_embedding = self.encoder.encode(code, convert_to_tensor=False)
        
        scores = {}
        for lang, exemplar_embeddings in self.lang_embeddings.items():
            similarities = []
            for exemplar_emb in exemplar_embeddings:
                sim = np.dot(code_embedding, exemplar_emb) / (
                    np.linalg.norm(code_embedding) * np.linalg.norm(exemplar_emb) + 1e-9
                )
                similarities.append(sim)
            scores[lang] = float(np.max(similarities)) if similarities else 0.0
        
        return scores
    
    def _validate_python_ast(self, code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except:
            return False
    
    def _looks_like_code(self, line: str, expected_language: Optional[str]) -> bool:
        if not line.strip():
            return False
        
        code_indicators = [
            (r'\b(def|class|function|public|private|fn|func)\s+\w+', 0.9),
            (r'[{}\[\]()].*[{}\[\]()]', 0.7),
            (r'\w+\s*=\s*\w+', 0.6),
            (r'(import|#include|package)\s+\w+', 0.8),
            (r'(if|for|while)\s*\(', 0.7),
            (r'->\s*\w+|=>\s*', 0.7)
        ]
        
        total_score = 0.0
        for pattern, weight in code_indicators:
            if re.search(pattern, line):
                total_score += weight
        
        if expected_language and expected_language in self.language_signatures:
            for pattern, weight in self.language_signatures[expected_language]['strong_patterns']:
                if re.search(pattern, line):
                    total_score += weight * 0.5
        
        return total_score > 0.6
    
    def _is_continuation(self, line: str, current_block: List[str]) -> bool:
        if not current_block:
            return False
        
        last_line = current_block[-1].rstrip()
        
        continuation_indicators = [',', '\\', '(', '[', '{', 'and', 'or', '+', '-', '*', '/']
        if any(last_line.endswith(ind) for ind in continuation_indicators):
            return True
        
        if line.strip().startswith((')', ']', '}', '.', 'else', 'elif', 'except', 'finally', 'catch')):
            return True
        
        return False
    
    def _validate_syntax(self, code: str, language: str) -> bool:
        if language == 'python' and language in self.language_signatures:
            validator = self.language_signatures[language]['ast_validator']
            if validator:
                return validator(code)
        
        bracket_pairs = {'(': ')', '[': ']', '{': '}'}
        stack = []
        in_string = False
        string_char = None
        escape_next = False
        
        for i, char in enumerate(code):
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char in ['"', "'", '`']:
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
                continue
            
            if in_string:
                continue
            
            if char in bracket_pairs:
                stack.append((char, i))
            elif char in bracket_pairs.values():
                if not stack:
                    return False
                opening, pos = stack.pop()
                if bracket_pairs[opening] != char:
                    return False
        
        return len(stack) == 0
