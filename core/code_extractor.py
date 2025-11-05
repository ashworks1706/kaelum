import re
import ast
from typing import Dict, List, Optional


class CodeBlock:
    def __init__(self, code: str, language: str, confidence: float, method: str):
        self.code = code
        self.language = language
        self.confidence = confidence
        self.method = method


class CodeExtractor:
    
    FENCE_PATTERN = r'```(\w+)?\n(.*?)```'
    
    LANGUAGE_KEYWORDS = {
        'python': ['def ', 'class ', 'import ', 'from ', 'self.', '__init__', 'elif ', 'lambda '],
        'javascript': ['function ', 'const ', 'let ', 'var ', '=>', 'async ', 'await ', 'console.'],
        'java': ['public class', 'private ', 'protected ', 'void ', 'static ', 'extends ', 'implements '],
        'cpp': ['#include', 'std::', 'namespace ', 'template<', 'cout <<', 'cin >>'],
        'go': ['package ', 'func ', 'import ', 'type ', 'interface ', 'struct ', 'go func'],
        'rust': ['fn ', 'let mut', 'impl ', 'trait ', 'pub ', 'use ', 'mod '],
    }
    
    def __init__(self):
        pass
    
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
        scores = {}
        
        for lang, keywords in self.LANGUAGE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in code)
            if score > 0:
                scores[lang] = score
        
        if not scores:
            return None
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _looks_like_code(self, line: str, expected_language: Optional[str]) -> bool:
        code_indicators = [
            r'\bdef\s+\w+\s*\(',
            r'\bclass\s+\w+',
            r'\bfunction\s+\w+\s*\(',
            r'\bif\s*\(.+\)\s*{',
            r'\bfor\s*\(.+\)',
            r'\w+\s*=\s*.+[;{]',
            r'[{}()\[\]]{2,}',
        ]
        
        for pattern in code_indicators:
            if re.search(pattern, line):
                return True
        
        if expected_language:
            keywords = self.LANGUAGE_KEYWORDS.get(expected_language, [])
            if any(kw in line for kw in keywords):
                return True
        
        return False
    
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
        if language == 'python':
            try:
                ast.parse(code)
                return True
            except:
                return False
        
        bracket_pairs = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for char in code:
            if char in bracket_pairs:
                stack.append(char)
            elif char in bracket_pairs.values():
                if not stack:
                    return False
                opening = stack.pop()
                if bracket_pairs[opening] != char:
                    return False
        
        return len(stack) == 0
