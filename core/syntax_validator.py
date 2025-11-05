import ast
import re
import subprocess
import tempfile
import os
from typing import Dict, Optional, Tuple


class SyntaxValidator:
    
    SUPPORTED_LANGUAGES = ['python', 'javascript', 'typescript', 'java', 'cpp', 'go', 'rust']
    
    def __init__(self):
        self._validators = {
            'python': self._validate_python,
            'javascript': self._validate_javascript,
            'typescript': self._validate_javascript,
            'java': self._validate_java,
            'cpp': self._validate_cpp,
            'go': self._validate_go,
            'rust': self._validate_rust
        }
    
    def validate(self, code: str, language: str = 'python') -> Dict:
        language = language.lower()
        
        if language not in self.SUPPORTED_LANGUAGES:
            return {
                'is_valid': False,
                'error': f'Unsupported language: {language}',
                'method': 'unsupported'
            }
        
        validator = self._validators.get(language)
        if validator:
            return validator(code)
        
        return {'is_valid': False, 'error': 'Validator not implemented', 'method': 'unimplemented'}
    
    def _validate_python(self, code: str) -> Dict:
        try:
            ast.parse(code)
            return {
                'is_valid': True,
                'error': None,
                'method': 'ast_parser'
            }
        except SyntaxError as e:
            return {
                'is_valid': False,
                'error': str(e),
                'line': e.lineno,
                'offset': e.offset,
                'method': 'ast_parser'
            }
        except Exception as e:
            return {
                'is_valid': False,
                'error': f'Parse error: {str(e)}',
                'method': 'ast_parser'
            }
    
    def _validate_javascript(self, code: str, language: str = 'javascript') -> Dict:
        try:
            suffix = '.ts' if language == 'typescript' else '.js'
            with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
                f.write(code)
                temp_path = f.name
            
            try:
                result = subprocess.run(
                    ['node', '--check', temp_path],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    return {
                        'is_valid': True,
                        'error': None,
                        'method': 'node_check'
                    }
                else:
                    return {
                        'is_valid': False,
                        'error': result.stderr.strip(),
                        'method': 'node_check'
                    }
            finally:
                os.unlink(temp_path)
        except FileNotFoundError:
            return self._fallback_validate(code, language)
        except subprocess.TimeoutExpired:
            return {
                'is_valid': False,
                'error': 'Validation timeout',
                'method': 'node_check'
            }
        except Exception as e:
            return self._fallback_validate(code, language)
    
    def _validate_java(self, code: str) -> Dict:
        structural_check = self._validate_structure(code, {
            'class': r'\bclass\s+\w+',
            'method': r'(?:public|private|protected)?\s*(?:static\s+)?(?:void|int|String|boolean|double|float)\s+\w+\s*\('
        })
        
        if not structural_check['is_valid']:
            return structural_check
        
        return self._validate_balanced_syntax(code, 'java')
    
    def _validate_cpp(self, code: str) -> Dict:
        if not re.search(r'#include\s*<[^>]+>|#include\s*"[^"]+"', code):
            return {
                'is_valid': False,
                'error': 'Missing include directives',
                'method': 'structural_analysis'
            }
        
        return self._validate_balanced_syntax(code, 'cpp')
    
    def _validate_go(self, code: str) -> Dict:
        if not re.search(r'package\s+\w+', code):
            return {
                'is_valid': False,
                'error': 'Missing package declaration',
                'method': 'structural_analysis'
            }
        
        return self._validate_balanced_syntax(code, 'go')
    
    def _validate_rust(self, code: str) -> Dict:
        return self._validate_balanced_syntax(code, 'rust')
    
    def _validate_structure(self, code: str, required_patterns: Dict[str, str]) -> Dict:
        for name, pattern in required_patterns.items():
            if not re.search(pattern, code):
                return {
                    'is_valid': False,
                    'error': f'Missing {name} definition',
                    'method': 'structural_analysis'
                }
        
        return {'is_valid': True, 'error': None}
    
    def _validate_balanced_syntax(self, code: str, language: str) -> Dict:
        in_string = False
        in_char = False
        in_comment = False
        in_multiline_comment = False
        escape_next = False
        
        bracket_stack = []
        bracket_pairs = {'(': ')', '[': ']', '{': '}'}
        
        i = 0
        while i < len(code):
            char = code[i]
            
            if escape_next:
                escape_next = False
                i += 1
                continue
            
            if char == '\\' and (in_string or in_char):
                escape_next = True
                i += 1
                continue
            
            if not in_string and not in_char:
                if language in ['cpp', 'java', 'javascript', 'rust', 'go']:
                    if i + 1 < len(code) and code[i:i+2] == '//':
                        in_comment = True
                        i += 2
                        continue
                    
                    if i + 1 < len(code) and code[i:i+2] == '/*':
                        in_multiline_comment = True
                        i += 2
                        continue
                    
                    if in_multiline_comment and i + 1 < len(code) and code[i:i+2] == '*/':
                        in_multiline_comment = False
                        i += 2
                        continue
                
                if language == 'python' and char == '#':
                    in_comment = True
            
            if in_comment:
                if char == '\n':
                    in_comment = False
                i += 1
                continue
            
            if in_multiline_comment:
                i += 1
                continue
            
            if char == '"' and not in_char:
                in_string = not in_string
            elif char == "'" and not in_string:
                in_char = not in_char
            
            if not in_string and not in_char:
                if char in bracket_pairs:
                    bracket_stack.append((char, i))
                elif char in bracket_pairs.values():
                    if not bracket_stack:
                        return {
                            'is_valid': False,
                            'error': f'Unmatched closing bracket at position {i}',
                            'method': 'syntax_analysis'
                        }
                    opening, pos = bracket_stack.pop()
                    if bracket_pairs[opening] != char:
                        return {
                            'is_valid': False,
                            'error': f'Mismatched brackets: expected {bracket_pairs[opening]} but got {char} at position {i}',
                            'method': 'syntax_analysis'
                        }
            
            i += 1
        
        if bracket_stack:
            opening, pos = bracket_stack[-1]
            return {
                'is_valid': False,
                'error': f'Unclosed {opening} bracket at position {pos}',
                'method': 'syntax_analysis'
            }
        
        return {
            'is_valid': True,
            'error': None,
            'method': 'syntax_analysis'
        }
    
    def _fallback_validate(self, code: str, language: str) -> Dict:
        return self._validate_balanced_syntax(code, language)
