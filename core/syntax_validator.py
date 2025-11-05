import ast
import re
import subprocess
import tempfile
import os
from typing import Dict, Optional


class SyntaxValidator:
    
    SUPPORTED_LANGUAGES = ['python', 'javascript', 'typescript', 'java', 'cpp', 'go', 'rust']
    
    def __init__(self):
        pass
    
    def validate(self, code: str, language: str = 'python') -> Dict:
        language = language.lower()
        
        if language not in self.SUPPORTED_LANGUAGES:
            return {
                'is_valid': False,
                'error': f'Unsupported language: {language}',
                'method': 'unsupported'
            }
        
        if language == 'python':
            return self._validate_python(code)
        elif language in ['javascript', 'typescript']:
            return self._validate_javascript(code, language)
        elif language == 'java':
            return self._validate_java(code)
        elif language == 'cpp':
            return self._validate_cpp(code)
        elif language == 'go':
            return self._validate_go(code)
        elif language == 'rust':
            return self._validate_rust(code)
        else:
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
    
    def _validate_javascript(self, code: str, language: str) -> Dict:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
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
            return {
                'is_valid': False,
                'error': 'Node.js not found in system',
                'method': 'node_check'
            }
        except subprocess.TimeoutExpired:
            return {
                'is_valid': False,
                'error': 'Validation timeout',
                'method': 'node_check'
            }
        except Exception as e:
            return {
                'is_valid': False,
                'error': f'Validation error: {str(e)}',
                'method': 'node_check'
            }
    
    def _validate_java(self, code: str) -> Dict:
        basic_checks = [
            (r'class\s+\w+', 'No class definition found'),
            (r'(?:public|private|protected)?\s*(?:static\s+)?(?:void|int|String|boolean|double|float)\s+\w+\s*\(', 'No method signature found')
        ]
        
        for pattern, error_msg in basic_checks:
            if not re.search(pattern, code):
                return {
                    'is_valid': False,
                    'error': error_msg,
                    'method': 'regex_heuristic'
                }
        
        bracket_stack = []
        for i, char in enumerate(code):
            if char in '{[(':
                bracket_stack.append((char, i))
            elif char in '}])':
                if not bracket_stack:
                    return {
                        'is_valid': False,
                        'error': f'Unmatched closing bracket at position {i}',
                        'method': 'bracket_matching'
                    }
                opening, _ = bracket_stack.pop()
                expected = {'(': ')', '[': ']', '{': '}'}
                if expected[opening] != char:
                    return {
                        'is_valid': False,
                        'error': f'Mismatched brackets at position {i}',
                        'method': 'bracket_matching'
                    }
        
        if bracket_stack:
            return {
                'is_valid': False,
                'error': f'Unclosed bracket at position {bracket_stack[-1][1]}',
                'method': 'bracket_matching'
            }
        
        return {
            'is_valid': True,
            'error': None,
            'method': 'heuristic_validation'
        }
    
    def _validate_cpp(self, code: str) -> Dict:
        required_patterns = [
            (r'#include\s*<\w+>', 'Missing include statements'),
        ]
        
        for pattern, error_msg in required_patterns:
            if not re.search(pattern, code):
                return {
                    'is_valid': False,
                    'error': error_msg,
                    'method': 'regex_heuristic'
                }
        
        return self._validate_brackets(code)
    
    def _validate_go(self, code: str) -> Dict:
        if not re.search(r'package\s+\w+', code):
            return {
                'is_valid': False,
                'error': 'Missing package declaration',
                'method': 'regex_heuristic'
            }
        
        return self._validate_brackets(code)
    
    def _validate_rust(self, code: str) -> Dict:
        return self._validate_brackets(code)
    
    def _validate_brackets(self, code: str) -> Dict:
        bracket_stack = []
        for i, char in enumerate(code):
            if char in '{[(':
                bracket_stack.append((char, i))
            elif char in '}])':
                if not bracket_stack:
                    return {
                        'is_valid': False,
                        'error': f'Unmatched closing bracket at position {i}',
                        'method': 'bracket_matching'
                    }
                opening, _ = bracket_stack.pop()
                expected = {'(': ')', '[': ']', '{': '}'}
                if expected[opening] != char:
                    return {
                        'is_valid': False,
                        'error': f'Mismatched brackets at position {i}',
                        'method': 'bracket_matching'
                    }
        
        if bracket_stack:
            return {
                'is_valid': False,
                'error': f'Unclosed bracket at position {bracket_stack[-1][1]}',
                'method': 'bracket_matching'
            }
        
        return {
            'is_valid': True,
            'error': None,
            'method': 'bracket_matching'
        }
