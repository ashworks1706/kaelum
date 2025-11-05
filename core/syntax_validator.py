import ast
import re
import subprocess
import tempfile
import os
from typing import Dict, Optional, Tuple


class SyntaxValidator:
    
    SUPPORTED_LANGUAGES = ['python', 'javascript', 'typescript']
    
    def __init__(self):
        self._validators = {
            'python': self._validate_python,
            'javascript': self._validate_javascript,
            'typescript': self._validate_javascript
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
    

    
    def _validate_balanced_syntax(self, code: str, language: str) -> Dict:
        stack = []
        bracket_pairs = {'(': ')', '[': ']', '{': '}'}
        
        state = {
            'in_string': False,
            'string_char': None,
            'in_char': False,
            'in_comment': False,
            'in_multiline_comment': False,
            'escape_next': False
        }
        
        i = 0
        while i < len(code):
            char = code[i]
            
            if state['escape_next']:
                state['escape_next'] = False
                i += 1
                continue
            
            if char == '\\' and (state['in_string'] or state['in_char']):
                state['escape_next'] = True
                i += 1
                continue
            
            if not state['in_string'] and not state['in_char']:
                if language in ['cpp', 'java', 'javascript', 'rust', 'go']:
                    if i + 1 < len(code) and code[i:i+2] == '//':
                        state['in_comment'] = True
                        i += 2
                        continue
                    
                    if i + 1 < len(code) and code[i:i+2] == '/*':
                        state['in_multiline_comment'] = True
                        i += 2
                        continue
                    
                    if state['in_multiline_comment'] and i + 1 < len(code) and code[i:i+2] == '*/':
                        state['in_multiline_comment'] = False
                        i += 2
                        continue
                
                if language == 'python' and char == '#':
                    state['in_comment'] = True
            
            if state['in_comment']:
                if char == '\n':
                    state['in_comment'] = False
                i += 1
                continue
            
            if state['in_multiline_comment']:
                i += 1
                continue
            
            if char == '"' and not state['in_char']:
                if language == 'python' and i + 2 < len(code) and code[i:i+3] == '"""':
                    if state['in_string'] and state['string_char'] == '"""':
                        state['in_string'] = False
                        state['string_char'] = None
                    else:
                        state['in_string'] = True
                        state['string_char'] = '"""'
                    i += 3
                    continue
                else:
                    if state['in_string'] and state['string_char'] == '"':
                        state['in_string'] = False
                        state['string_char'] = None
                    else:
                        state['in_string'] = True
                        state['string_char'] = '"'
            elif char == "'" and not state['in_string']:
                if language == 'python' and i + 2 < len(code) and code[i:i+3] == "'''":
                    if state['in_char'] and state['string_char'] == "'''":
                        state['in_char'] = False
                        state['string_char'] = None
                    else:
                        state['in_char'] = True
                        state['string_char'] = "'''"
                    i += 3
                    continue
                else:
                    state['in_char'] = not state['in_char']
            
            if not state['in_string'] and not state['in_char']:
                if char in bracket_pairs:
                    stack.append((char, i))
                elif char in bracket_pairs.values():
                    if not stack:
                        return {
                            'is_valid': False,
                            'error': f'Unmatched closing bracket "{char}" at position {i}',
                            'method': 'syntax_analysis'
                        }
                    opening, pos = stack.pop()
                    if bracket_pairs[opening] != char:
                        return {
                            'is_valid': False,
                            'error': f'Mismatched brackets: "{opening}" at {pos} closed with "{char}" at {i}',
                            'method': 'syntax_analysis'
                        }
            
            i += 1
        
        if stack:
            opening, pos = stack[-1]
            return {
                'is_valid': False,
                'error': f'Unclosed "{opening}" bracket at position {pos}',
                'method': 'syntax_analysis'
            }
        
        return {
            'is_valid': True,
            'error': None,
            'method': 'syntax_analysis'
        }
    
    def _fallback_validate(self, code: str, language: str) -> Dict:
        return self._validate_balanced_syntax(code, language)
