"""Math expression preprocessor for standardizing queries to SymPy format.

Converts natural language math expressions and various notations into
standardized SymPy-compatible format before passing to SympyEngine.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class MathExpression:
    """Standardized math expression ready for SymPy processing."""
    original: str
    standardized: str
    expression_type: str  # 'equation', 'derivative', 'integral', 'arithmetic'
    variables: List[str]
    metadata: Dict[str, Any]


class MathPreprocessor:
    """Preprocessor that standardizes math expressions for SymPy."""
    
    debug = False
    
    # Symbol mappings for standardization
    SYMBOL_MAPPINGS = {
        '×': '*',
        '÷': '/',
        '·': '*',
        '∗': '*',
        '∙': '*',
        '⋅': '*',
        '^': '**',
        '²': '**2',
        '³': '**3',
        '⁴': '**4',
        '⁵': '**5',
        '√': 'sqrt',
        '∛': 'cbrt',
        '∞': 'oo',
        'π': 'pi',
        'е': 'E',
        '≈': '==',
        '≡': '==',
        '∂': 'diff',
        '∫': 'integrate',
    }
    
    # Function name mappings
    FUNCTION_MAPPINGS = {
        'sine': 'sin',
        'cosine': 'cos',
        'tangent': 'tan',
        'logarithm': 'log',
        'natural log': 'log',
        'ln': 'log',
        'square root': 'sqrt',
        'cube root': 'cbrt',
        'absolute value': 'Abs',
        'exponential': 'exp',
    }
    
    @classmethod
    def set_debug(cls, enabled: bool):
        """Enable or disable debug logging."""
        cls.debug = enabled
    
    @classmethod
    def _log_debug(cls, message: str):
        """Print debug message if debug mode is enabled."""
        if cls.debug:
            print(f"    [MATH PREPROCESSOR] {message}")
    
    @classmethod
    def extract_math_blocks(cls, text: str) -> List[Tuple[str, str]]:
        """Extract [MATH: ...] blocks from text.
        
        Returns:
            List of (original_block, math_content) tuples
        """
        cls._log_debug(f"-> extract_math_blocks from text length {len(text)}")
        
        # Pattern for [MATH: expression]
        pattern = r'\[MATH:\s*([^\]]+)\]'
        matches = []
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            full_match = match.group(0)
            math_content = match.group(1).strip()
            matches.append((full_match, math_content))
            cls._log_debug(f"  Found math block: '{math_content}'")
        
        return matches
    
    @classmethod
    def normalize_expression(cls, expression: str) -> MathExpression:
        """Convert expression to standardized SymPy format."""
        cls._log_debug(f"-> normalize_expression('{expression}')")
        
        original = expression.strip()
        standardized = original
        
        # Step 1: Clean markdown and formatting
        standardized = cls._clean_markdown(standardized)
        cls._log_debug(f"  After markdown cleaning: '{standardized}'")
        
        # Step 2: Replace symbols
        standardized = cls._replace_symbols(standardized)
        cls._log_debug(f"  After symbol replacement: '{standardized}'")
        
        # Step 3: Normalize functions
        standardized = cls._normalize_functions(standardized)
        cls._log_debug(f"  After function normalization: '{standardized}'")
        
        # Step 4: Fix common syntax issues
        standardized = cls._fix_syntax(standardized)
        cls._log_debug(f"  After syntax fixes: '{standardized}'")
        
        # Step 5: Remove currency and units
        standardized = cls._clean_currency_units(standardized)
        cls._log_debug(f"  After currency/unit cleaning: '{standardized}'")
        
        # Step 6: Detect expression type
        expr_type = cls._detect_type(standardized)
        cls._log_debug(f"  Detected type: {expr_type}")
        
        # Step 7: Extract variables
        variables = cls._extract_variables(standardized)
        cls._log_debug(f"  Extracted variables: {variables}")
        
        return MathExpression(
            original=original,
            standardized=standardized,
            expression_type=expr_type,
            variables=variables,
            metadata={'preprocessing_steps': 7}
        )
    
    @classmethod
    def _clean_markdown(cls, expr: str) -> str:
        """Remove markdown formatting."""
        # Remove bold/italic
        expr = re.sub(r'\*\*([^*]+)\*\*', r'\1', expr)  # **bold**
        expr = re.sub(r'__([^_]+)__', r'\1', expr)      # __bold__
        expr = re.sub(r'\*([^*]+)\*', r'\1', expr)      # *italic*
        expr = re.sub(r'_([^_]+)_', r'\1', expr)        # _italic_
        return expr
    
    @classmethod
    def _replace_symbols(cls, expr: str) -> str:
        """Replace mathematical symbols with SymPy equivalents."""
        for symbol, replacement in cls.SYMBOL_MAPPINGS.items():
            expr = expr.replace(symbol, replacement)
        return expr
    
    @classmethod
    def _normalize_functions(cls, expr: str) -> str:
        """Normalize function names."""
        for func_name, sympy_name in cls.FUNCTION_MAPPINGS.items():
            # Word boundary replacement
            pattern = rf'\b{re.escape(func_name)}\b'
            expr = re.sub(pattern, sympy_name, expr, flags=re.IGNORECASE)
        return expr
    
    @classmethod
    def _fix_syntax(cls, expr: str) -> str:
        """Fix common syntax issues."""
        # Add multiplication operator between number and variable: 2x -> 2*x
        expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
        
        # Add multiplication between parentheses: )(  -> )*(
        expr = re.sub(r'\)\s*\(', ')*(', expr)
        
        # Fix function calls without parentheses: sin x -> sin(x)
        expr = re.sub(r'\b(sin|cos|tan|log|exp|sqrt)\s+([a-zA-Z0-9_]+)', r'\1(\2)', expr)
        
        # Balance parentheses (basic attempt)
        open_count = expr.count('(')
        close_count = expr.count(')')
        if open_count > close_count:
            expr += ')' * (open_count - close_count)
        
        return expr
    
    @classmethod
    def _clean_currency_units(cls, expr: str) -> str:
        """Remove currency symbols and units."""
        # Remove currency
        expr = expr.replace('$', '')
        expr = expr.replace('€', '')
        expr = expr.replace('£', '')
        
        # Remove common units (but preserve variables)
        expr = re.sub(r'\b(meters?|feet|inches?|cm|mm|kg|lbs?|seconds?|minutes?|hours?)\b', '', expr, flags=re.IGNORECASE)
        
        return expr.strip()
    
    @classmethod
    def _detect_type(cls, expr: str) -> str:
        """Detect the type of mathematical expression."""
        if 'diff(' in expr or 'd/d' in expr or '∂' in expr:
            return 'derivative'
        elif 'integrate(' in expr or '∫' in expr:
            return 'integral'
        elif '=' in expr:
            return 'equation'
        else:
            return 'arithmetic'
    
    @classmethod
    def _extract_variables(cls, expr: str) -> List[str]:
        """Extract variable names from expression."""
        # Find single letter variables (common in math)
        variables = set(re.findall(r'\b[a-zA-Z]\b', expr))
        
        # Remove common function names and constants
        exclude = {'e', 'E', 'pi', 'sin', 'cos', 'tan', 'log', 'exp', 'sqrt', 'diff', 'integrate'}
        variables = variables - exclude
        
        return sorted(list(variables))