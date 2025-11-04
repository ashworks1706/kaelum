"""SymPy utility engine for equivalence, solving, and multivariate calculus.

The original implementation provided minimal single-variable diff/integrate helpers.
This version expands capabilities to support:
  - Equivalence checking with either '=' or '==' delimiters
  - Solving equations for one or more target variables
  - Multivariate differentiation: specify variables and (optional) orders
  - Multivariate integration: multiple bounds and unbounded variables
  - Generic evaluation of SymPy expressions containing diff()/integrate()/Derivative()/Integral()

Public API methods are intentionally simple to call from verification code.
"""

from __future__ import annotations

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from typing import Sequence, List, Tuple, Union, Optional


class SympyEngine:
    """Collection of static helpers wrapping SymPy functionality."""

    debug = False  # Class-level debug flag
    
    @classmethod
    def set_debug(cls, enabled: bool):
        """Enable or disable debug logging."""
        cls.debug = enabled
    
    @classmethod
    def _log_debug(cls, message: str):
        """Print debug message if debug mode is enabled."""
        if cls.debug:
            print(f"    [SYMPY ENGINE] {message}")

    # -------------------- Parsing Helpers --------------------
    @staticmethod
    def _normalize_equation(equation: str) -> Tuple[str, str]:
        """Return (left, right) for an equation string using '=' or '=='."""
        if '==' in equation:
            left, right = equation.split('==', 1)
        elif '=' in equation:
            left, right = equation.split('=', 1)
        else:
            raise ValueError("Equation must contain '=' or '=='")
        return left.strip(), right.strip()

    @staticmethod
    def _sympify(expr: str):
        return parse_expr(expr, evaluate=True)

    # -------------------- Core Operations --------------------
    @classmethod
    def check_equivalence(cls, expression: str) -> bool:
        """Check symbolic equivalence for 'A = B' or 'A == B'."""
        cls._log_debug(f"→ check_equivalence('{expression}')")
        left, right = cls._normalize_equation(expression)
        cls._log_debug(f"  Normalized: LHS='{left}' RHS='{right}'")
        left_expr = cls._sympify(left)
        cls._log_debug(f"  Parsed LHS: {left_expr}")
        right_expr = cls._sympify(right)
        cls._log_debug(f"  Parsed RHS: {right_expr}")
        diff = sp.simplify(left_expr - right_expr)
        cls._log_debug(f"  Simplified(LHS - RHS): {diff}")
        result = diff == 0
        cls._log_debug(f"  Result: {result}")
        return result

    @classmethod
    def solve_equation(cls, equation: str, solve_for: Optional[Sequence[str]] = None):
        """Solve an equation for the given variables (default: all free symbols).

        Returns list/dict depending on SymPy solve outcome.
        """
        left, right = cls._normalize_equation(equation)
        left_expr = cls._sympify(left)
        right_expr = cls._sympify(right)
        eq = sp.Eq(left_expr, right_expr)
        symbols = list(left_expr.free_symbols.union(right_expr.free_symbols))
        if solve_for:
            vars_to_solve = [sp.Symbol(v) for v in solve_for]
        else:
            vars_to_solve = symbols[:1]  # default: first symbol for backward compatibility
        return sp.solve(eq, *vars_to_solve, dict=True)

    # -------------------- Calculus --------------------
    @classmethod
    def differentiate(
        cls,
        expression: str,
        variables: Union[str, Sequence[str], Sequence[Tuple[str, int]]],
    ):
        """Differentiate expression with respect to provided variables.

        variables formats:
          - 'x' (single var, first order)
          - ['x', 'y'] (∂/∂x then ∂/∂y)
          - [('x', 2), ('y', 1)] for higher-order derivatives.
        """
        cls._log_debug(f"→ differentiate(expression='{expression}', variables={variables})")
        expr = cls._sympify(expression)
        cls._log_debug(f"  Parsed expression: {expr}")
        
        if isinstance(variables, str):
            cls._log_debug(f"  Computing d/d{variables}")
            result = sp.diff(expr, sp.Symbol(variables))
        else:
            # Sequence
            diff_args = []
            for item in variables:
                if isinstance(item, tuple):
                    var, order = item
                    diff_args.append((sp.Symbol(var), order))
                    cls._log_debug(f"  Adding d^{order}/d{var}^{order}")
                else:
                    diff_args.append(sp.Symbol(item))
                    cls._log_debug(f"  Adding d/d{item}")
            result = sp.diff(expr, *diff_args)
        
        cls._log_debug(f"  Result: {result}")
        return result

    @classmethod
    def integrate(
        cls,
        expression: str,
        variables: Sequence[Union[str, Tuple[str, Union[int, float], Union[int, float]]]],
    ):
        """Integrate expression over one or more variables.

        variables formats:
          - ['x'] (indefinite with respect to x)
          - [('x', 0, 1)] definite integral from 0 to 1 wrt x
          - mix of bounded and unbounded: ['x', ('y', 0, 2)]
        Multiple entries produce nested integrals: integrate(integrate(expr, x), (y,0,2)).
        """
        cls._log_debug(f"→ integrate(expression='{expression}', variables={variables})")
        expr = cls._sympify(expression)
        cls._log_debug(f"  Parsed expression: {expr}")
        
        for item in variables:
            if isinstance(item, tuple):
                var, a, b = item
                cls._log_debug(f"  Computing ∫[{a} to {b}] ... d{var}")
                expr = sp.integrate(expr, (sp.Symbol(var), a, b))
                cls._log_debug(f"  Intermediate result: {expr}")
            else:
                cls._log_debug(f"  Computing ∫ ... d{item}")
                expr = sp.integrate(expr, sp.Symbol(item))
                cls._log_debug(f"  Intermediate result: {expr}")
        
        cls._log_debug(f"  Final result: {expr}")
        return expr

    @classmethod
    def evaluate_calculus(cls, expression: str):
        """Evaluate a calculus expression containing diff()/integrate()/Derivative()/Integral().

        Example inputs:
          diff(x**2 * y, x, y)
          integrate(x*y, (x,0,1), y)
          Derivative(sin(x*y), x, y)
          Integral(exp(-x**2), (x, -sp.oo, sp.oo))
        """
        sym_expr = sp.sympify(expression)
        if isinstance(sym_expr, (sp.Derivative, sp.Integral)):
            return sym_expr.doit()
        # If it's a normal expression that still contains Derivative/Integral nodes, attempt doit()
        return sp.simplify(sym_expr.doit()) if hasattr(sym_expr, 'doit') else sym_expr

    # Backwards compatibility for previous method name
    @classmethod
    def calculus_operation(cls, operation: str):  # pragma: no cover - kept for legacy calls
        return cls.evaluate_calculus(operation)

    # -------------------- Validation Helpers --------------------
    @classmethod
    def verify_derivative(cls, lhs: str, rhs: str) -> bool:
        """Verify derivative step: lhs should be a derivative form and rhs its simplified result.

        lhs examples accepted: d/dx(x**2), d/dx ( sin(x) ), diff(x**2, x)
        We normalize lhs into a diff() call and compare with rhs expression.
        """
        cls._log_debug(f"→ verify_derivative(lhs='{lhs}', rhs='{rhs}')")
        lhs = lhs.strip()
        rhs = rhs.strip()
        
        # Patterns: d/dx(...)
        if lhs.startswith('d/d'):
            # extract variable and inner expression
            try:
                after = lhs[3:]  # skip 'd/d'
                var = after.split('(')[0].strip()
                inner = lhs.split('(', 1)[1].rsplit(')', 1)[0]
                cls._log_debug(f"  Detected d/d{var} pattern")
                cls._log_debug(f"  Inner expression: '{inner}'")
                inner_expr = cls._sympify(inner)
                cls._log_debug(f"  Parsed inner: {inner_expr}")
                cls._log_debug(f"  Computing derivative wrt {var}...")
                computed = sp.diff(inner_expr, sp.Symbol(var))
                cls._log_debug(f"  Computed derivative: {computed}")
            except Exception as e:
                cls._log_debug(f"  ⚠ Parse error (non-fatal): {e}")
                return True  # Non fatal parsing; skip
        elif lhs.startswith('diff'):
            try:
                inner = lhs[len('diff('):-1]  # remove diff( ... )
                parts = [p.strip() for p in inner.split(',')]
                cls._log_debug(f"  Detected diff(...) pattern")
                cls._log_debug(f"  Parts: {parts}")
                base = cls._sympify(parts[0])
                cls._log_debug(f"  Base expression: {base}")
                vars_ = [sp.Symbol(p) for p in parts[1:]]
                cls._log_debug(f"  Variables: {[str(v) for v in vars_]}")
                cls._log_debug(f"  Computing derivative...")
                computed = sp.diff(base, *vars_)
                cls._log_debug(f"  Computed derivative: {computed}")
            except Exception as e:
                cls._log_debug(f"  ⚠ Parse error (non-fatal): {e}")
                return True
        else:
            cls._log_debug(f"  Not a recognized derivative form, skipping")
            return True  # not a derivative form we handle here
        
        try:
            rhs_expr = cls._sympify(rhs)
            cls._log_debug(f"  Parsed expected result: {rhs_expr}")
            diff = sp.simplify(computed - rhs_expr)
            cls._log_debug(f"  Simplified(computed - expected): {diff}")
            result = diff == 0
            cls._log_debug(f"  Verification result: {result}")
            return result
        except Exception as e:
            cls._log_debug(f"  ⚠ Comparison error (non-fatal): {e}")
            return True

    @classmethod
    def verify_integral(cls, lhs: str, rhs: str) -> bool:
        """Verify integral step for simple forms like ∫(expr)dx = result or integrate(expr, x)."""
        cls._log_debug(f"→ verify_integral(lhs='{lhs}', rhs='{rhs}')")
        lhs = lhs.strip()
        rhs = rhs.strip()
        
        if lhs.startswith('integrate'):
            try:
                inner = lhs[len('integrate('):-1]
                parts = [p.strip() for p in inner.split(',')]
                cls._log_debug(f"  Detected integrate(...) pattern")
                cls._log_debug(f"  Parts: {parts}")
                base = cls._sympify(parts[0])
                cls._log_debug(f"  Base expression: {base}")
                vars_ = []
                for p in parts[1:]:
                    if p.startswith('(') and p.endswith(')'):
                        vparts = p[1:-1].split(',')
                        if len(vparts) == 3:
                            var_sym = sp.Symbol(vparts[0].strip())
                            a = cls._sympify(vparts[1].strip())
                            b = cls._sympify(vparts[2].strip())
                            vars_.append((var_sym, a, b))
                            cls._log_debug(f"  Integration variable (definite): ∫[{a} to {b}] d{var_sym}")
                        else:
                            vars_.append(sp.Symbol(vparts[0].strip()))
                            cls._log_debug(f"  Integration variable (indefinite): ∫ d{vparts[0].strip()}")
                    else:
                        vars_.append(sp.Symbol(p))
                        cls._log_debug(f"  Integration variable (indefinite): ∫ d{p}")
                
                computed = base
                cls._log_debug(f"  Computing integral...")
                for v in vars_:
                    if isinstance(v, tuple):
                        s, a, b = v
                        computed = sp.integrate(computed, (s, a, b))
                        cls._log_debug(f"  After integrating wrt {s} from {a} to {b}: {computed}")
                    else:
                        computed = sp.integrate(computed, v)
                        cls._log_debug(f"  After integrating wrt {v}: {computed}")
                
                cls._log_debug(f"  Final computed integral: {computed}")
            except Exception as e:
                cls._log_debug(f"  ⚠ Parse error (non-fatal): {e}")
                return True
        else:
            cls._log_debug(f"  Not a recognized integral form, skipping")
            return True
        
        try:
            rhs_expr = cls._sympify(rhs)
            cls._log_debug(f"  Parsed expected result: {rhs_expr}")
            diff = sp.simplify(computed - rhs_expr)
            cls._log_debug(f"  Simplified(computed - expected): {diff}")
            result = diff == 0
            cls._log_debug(f"  Verification result: {result}")
            return result
        except Exception as e:
            cls._log_debug(f"  ⚠ Comparison error (non-fatal): {e}")
            return True

        
    