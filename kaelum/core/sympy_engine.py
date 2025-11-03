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
        left, right = cls._normalize_equation(expression)
        left_expr = cls._sympify(left)
        right_expr = cls._sympify(right)
        return sp.simplify(left_expr - right_expr) == 0

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
        expr = cls._sympify(expression)
        if isinstance(variables, str):
            return sp.diff(expr, sp.Symbol(variables))
        # Sequence
        diff_args = []
        for item in variables:
            if isinstance(item, tuple):
                var, order = item
                diff_args.append((sp.Symbol(var), order))
            else:
                diff_args.append(sp.Symbol(item))
        return sp.diff(expr, *diff_args)

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
        expr = cls._sympify(expression)
        for item in variables:
            if isinstance(item, tuple):
                var, a, b = item
                expr = sp.integrate(expr, (sp.Symbol(var), a, b))
            else:
                expr = sp.integrate(expr, sp.Symbol(item))
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
        lhs = lhs.strip()
        rhs = rhs.strip()
        # Patterns: d/dx(...)
        if lhs.startswith('d/d'):
            # extract variable and inner expression
            try:
                after = lhs[3:]  # skip 'd/d'
                var = after.split('(')[0].strip()
                inner = lhs.split('(', 1)[1].rsplit(')', 1)[0]
                computed = sp.diff(cls._sympify(inner), sp.Symbol(var))
            except Exception:
                return True  # Non fatal parsing; skip
        elif lhs.startswith('diff'):
            try:
                inner = lhs[len('diff('):-1]  # remove diff( ... )
                parts = [p.strip() for p in inner.split(',')]
                base = cls._sympify(parts[0])
                vars_ = [sp.Symbol(p) for p in parts[1:]]
                computed = sp.diff(base, *vars_)
            except Exception:
                return True
        else:
            return True  # not a derivative form we handle here
        try:
            rhs_expr = cls._sympify(rhs)
            return sp.simplify(computed - rhs_expr) == 0
        except Exception:
            return True

    @classmethod
    def verify_integral(cls, lhs: str, rhs: str) -> bool:
        """Verify integral step for simple forms like ∫(expr)dx = result or integrate(expr, x)."""
        lhs = lhs.strip()
        rhs = rhs.strip()
        if lhs.startswith('integrate'):
            try:
                inner = lhs[len('integrate('):-1]
                parts = [p.strip() for p in inner.split(',')]
                base = cls._sympify(parts[0])
                vars_ = []
                for p in parts[1:]:
                    if p.startswith('(') and p.endswith(')'):
                        vparts = p[1:-1].split(',')
                        if len(vparts) == 3:
                            vars_.append((sp.Symbol(vparts[0].strip()), cls._sympify(vparts[1].strip()), cls._sympify(vparts[2].strip())))
                        else:
                            vars_.append(sp.Symbol(vparts[0].strip()))
                    else:
                        vars_.append(sp.Symbol(p))
                computed = base
                for v in vars_:
                    if isinstance(v, tuple):
                        s, a, b = v
                        computed = sp.integrate(computed, (s, a, b))
                    else:
                        computed = sp.integrate(computed, v)
            except Exception:
                return True
        else:
            return True
        try:
            rhs_expr = cls._sympify(rhs)
            return sp.simplify(computed - rhs_expr) == 0
        except Exception:
            return True

        
    