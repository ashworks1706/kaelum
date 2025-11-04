"""Function calling tools for commercial LLMs to use Kaelum reasoning."""

from typing import Dict, List, Any, Optional, Union
from ..core.sympy_engine import SympyEngine
from ..core.verification import SymbolicVerifier
from ..core.math_preprocessor import MathPreprocessor


def get_kaelum_function_schema() -> Dict[str, Any]:
    """
    Get the function schema for Kaelum reasoning enhancement.
    This can be passed to commercial LLMs (Gemini, GPT-4, Claude, etc.) 
    as a function/tool they can call.
    
    Returns:
        Function schema compatible with OpenAI/Gemini function calling format
    """
    return {
        "name": "kaelum_enhance_reasoning",
        "description": (
            "Enhances reasoning for complex questions by breaking them down into "
            "logical steps. Use this when you need to solve math problems, "
            "logical puzzles, multi-step reasoning tasks, or any question that "
            "requires careful step-by-step thinking. Returns a structured reasoning "
            "trace that you can use to formulate your final answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or problem that needs reasoning enhancement"
                },
                "domain": {
                    "type": "string",
                    "description": "Optional domain hint (e.g., 'math', 'logic', 'code', 'science')",
                    "enum": ["math", "logic", "code", "science", "general"]
                }
            },
            "required": ["query", "domain"]
        }
    }


def get_math_verification_schema() -> Dict[str, Any]:
    """
    Get the function schema for math verification capabilities.
    This allows LLMs to specifically request mathematical verification
    of expressions, equations, derivatives, and integrals.
    
    Returns:
        Function schema for math verification
    """
    return {
        "name": "kaelum_verify_math",
        "description": (
            "Verifies mathematical expressions, equations, derivatives, and integrals "
            "using symbolic computation. Use this to check if mathematical statements "
            "are correct, verify calculus operations, or validate equation solutions. "
            "Supports multivariate calculus and complex mathematical expressions."
        ),
        "parameters": {
            "type": "object", 
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to verify (e.g., 'diff(x^2, x) = 2*x')"
                },
                "verification_type": {
                    "type": "string",
                    "description": "Type of verification to perform",
                    "enum": ["derivative", "integral", "equation", "equivalence", "general"]
                },
                "variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of variables in the expression (optional)"
                }
            },
            "required": ["expression", "verification_type"]
        }
    }


def get_math_computation_schema() -> Dict[str, Any]:
    """
    Get the function schema for direct math computation capabilities.
    This allows LLMs to request specific mathematical computations
    like derivatives, integrals, equation solving, etc.
    
    Returns:
        Function schema for math computation
    """
    return {
        "name": "kaelum_compute_math",
        "description": (
            "Performs mathematical computations including derivatives, integrals, "
            "equation solving, and expression evaluation. Use this when you need "
            "to compute specific mathematical operations symbolically with exact results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string", 
                    "description": "The mathematical expression to compute"
                },
                "operation": {
                    "type": "string",
                    "description": "The operation to perform",
                    "enum": ["differentiate", "integrate", "solve", "evaluate", "simplify"]
                },
                "variable": {
                    "type": "string",
                    "description": "Variable to differentiate/integrate with respect to (required for calculus)"
                },
                "bounds": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Integration bounds [lower, upper] for definite integrals"
                },
                "solve_for": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "Variables to solve for in equations"
                }
            },
            "required": ["expression", "operation"]
        }
    }


def get_all_kaelum_schemas() -> List[Dict[str, Any]]:
    """
    Get all available Kaelum function schemas.
    
    Returns:
        List of all function schemas for integration with commercial LLMs
    """
    return [
        get_kaelum_function_schema(),
        get_math_verification_schema(), 
        get_math_computation_schema()
    ]


# Legacy function for backward compatibility
def get_kaelum_function_schemas() -> List[Dict[str, Any]]:
    """Legacy function - use get_all_kaelum_schemas() instead."""
    return get_all_kaelum_schemas()


# Implementation functions for the schemas above

def kaelum_verify_math(expression: str, verification_type: str, variables: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Verify mathematical expressions using symbolic computation.
    
    Args:
        expression: Mathematical expression to verify
        verification_type: Type of verification (derivative, integral, equation, equivalence, general)
        variables: Optional list of variables in the expression
        
    Returns:
        Dictionary with verification results
    """
    try:
        verifier = SymbolicVerifier(debug=True)
        preprocessor = MathPreprocessor()
        
        # Preprocess the expression
        normalized = preprocessor.normalize_expression(expression)
        
        if verification_type == "derivative":
            # Look for derivative verification pattern
            if "=" in expression:
                lhs, rhs = expression.split("=", 1)
                lhs = lhs.strip()
                rhs = rhs.strip()
                
                # Check if it's a derivative verification
                if "diff(" in lhs or "d/" in lhs:
                    is_valid = SympyEngine.verify_derivative(lhs, rhs)
                    return {
                        "valid": is_valid,
                        "type": "derivative_verification",
                        "expression": expression,
                        "normalized": normalized.standardized,
                        "message": "Derivative verification completed" if is_valid else "Derivative verification failed"
                    }
            
        elif verification_type == "integral":
            # Look for integral verification pattern  
            if "=" in expression:
                lhs, rhs = expression.split("=", 1)
                lhs = lhs.strip()
                rhs = rhs.strip()
                
                if "integrate(" in lhs or "∫" in lhs:
                    is_valid = SympyEngine.verify_integral(lhs, rhs)
                    return {
                        "valid": is_valid,
                        "type": "integral_verification", 
                        "expression": expression,
                        "normalized": normalized.standardized,
                        "message": "Integral verification completed" if is_valid else "Integral verification failed"
                    }
                    
        elif verification_type == "equation":
            # Verify equation solution
            is_valid = SympyEngine.check_equivalence(normalized.standardized)
            return {
                "valid": is_valid,
                "type": "equation_verification",
                "expression": expression,
                "normalized": normalized.standardized,
                "message": "Equation is valid" if is_valid else "Equation is invalid"
            }
            
        elif verification_type == "equivalence":
            # Check mathematical equivalence
            is_valid = SympyEngine.check_equivalence(normalized.standardized)
            return {
                "valid": is_valid,
                "type": "equivalence_check",
                "expression": expression,
                "normalized": normalized.standardized,
                "message": "Expressions are equivalent" if is_valid else "Expressions are not equivalent"
            }
            
        # General verification using step verification
        is_valid, error = verifier.verify_step(f"[MATH: {normalized.standardized}]")
        return {
            "valid": is_valid,
            "type": "general_verification",
            "expression": expression,
            "normalized": normalized.standardized,
            "message": "Expression is mathematically valid" if is_valid else f"Verification failed: {error}",
            "error": error if not is_valid else None
        }
        
    except Exception as e:
        return {
            "valid": False,
            "type": "error",
            "expression": expression,
            "message": f"Verification error: {str(e)}",
            "error": str(e)
        }


def kaelum_compute_math(expression: str, operation: str, variable: Optional[str] = None, 
                       bounds: Optional[List[float]] = None, solve_for: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Perform mathematical computations using symbolic computation.
    
    Args:
        expression: Mathematical expression to compute
        operation: Operation to perform (differentiate, integrate, solve, evaluate, simplify)  
        variable: Variable for calculus operations
        bounds: Integration bounds for definite integrals
        solve_for: Variables to solve for in equations
        
    Returns:
        Dictionary with computation results
    """
    try:
        preprocessor = MathPreprocessor()
        normalized = preprocessor.normalize_expression(expression)
        expr = normalized.standardized
        
        if operation == "differentiate":
            if not variable:
                return {"error": "Variable required for differentiation", "valid": False}
            
            result = SympyEngine.differentiate(expr, [variable])
            return {
                "valid": True,
                "operation": "differentiate",
                "input": expression,
                "result": str(result),
                "variable": variable,
                "normalized_input": expr
            }
            
        elif operation == "integrate":
            if not variable:
                return {"error": "Variable required for integration", "valid": False}
                
            if bounds and len(bounds) == 2:
                # Definite integral with bounds
                variables = [(variable, bounds[0], bounds[1])]
            else:
                # Indefinite integral
                variables = [variable]
                
            result = SympyEngine.integrate(expr, variables)
                
            return {
                "valid": True,
                "operation": "integrate",
                "input": expression,
                "result": str(result),
                "variable": variable,
                "bounds": bounds,
                "normalized_input": expr
            }
            
        elif operation == "solve":
            if not solve_for:
                # Try to auto-detect variables
                solve_for = list(normalized.variables)
                
            result = SympyEngine.solve_equation(expr, solve_for=solve_for)
            return {
                "valid": True,
                "operation": "solve",
                "input": expression,
                "result": str(result) if result else "No solution found",
                "solve_for": solve_for,
                "normalized_input": expr
            }
            
        elif operation == "evaluate":
            result = SympyEngine.evaluate_calculus(expr)
            return {
                "valid": True,
                "operation": "evaluate",
                "input": expression,
                "result": str(result),
                "normalized_input": expr
            }
            
        elif operation == "simplify":
            from sympy import simplify, sympify
            result = simplify(sympify(expr))
            return {
                "valid": True,
                "operation": "simplify",
                "input": expression,
                "result": str(result),
                "normalized_input": expr
            }
            
        else:
            return {
                "valid": False,
                "error": f"Unknown operation: {operation}",
                "supported_operations": ["differentiate", "integrate", "solve", "evaluate", "simplify"]
            }
            
    except Exception as e:
        return {
            "valid": False,
            "operation": operation,
            "input": expression,
            "error": str(e),
            "message": f"Computation failed: {str(e)}"
        }
