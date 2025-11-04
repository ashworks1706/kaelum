"""Standalone test to demonstrate SympyEngine debug logging without LLM."""

import sys
sys.path.insert(0, r'c:\Users\aadit\Projects\kaelum_mcp\kaelum')

from kaelum.core.sympy_engine import SympyEngine
from kaelum.core.verification import SymbolicVerifier

print("="*80)
print("SympyEngine Debug Logging Test")
print("="*80)

# Enable debug mode
SympyEngine.set_debug(True)

print("\n" + "="*80)
print("Test 1: Equivalence Check")
print("="*80)
result = SympyEngine.check_equivalence("2*x + 3*x = 5*x")
print(f"\nFinal result: {result}\n")

print("="*80)
print("Test 2: Differentiation (multivariate)")
print("="*80)
result = SympyEngine.differentiate("x**2 * y + sin(x*y)", ["x", "y"])
print(f"\nFinal result: {result}\n")

print("="*80)
print("Test 3: Integration (definite)")
print("="*80)
result = SympyEngine.integrate("x*y", [("x", 0, 2), ("y", 0, 1)])
print(f"\nFinal result: {result}\n")

print("="*80)
print("Test 4: Verify Derivative")
print("="*80)
result = SympyEngine.verify_derivative("diff(x**2, x)", "2*x")
print(f"\nFinal result: {result}\n")

print("="*80)
print("Test 5: Verify Integral")
print("="*80)
result = SympyEngine.verify_integral("integrate(x, x)", "x**2/2")
print(f"\nFinal result: {result}\n")

print("="*80)
print("Test 6: Full SymbolicVerifier Check")
print("="*80)
verifier = SymbolicVerifier(debug=True)
step = "Taking the derivative: diff(x**3, x) = 3*x**2"
is_valid, error = verifier.verify_step(step)
print(f"\nStep valid: {is_valid}, Error: {error}\n")

print("="*80)
print("All debug tests complete!")
print("="*80)
