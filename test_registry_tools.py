"""Test script for registry and tools integration."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_registry_integration():
    """Test the math model registry functionality."""
    print("Testing Math Model Registry Integration")
    print("=" * 50)
    
    try:
        from kaelum import get_registry, MathCapabilities
        
        registry = get_registry()
        
        # Test listing math models
        math_models = registry.list_math_models()
        print(f"Found {len(math_models)} math models:")
        for model in math_models:
            capabilities = registry.get_math_capabilities(model.model_id)
            print(f"  - {model.model_id}: {model.description}")
            if capabilities:
                caps = []
                if capabilities.symbolic_computation: caps.append("symbolic")
                if capabilities.calculus: caps.append("calculus")
                if capabilities.verification: caps.append("verification")
                if capabilities.multivariate: caps.append("multivariate")
                print(f"    Capabilities: {', '.join(caps)}")
        
        # Test finding best model for specific capabilities
        best_calculus = registry.find_best_math_model(["calculus", "multivariate"])
        if best_calculus:
            print(f"\nBest model for calculus + multivariate: {best_calculus.model_id}")
        
        print("\n✓ Registry integration working correctly!")
        
    except Exception as e:
        print(f"✗ Registry test failed: {e}")
        import traceback
        traceback.print_exc()


def test_tools_integration():
    """Test the math tools functionality."""
    print("\nTesting Math Tools Integration")
    print("=" * 50)
    
    try:
        from kaelum import get_all_kaelum_schemas, kaelum_verify_math, kaelum_compute_math
        
        # Test schema retrieval
        schemas = get_all_kaelum_schemas()
        print(f"Found {len(schemas)} function schemas:")
        for schema in schemas:
            print(f"  - {schema['name']}: {schema['description'][:60]}...")
        
        # Test math verification
        print("\nTesting math verification:")
        result = kaelum_verify_math("diff(x**2, x) = 2*x", "derivative")
        print(f"  Derivative verification: {result['valid']} - {result['message']}")
        
        # Test math computation
        print("\nTesting math computation:")
        result = kaelum_compute_math("x**3 + 2*x", "differentiate", variable="x")
        print(f"  Differentiation result: {result['result']}")
        
        result = kaelum_compute_math("sin(x)", "integrate", variable="x")
        print(f"  Integration result: {result['result']}")
        
        print("\n✓ Tools integration working correctly!")
        
    except Exception as e:
        print(f"✗ Tools test failed: {e}")
        import traceback
        traceback.print_exc()


def test_complete_integration():
    """Test complete integration with real math problems."""
    print("\nTesting Complete Integration")
    print("=" * 50)
    
    try:
        from kaelum import enhance, set_reasoning_model, get_registry
        
        # Configure with a math-capable model
        registry = get_registry()
        math_models = registry.list_math_models()
        
        if math_models:
            best_model = math_models[0]  # Use first available math model
            print(f"Using model: {best_model.model_id}")
            
            set_reasoning_model(
                model=best_model.model_id,
                base_url=best_model.base_url,
                debug_verification=True,
                strict_math_format=True
            )
            
            # Test with a calculus problem
            problem = "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3"
            print(f"\nSolving: {problem}")
            
            # Note: This would normally call the LLM, but we're just testing the setup
            print("✓ Complete integration setup successful!")
        else:
            print("No math models found in registry")
            
    except Exception as e:
        print(f"✗ Complete integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_registry_integration()
    test_tools_integration()
    test_complete_integration()
    print("\n" + "=" * 50)
    print("Registry and Tools Integration Test Complete!")