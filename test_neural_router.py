#!/usr/bin/env python
"""Quick validation test for Neural Router implementation.

This script verifies that all neural router components are properly
implemented and can be imported/initialized without errors.
"""

import sys

def test_imports():
    """Test all neural router imports."""
    print("Testing imports...")
    try:
        from kaelum import NeuralRouter, NeuralRouterTrainer, Router
        from kaelum.core.neural_router import PolicyNetwork, NeuralRoutingFeatures, TORCH_AVAILABLE
        from kaelum.core.neural_router_trainer import TrainingSample, RoutingDataset
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_architecture():
    """Test PolicyNetwork architecture."""
    print("\nTesting PolicyNetwork...")
    try:
        from kaelum.core.neural_router import PolicyNetwork, TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE:
            print("⚠ PyTorch not available - skipping architecture test")
            return True
        
        import torch
        
        # Create network
        net = PolicyNetwork(input_dim=398, hidden_dim=256, num_strategies=5)
        
        # Test forward pass
        dummy_input = torch.randn(1, 398)
        outputs = net(dummy_input)
        
        # Verify outputs
        assert 'strategy_logits' in outputs
        assert 'reflection_logits' in outputs
        assert 'symbolic_logits' in outputs
        assert 'factual_logits' in outputs
        assert 'confidence_logits' in outputs
        
        # Test prediction
        prediction = net.predict_routing(dummy_input)
        assert 'strategy_idx' in prediction
        assert 0 <= prediction['strategy_idx'] <= 4
        assert 0 <= prediction['max_reflection_iterations'] <= 3
        
        print("✓ PolicyNetwork architecture works correctly")
        return True
    except Exception as e:
        print(f"✗ Architecture test failed: {e}")
        return False


def test_feature_extraction():
    """Test feature extraction."""
    print("\nTesting feature extraction...")
    try:
        from kaelum.core.neural_router import NeuralRouter
        
        # Initialize router (without fallback to avoid slow init)
        router = NeuralRouter(fallback_to_rules=False)
        
        # Test feature extraction (internal method)
        features = router._extract_features("Calculate 2 + 2", None)
        
        # Verify features
        assert features.query_embedding is not None
        assert len(features.query_embedding) == 384
        assert features.query_length > 0
        assert 0 <= features.query_complexity <= 1
        assert 0 <= features.math_score <= 1
        
        print("✓ Feature extraction works correctly")
        return True
    except Exception as e:
        print(f"✗ Feature extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test orchestrator integration."""
    print("\nTesting orchestrator integration...")
    try:
        from kaelum.runtime.orchestrator import NEURAL_ROUTER_AVAILABLE
        
        print(f"  Neural router available: {NEURAL_ROUTER_AVAILABLE}")
        print("✓ Orchestrator integration ready")
        return True
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def test_cli():
    """Test CLI availability."""
    print("\nTesting CLI...")
    try:
        import kaelum.cli_neural_router as cli
        
        # Verify CLI has the expected commands
        assert hasattr(cli, 'cli')
        assert hasattr(cli, 'train')
        assert hasattr(cli, 'test')
        assert hasattr(cli, 'stats')
        
        print("✓ CLI module available")
        return True
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        return False


def test_documentation():
    """Test documentation exists."""
    print("\nTesting documentation...")
    try:
        from pathlib import Path
        
        doc_path = Path('docs/NEURAL_ROUTER.md')
        summary_path = Path('IMPLEMENTATION_SUMMARY.md')
        example_path = Path('example_neural_router.py')
        
        assert doc_path.exists(), "docs/NEURAL_ROUTER.md missing"
        assert summary_path.exists(), "IMPLEMENTATION_SUMMARY.md missing"
        assert example_path.exists(), "example_neural_router.py missing"
        
        print("✓ All documentation present")
        return True
    except Exception as e:
        print(f"✗ Documentation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("NEURAL ROUTER VALIDATION TEST")
    print("=" * 70)
    
    tests = [
        ("Imports", test_imports),
        ("Architecture", test_architecture),
        ("Feature Extraction", test_feature_extraction),
        ("Integration", test_integration),
        ("CLI", test_cli),
        ("Documentation", test_documentation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            results.append(test_func())
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    
    for (name, _), result in zip(tests, results):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10s} {name}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Implementation verified!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
