#!/usr/bin/env python3
"""
Verify that the Flask backend is running in CPU-only mode.
This ensures no GPU memory conflicts with vLLM.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def verify_cpu_mode():
    print("=" * 70)
    print("VERIFYING CPU-ONLY MODE FOR FLASK BACKEND")
    print("=" * 70)
    
    # Check environment variables
    print("\n1. Environment Variables:")
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')
    print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")
    print(f"   Expected: -1 (disables CUDA)")
    print(f"   Status: {'✓ PASS' if cuda_visible == '-1' else '✗ FAIL'}")
    
    # Check PyTorch
    print("\n2. PyTorch Device:")
    try:
        import torch
        default_device = torch.tensor([1.0]).device
        cuda_available = torch.cuda.is_available()
        print(f"   Default device: {default_device}")
        print(f"   CUDA available: {cuda_available}")
        print(f"   Status: {'✓ PASS (CPU)' if str(default_device) == 'cpu' and not cuda_available else '✗ FAIL (GPU DETECTED)'}")
    except ImportError:
        print("   PyTorch not installed - SKIP")
    
    # Check SentenceTransformer
    print("\n3. SentenceTransformer Device:")
    try:
        from core.shared_encoder import get_shared_encoder, reset_shared_encoder
        
        # Reset to ensure fresh load
        reset_shared_encoder()
        
        # Get encoder with explicit CPU device
        encoder = get_shared_encoder('all-MiniLM-L6-v2', device='cpu')
        
        # Check device
        encoder_device = str(encoder.device)
        print(f"   Encoder device: {encoder_device}")
        print(f"   Status: {'✓ PASS (CPU)' if encoder_device == 'cpu' else '✗ FAIL (GPU DETECTED)'}")
        
        # Verify encoding works
        test_embedding = encoder.encode("test query")
        print(f"   Encoding test: ✓ PASS (dimension={len(test_embedding)})")
        
    except Exception as e:
        print(f"   ✗ FAIL: {e}")
    
    # Check Router
    print("\n4. Neural Router:")
    try:
        from core.search import Router
        router = Router(learning_enabled=True, data_dir=".kaelum/routing")
        
        # Check if model uses CPU
        if hasattr(router, 'model') and router.model is not None:
            model_device = next(router.model.parameters()).device
            print(f"   Router model device: {model_device}")
            print(f"   Status: {'✓ PASS (CPU)' if str(model_device) == 'cpu' else '✗ FAIL (GPU DETECTED)'}")
        else:
            print("   Router model not initialized - SKIP")
    except Exception as e:
        print(f"   ✗ FAIL: {e}")
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    print("\nIf all checks passed, the Flask backend will NOT conflict with vLLM GPU usage.")
    print("If any checks failed, review the configuration and shared_encoder.py settings.\n")

if __name__ == '__main__':
    verify_cpu_mode()
