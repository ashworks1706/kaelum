#!/usr/bin/env python3
"""
Test script to verify vLLM connection and compatibility with Kaelum.
Run this before starting full Kaelum tests to ensure your vLLM server is working correctly.
"""

import sys
import argparse
import httpx
import json


def test_vllm_connection(base_url: str, model: str, api_key: str = None):
    """Test connection to vLLM server and verify it works with Kaelum."""
    
    print("=" * 80)
    print(" " * 25 + "vLLM Connection Test")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"  Base URL: {base_url}")
    print(f"  Model: {model}")
    print(f"  API Key: {'Set' if api_key else 'Not set'}")
    print()
    
    # Test 1: Check if server is reachable
    print("Test 1: Checking if vLLM server is reachable...")
    try:
        response = httpx.get(f"{base_url}/models", timeout=5.0)
        response.raise_for_status()
        print("  ✓ Server is reachable")
    except httpx.ConnectError:
        print("  ✗ FAILED: Cannot connect to server")
        print(f"    Make sure vLLM is running at {base_url}")
        print(f"    Start vLLM with: python -m vllm.entrypoints.openai.api_server --model {model} --port <port>")
        return False
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 2: Check if model is available
    print("\nTest 2: Checking if model is loaded...")
    try:
        data = response.json()
        available_models = [m.get('id', '') for m in data.get('data', [])]
        
        if not available_models:
            print("  ✗ FAILED: No models found on server")
            print(f"    Make sure vLLM was started with: --model {model}")
            return False
        
        print(f"  ✓ Available models: {', '.join(available_models)}")
        
        # Check if requested model matches
        model_found = any(model.lower() in m.lower() or m.lower() in model.lower() for m in available_models)
        if not model_found:
            print(f"  ⚠ WARNING: Requested model '{model}' not found in available models")
            print(f"    Available: {available_models}")
            print(f"    This may cause issues. Make sure model names match.")
        else:
            print(f"  ✓ Model '{model}' is available")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 3: Test chat completion
    print("\nTest 3: Testing chat completion API...")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello from vLLM!' if you can read this."}
        ],
        "temperature": 0.7,
        "max_tokens": 50
    }
    
    try:
        response = httpx.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=30.0
        )
        response.raise_for_status()
        data = response.json()
        
        if "choices" in data and len(data["choices"]) > 0:
            message = data["choices"][0].get("message", {}).get("content", "")
            print(f"  ✓ Chat completion successful")
            print(f"  Response: {message}")
        else:
            print("  ✗ FAILED: Empty response from server")
            print(f"    Response: {data}")
            return False
            
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            print("  ✗ FAILED: Authentication error (401)")
            print("    vLLM requires an API key. Try adding --api-key EMPTY")
            print("    Or any dummy key will work for local servers")
            return False
        elif e.response.status_code == 404:
            print("  ✗ FAILED: Model not found (404)")
            print(f"    Model '{model}' not loaded on server")
            print(f"    Check vLLM was started with correct model name")
            return False
        else:
            print(f"  ✗ FAILED: HTTP {e.response.status_code}")
            print(f"    Response: {e.response.text}")
            return False
    except httpx.TimeoutException:
        print("  ✗ FAILED: Request timed out")
        print("    Server may be too slow or overloaded")
        print("    Try a smaller model or reduce max_tokens")
        return False
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 4: Test with Kaelum's LLMClient
    print("\nTest 4: Testing with Kaelum's LLMClient...")
    try:
        from core.config import LLMConfig
        from core.reasoning import LLMClient, Message
        
        config = LLMConfig(
            base_url=base_url,
            model=model,
            api_key=api_key,
            temperature=0.7,
            max_tokens=50
        )
        
        client = LLMClient(config)
        messages = [
            Message(role="system", content="You are a math tutor."),
            Message(role="user", content="What is 2+2?")
        ]
        
        response = client.generate(messages, stream=False)
        print(f"  ✓ Kaelum LLMClient works correctly")
        print(f"  Response: {response}")
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        print("    Kaelum's LLMClient encountered an error")
        return False
    
    # All tests passed
    print("\n" + "=" * 80)
    print(" " * 30 + "✓ ALL TESTS PASSED")
    print("=" * 80)
    print("\n✓ Your vLLM server is configured correctly!")
    print("✓ Kaelum can communicate with your vLLM server")
    print("\nYou can now run Kaelum with:")
    print(f"  python run.py --model {model} --base-url {base_url}", end="")
    if api_key:
        print(f" --api-key {api_key}")
    else:
        print()
    print()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test vLLM connection and compatibility with Kaelum",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test default Ollama server
  python test_vllm_connection.py --model qwen2.5:3b --base-url http://localhost:11434/v1
  
  # Test vLLM server
  python test_vllm_connection.py --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8000/v1 --api-key EMPTY
  
  # Test with custom model
  python test_vllm_connection.py --model microsoft/phi-4 --base-url http://localhost:8000/v1 --api-key EMPTY
        """
    )
    
    parser.add_argument("--base-url", default="http://localhost:8000/v1",
                       help="LLM API base URL (default: http://localhost:8000/v1)")
    parser.add_argument("--model", required=True,
                       help="Model name (e.g., Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--api-key", default=None,
                       help="API key (use 'EMPTY' for local vLLM servers)")
    
    args = parser.parse_args()
    
    success = test_vllm_connection(args.base_url, args.model, args.api_key)
    
    if not success:
        print("\n" + "=" * 80)
        print(" " * 28 + "✗ TESTS FAILED")
        print("=" * 80)
        print("\nSome tests failed. Please check the error messages above.")
        print("\nCommon solutions:")
        print("  1. Make sure vLLM server is running")
        print("  2. Verify the model name matches what vLLM loaded")
        print("  3. Add --api-key EMPTY if you get authentication errors")
        print("  4. Check firewall/port settings")
        print("\nSee VLLM_SETUP.md for detailed troubleshooting guide")
        print()
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
