"""
Integration test for FastAPI endpoints.

This test verifies the API works without requiring actual API keys.
"""

import sys
from unittest.mock import Mock, patch

# Mock the LLM clients before importing app
mock_openai = Mock()
mock_anthropic = Mock()

sys.modules['openai'] = mock_openai
sys.modules['anthropic'] = mock_anthropic

# Create mock response
mock_response = Mock()
mock_response.choices = [Mock()]
mock_response.choices[0].message.content = "The answer is x = 2"
mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
mock_openai.OpenAI = Mock(return_value=Mock(chat=Mock(completions=Mock(create=Mock(return_value=mock_response)))))

# Now we can import and test
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("✓ Health endpoint working")


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "KaelumAI" in data["name"]
    print("✓ Root endpoint working")


def test_metrics():
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "verification_rate" in data
    print("✓ Metrics endpoint working")


if __name__ == "__main__":
    print("Testing FastAPI Endpoints...")
    print("=" * 50)
    
    try:
        test_health()
        test_root()
        test_metrics()
        
        print("=" * 50)
        print("✅ All API tests passed!")
        print("\nThe FastAPI application is working correctly.")
        print("\nTo run the server:")
        print("  uvicorn app.main:app --reload")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
