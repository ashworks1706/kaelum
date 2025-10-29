"""
Example 5: Using the FastAPI service

This example demonstrates calling the FastAPI endpoints.
"""

import httpx
import json

# Base URL of the KaelumAI API
BASE_URL = "http://localhost:8000"


async def test_api():
    """Test the KaelumAI API endpoints."""
    async with httpx.AsyncClient() as client:
        # Test health endpoint
        print("Testing /health endpoint...")
        response = await client.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}\n")

        # Test verify_reasoning endpoint
        print("Testing /verify_reasoning endpoint...")
        request_data = {
            "query": "If 2x + 3 = 11, what is x?",
            "context": None,
        }
        response = await client.post(
            f"{BASE_URL}/verify_reasoning",
            json=request_data,
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Verified: {result['verified']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Answer: {result['final_answer']}")
        print(f"Trace: {result['trace']}\n")

        # Test metrics endpoint
        print("Testing /metrics endpoint...")
        response = await client.get(f"{BASE_URL}/metrics")
        print(f"Status: {response.status_code}")
        metrics = response.json()
        print(f"Total requests: {metrics['total_requests']}")
        print(f"Verification rate: {metrics['verification_rate']:.2%}")
        print(f"Avg confidence: {metrics['avg_confidence']:.2f}")


if __name__ == "__main__":
    import asyncio

    print("=" * 60)
    print("KaelumAI API Test")
    print("=" * 60)
    print("\nMake sure the API is running:")
    print("  uvicorn app.main:app --reload")
    print("\nThen run this script.")
    print("=" * 60)
    print()

    try:
        asyncio.run(test_api())
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure the API server is running!")
