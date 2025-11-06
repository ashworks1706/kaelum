#!/usr/bin/env python3
"""
Quick test to verify streaming from backend works
"""

import requests
import json

def test_streaming():
    url = "http://localhost:5000/api/query"
    
    payload = {
        "query": "What is 2+2?",
        "stream": True
    }
    
    print("=" * 70)
    print("TESTING STREAMING API")
    print("=" * 70)
    print(f"Query: {payload['query']}")
    print()
    
    try:
        response = requests.post(url, json=payload, stream=True, timeout=30)
        
        print(f"Status: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print()
        print("Stream events:")
        print("-" * 70)
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    try:
                        event = json.loads(data_str)
                        print(f"  [{event['type'].upper()}]", end="")
                        
                        if event['type'] == 'status':
                            print(f" {event.get('message', '')}")
                        elif event['type'] == 'router':
                            print(f" Worker={event.get('worker')}, Confidence={event.get('confidence', 0):.2f}")
                        elif event['type'] == 'reasoning_step':
                            print(f" Step {event.get('index', 0)}: {event.get('content', '')[:50]}...")
                        elif event['type'] == 'answer':
                            print(f" {event.get('content', '')[:80]}...")
                        elif event['type'] == 'verification':
                            print(f" Passed={event.get('passed')}")
                        elif event['type'] == 'done':
                            print(f" Time={event.get('execution_time', 0):.2f}s")
                        elif event['type'] == 'error':
                            print(f" ERROR: {event.get('message')}")
                    except json.JSONDecodeError as e:
                        print(f"  [PARSE ERROR] {e}: {data_str}")
        
        print("-" * 70)
        print("✓ Streaming completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("✗ ERROR: Cannot connect to backend at http://localhost:5000")
        print("  Make sure the Flask server is running: python backend/app.py")
    except requests.exceptions.Timeout:
        print("✗ ERROR: Request timed out after 30 seconds")
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_streaming()
