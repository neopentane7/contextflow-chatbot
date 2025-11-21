import requests
import json

BASE_URL = "http://localhost:5000"

def print_response(title, response):
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(json.dumps(response.json(), indent=2))

def test_inference_methods():
    """Test different inference methods"""
    session_id = "test_session_advanced"
    message = "Hello, how can I improve my study skills?"
    
    methods = [
        {
            'name': 'Greedy',
            'method': 'greedy',
            'params': {'max_length': 50}
        },
        {
            'name': 'Beam Search',
            'method': 'beam_search',
            'params': {'beam_width': 5, 'temperature': 0.7, 'max_length': 50}
        },
        {
            'name': 'Temperature Sampling (Creative)',
            'method': 'temperature_sampling',
            'params': {'temperature': 1.0, 'top_k': 50, 'top_p': 0.9, 'max_length': 50}
        },
        {
            'name': 'Temperature Sampling (Focused)',
            'method': 'temperature_sampling',
            'params': {'temperature': 0.6, 'top_k': 30, 'top_p': 0.8, 'max_length': 50}
        }
    ]
    
    for method_info in methods:
        payload = {
            'message': message,
            'session_id': session_id,
            'method': method_info['method'],
            'memory_type': 'full',
            'params': method_info['params']
        }
        
        response = requests.post(f"{BASE_URL}/api/chat", json=payload)
        print_response(f"Response using {method_info['name']}", response)

def test_context_management():
    """Test context window management"""
    session_id = "test_session_context"
    
    # Turn 1
    print("\n--- Turn 1 ---")
    payload1 = {
        'message': "I love Python programming",
        'session_id': session_id,
        'method': 'beam_search',
        'memory_type': 'full'
    }
    r1 = requests.post(f"{BASE_URL}/api/chat", json=payload1)
    print_response("Turn 1", r1)
    
    # Turn 2
    print("\n--- Turn 2 (with context) ---")
    payload2 = {
        'message': "What else should I learn?",
        'session_id': session_id,
        'method': 'beam_search',
        'memory_type': 'full'
    }
    r2 = requests.post(f"{BASE_URL}/api/chat", json=payload2)
    print_response("Turn 2 (should remember Python)", r2)
    
    # Check context
    context_response = requests.get(f"{BASE_URL}/api/context/{session_id}")
    print_response("Current Context", context_response)

def test_available_methods():
    """Show available methods"""
    response = requests.get(f"{BASE_URL}/api/inference-methods")
    print_response("Available Inference Methods", response)

if __name__ == '__main__':
    print("Testing Advanced ContextFlow Backend...\n")
    
    try:
        test_available_methods()
        test_inference_methods()
        test_context_management()
        print("\n✅ All tests completed!")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
