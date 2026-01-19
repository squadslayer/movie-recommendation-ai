import requests
import json

def test_endpoint(url):
    print(f"Testing {url}...")
    try:
        response = requests.get(url, timeout=5)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            # Print first 200 chars of JSON
            print(f"Response: {str(data)[:200]}...")
            return True
    except Exception as e:
        print(f"Error: {e}")
    return False

print("="*50)
print("VERIFYING BACKEND API")
print("="*50)

# 1. Health Check
ok1 = test_endpoint("http://localhost:5000/api/health")

# 2. Artist Search (Tom Cruise) - Case sensitive?
ok2 = test_endpoint("http://localhost:5000/api/actors/Tom%20Cruise/info")

if ok1 and ok2:
    print("\n✅ API IS WORKING CORRECTLY!")
else:
    print("\n❌ API ISSUES DETECTED")
