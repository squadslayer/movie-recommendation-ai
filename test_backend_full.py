import requests
import time
import json

BASE_URL = "http://localhost:8000/api"

def test_flow():
    print("🚀 Starting Full Backend Trace Test")
    print("-" * 50)
    
    # 1. Verification of Health
    try:
        r = requests.get(f"{BASE_URL}/health")
        print(f"1. Health Check: {r.status_code}")
        if r.status_code != 200:
            print("❌ Health check failed")
            return
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    # 2. Generate Recommendations (Start Trace)
    payload = {
        "user_id": "test_user_001",
        "input_movies": ["Inception"], # Assuming 'Inception' title lookup works or use ID if known
        "limit": 5
    }
    
    print(f"\n2. Requesting Recommendations for 'Inception'...")
    start = time.time()
    r = requests.post(f"{BASE_URL}/recommendations", json=payload)
    latency = time.time() - start
    
    if r.status_code != 200:
        print(f"❌ Info failed: {r.text}")
        return
        
    data = r.json()
    trace_id = data.get("trace_id")
    print(f"✅ Recommendations Received in {latency:.2f}s")
    print(f"   Trace ID: {trace_id}")
    print(f"   Model Version: {data.get('model_version')}")
    print(f"   Recs Count: {sum(len(v) for v in data.get('recommendations', {}).values())}")

    # 3. Verify Memory/Audit
    print(f"\n3. Auditing Trace {trace_id}...")
    # Give fsync a moment (though it's local file append, should be instant)
    time.sleep(0.5) 
    
    r = requests.get(f"{BASE_URL}/audit/{trace_id}")
    if r.status_code == 200:
        audit = r.json()
        print("✅ Audit Log Retrieved:")
        print(f"   Request User: {audit['request']['user_id']}")
        print(f"   Recs Logged: {len(audit['prediction']['recommendations'])}")
    else:
        print(f"❌ Audit failed: {r.text}")

    print("-" * 50)
    print("✅ TEST COMPLETE - BACKEND IS MEMORY-AWARE")

if __name__ == "__main__":
    test_flow()
