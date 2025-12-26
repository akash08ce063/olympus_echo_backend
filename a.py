import requests
import threading
import time
from datetime import datetime

# API endpoint and configuration
url = 'http://localhost:8000/v1/twilio/calls'
headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
}
payload = {
    "agent_a_number": "+12368044036",
    "agent_b_number": "+13434538795",
    "max_duration_seconds": 30
}

# Store results
results = []
lock = threading.Lock()

def make_call(call_number):
    """Make a single API call and store the result"""
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        end_time = time.time()

        result = {
            'call_number': call_number,
            'status_code': response.status_code,
            'response_time': round(end_time - start_time, 3),
            'success': response.status_code == 200,
            'response_body': response.text[:200] if response.text else 'No response body'
        }

        with lock:
            results.append(result)
            print(f"Call {call_number} completed: Status {response.status_code} in {result['response_time']}s")

    except requests.exceptions.ConnectionError:
        with lock:
            results.append({
                'call_number': call_number,
                'status_code': 'Connection Error',
                'error': 'Could not connect to localhost:8000 - server may not be running',
                'success': False
            })
            print(f"Call {call_number} failed: Connection Error")

    except Exception as e:
        with lock:
            results.append({
                'call_number': call_number,
                'status_code': 'Error',
                'error': str(e),
                'success': False
            })
            print(f"Call {call_number} failed: {str(e)}")

# Create and start 5 threads
print(f"Starting 5 simultaneous API calls at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}...\n")
threads = []

for i in range(1, 6):
    thread = threading.Thread(target=make_call, args=(i,))
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

print(f"\nAll calls completed at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}\n")

# Sort results by call number and display
results.sort(key=lambda x: x['call_number'])

print("=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

for result in results:
    print(f"\nCall #{result['call_number']}:")
    print(f"  Status: {result['status_code']}")
    if 'response_time' in result:
        print(f"  Response Time: {result['response_time']}s")
    if 'response_body' in result:
        print(f"  Response: {result['response_body']}")
    if 'error' in result:
        print(f"  Error: {result['error']}")

successful_calls = sum(1 for r in results if r['success'])
print(f"\n{'=' * 70}")
print(f"SUCCESS RATE: {successful_calls}/5 calls successful")
print("=" * 70)
