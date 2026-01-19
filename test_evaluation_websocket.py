#!/usr/bin/env python3
"""
Test script to verify WebSocket connection to Pranthora text endpoint.
This tests the connection without authentication.
"""

import asyncio
import json
import websockets
from static_memory_cache import StaticMemoryCache

EVALUATION_AGENT_ID = "c5887ff1-b36d-4794-b380-bad598a0aac6"

async def test_websocket_connection():
    """Test WebSocket connection to evaluation agent."""
    # Get base URL from config
    base_url = StaticMemoryCache.get_pranthora_base_url()
    
    # Convert HTTP to WS
    if base_url.startswith("https://"):
        ws_url = base_url.replace("https://", "wss://")
    else:
        ws_url = base_url.replace("http://", "ws://")
    
    # Correct path is /api/text/stream_text_ws (not /api/v1/text/stream_text_ws)
    ws_url = f"{ws_url}/api/text/stream_text_ws"
    
    print(f"Connecting to: {ws_url}")
    print(f"No authentication required")
    
    try:
        # Connect without any headers (no auth needed)
        async with websockets.connect(
            ws_url,
            open_timeout=30,
            close_timeout=10
        ) as websocket:
            print("✓ WebSocket connected successfully")
            
            # Send initial message with agent_id (required first message)
            init_message = {
                "agent_id": EVALUATION_AGENT_ID
            }
            await websocket.send(json.dumps(init_message))
            print(f"✓ Sent init message: {init_message}")
            
            # Wait a bit for agent initialization
            await asyncio.sleep(0.5)
            
            # Send a test message
            test_message = {
                "input_text": "Hello name of usa full ?."
            }
            await websocket.send(json.dumps(test_message))
            print(f"✓ Sent test message: {test_message['input_text']}")
            
            # Receive response
            print("\nWaiting for response...")
            response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            response_data = json.loads(response)
            print(response_data)
            if "response" in response_data:
                print(f"✓ Received response: {response_data['response']}")
                print("\n✅ WebSocket connection test PASSED!")
                return True
            else:
                print(f"⚠ Unexpected response format: {response_data}")
                return False
                
    except websockets.exceptions.InvalidURI as e:
        print(f"❌ Invalid WebSocket URI: {e}")
        return False
    except websockets.exceptions.InvalidStatus as e:
        print(f"❌ Connection rejected (HTTP {e.status_code}): {e}")
        print(f"   This usually means authentication is required or endpoint is wrong")
        return False
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_websocket_connection())
    exit(0 if result else 1)

