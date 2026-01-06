import asyncio
import json
import uuid
import base64
import websockets
import logging
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TwilioConcurrentSim")

# --- CONFIGURATION ---
# Replace with valid agent IDs from your database
AGENT_ID_A = "d7794b6e-20ff-4e28-9a97-98e327eea2d3" 
AGENT_ID_B = "028cdb34-ef3e-40d9-a4bd-35b92d9c5d29"

WS_URL_BASE = "ws://localhost:5050"

# Audio settings
CHUNK_SIZE = 160  # 20ms of audio at 8kHz
SAMPLE_RATE = 8000
SILENCE_BYTE = b'\xff' # Mu-law silence is often 0xFF or 0x7F. 0xFF is common.

# Concurrency settings
NUM_CONCURRENT_CALLS = 3
RAMP_UP_DELAY = 0.5

async def agent_connection(pair_id, name, agent_id, call_sid, incoming_queue, outgoing_queue):
    """
    Manages the WebSocket connection for one agent.
    """
    ws_url = f"{WS_URL_BASE}/api/call/media-stream/agents/{agent_id}?call_sid={call_sid}"
    logger.info(f"[Pair {pair_id} - {name}] Connecting to {ws_url}...")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            # 1. Send Start Event
            start_event = {
                "event": "start",
                "sequenceNumber": "1",
                "start": {
                    "accountSid": "AC_SIMULATION",
                    "callSid": call_sid,
                    "streamSid": f"stream_{call_sid}",
                    "tracks": ["inbound"],
                    "customParameters": {}
                },
                "streamSid": f"stream_{call_sid}"
            }
            await websocket.send(json.dumps(start_event))
            logger.info(f"[Pair {pair_id} - {name}] Sent 'start' event")

            # 2. Reader Task (Receive from Agent -> Put to Outgoing Queue)
            async def read_from_ws():
                try:
                    async for message in websocket:
                        data = json.loads(message)
                        event_type = data.get("event")
                        
                        if event_type == "media":
                            payload = data["media"]["payload"]
                            # Decode base64 to bytes
                            audio_data = base64.b64decode(payload)
                            # Put raw bytes into outgoing queue
                            await outgoing_queue.put(audio_data)
                        elif event_type == "mark":
                            logger.info(f"[Pair {pair_id} - {name}] Received Mark: {data.get('mark', {}).get('name')}")
                        elif event_type == "clear":
                            logger.info(f"[Pair {pair_id} - {name}] Received Clear")
                            # Clear the incoming queue to stop sending old audio
                            while not incoming_queue.empty():
                                try:
                                    incoming_queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    break
                        elif event_type == "stop":
                            logger.info(f"[Pair {pair_id} - {name}] Received Stop")
                            break
                except Exception as e:
                    logger.error(f"[Pair {pair_id} - {name}] Error reading from WS: {e}")

            # 3. Writer Task (Get from Incoming Queue -> Send to Agent)
            # This runs every 20ms to simulate the Twilio cadence
            async def write_to_ws():
                try:
                    stream_sid = f"stream_{call_sid}"
                    next_tick = time.time()
                    
                    while True:
                        # Calculate time to sleep to maintain 20ms cadence
                        next_tick += 0.02
                        sleep_time = next_tick - time.time()
                        if sleep_time > 0:
                            await asyncio.sleep(sleep_time)
                        
                        # Prepare payload
                        payload_to_send = None
                        
                        # Check if we have audio to relay
                        if not incoming_queue.empty():
                            try:
                                audio_data = incoming_queue.get_nowait()
                                payload_to_send = base64.b64encode(audio_data).decode('utf-8')
                            except asyncio.QueueEmpty:
                                pass
                        
                        # If no audio, send silence
                        if not payload_to_send:
                            silence = SILENCE_BYTE * CHUNK_SIZE
                            payload_to_send = base64.b64encode(silence).decode('utf-8')
                        
                        # Send media event
                        media_event = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": payload_to_send
                            }
                        }
                        await websocket.send(json.dumps(media_event))
                        
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"[Pair {pair_id} - {name}] Error writing to WS: {e}")

            # Run both tasks
            reader_task = asyncio.create_task(read_from_ws())
            writer_task = asyncio.create_task(write_to_ws())
            
            # Wait for reader to finish (connection closed) or error
            await reader_task
            writer_task.cancel()
            
    except Exception as e:
        logger.error(f"[Pair {pair_id} - {name}] Connection failed: {e}")

async def run_simulation_pair(pair_id):
    """
    Runs a single simulation pair (Agent A <-> Agent B).
    """
    # Queues for bridging
    queue_a_to_b = asyncio.Queue()
    queue_b_to_a = asyncio.Queue()

    call_sid_a = str(uuid.uuid4())
    call_sid_b = str(uuid.uuid4())

    logger.info(f"[Pair {pair_id}] Starting bridge...")
    
    # Start connection handlers
    task_a = asyncio.create_task(
        agent_connection(pair_id, "Agent A", AGENT_ID_A, call_sid_a, incoming_queue=queue_b_to_a, outgoing_queue=queue_a_to_b)
    )
    task_b = asyncio.create_task(
        agent_connection(pair_id, "Agent B", AGENT_ID_B, call_sid_b, incoming_queue=queue_a_to_b, outgoing_queue=queue_b_to_a)
    )

    # Wait for them to finish
    await asyncio.gather(task_a, task_b)

async def main():
    logger.info(f"Starting concurrent simulation with {NUM_CONCURRENT_CALLS} calls...")
    
    tasks = []
    for i in range(NUM_CONCURRENT_CALLS):
        logger.info(f"Launching Pair {i+1}...")
        tasks.append(asyncio.create_task(run_simulation_pair(i+1)))
        await asyncio.sleep(RAMP_UP_DELAY)
        
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Simulation stopped by user.")
