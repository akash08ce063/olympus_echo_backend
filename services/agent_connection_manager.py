import asyncio
import json
import base64
import time
import logging
from typing import Optional, Dict
import websockets
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

logger = logging.getLogger(__name__)


class AgentConnectionManager:
    """
    Manages a single WebSocket connection for an agent (User or Target) in the test simulation.
    Handles the connection lifecycle, audio streaming (read/write), and silence detection.
    """

    def __init__(
        self,
        name: str,
        ws_url: str,
        call_sid: str,
        incoming_queue: asyncio.Queue,
        outgoing_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        my_ready: asyncio.Event,
        other_ready: asyncio.Event,
        chunk_duration_seconds: float = 0.02,
        sync_timeout: float = 10.0,
        extra_headers: Optional[Dict[str, str]] = None,
        record_sent_callback: Optional[callable] = None,
    ):
        """
        Initialize the connection manager.

        Args:
            name: The display name for logging (e.g., "Target", "User").
            ws_url: The WebSocket URL to connect to.
            call_sid: The Call SID for the handshake.
            incoming_queue: Queue to read audio FROM the other agent (to send TO this agent).
            outgoing_queue: Queue to put audio read FROM this agent (to send TO the other agent).
            stop_event: Event to signal stop/cancellation.
            my_ready: Event to signal this connection is ready.
            other_ready: Event to wait for the other connection to be ready.
            chunk_duration_seconds: Duration of one audio chunk in seconds (for write cadence).
            sync_timeout: Timeout for waiting for the other agent to be ready.
            extra_headers: Optional HTTP headers for the initial connection.
            record_sent_callback: Optional callback to record audio sent TO this agent.
        """
        self.name = name
        self.ws_url = ws_url
        self.call_sid = call_sid
        self.incoming_queue = incoming_queue
        self.outgoing_queue = outgoing_queue
        self.stop_event = stop_event
        self.my_ready = my_ready
        self.other_ready = other_ready
        self.chunk_duration_seconds = chunk_duration_seconds
        self.sync_timeout = sync_timeout
        self.extra_headers = extra_headers or {}
        self.record_sent_callback = record_sent_callback

        # Internal state for silence detection
        self.silence_timeout = 0.3  # 300ms

        # We track speaking state for BOTH sides to log interactions coherently
        # Note: In the original code, each connection handler tracked both.
        # This might be redundant if we have two managers, but we keep it to match behavior
        # where we log "User Started speaking" relative to the "Target" connection's timeline if needed.
        # However, strictly speaking, this class primarily knows about *its* agent (Audio IN)
        # and the audio it sends (Audio OUT).

        self.my_speaking = False
        self.my_last_audio_time = None
        self.my_last_stop_time = None

        self.other_speaking = False
        self.other_last_audio_time = None
        self.other_last_stop_time = None

    async def connect(self):
        """Establish connection and run read/write loops."""
        logger.info(f"[{self.name}] Connecting to {self.ws_url}")

        try:
            async with websockets.connect(
                self.ws_url, additional_headers=self.extra_headers
            ) as websocket:
                # Handshake
                await self._perform_handshake(websocket)

                # Synchronization
                if not await self._synchronize_ready_state():
                    return

                # Create tasks
                reader_task = asyncio.create_task(self._read_loop(websocket))
                writer_task = asyncio.create_task(self._write_loop(websocket))
                my_silence_task = asyncio.create_task(self._monitor_my_silence())
                other_silence_task = asyncio.create_task(self._monitor_other_silence())

                # Wait for reader to finish (usually on stop event or connection close)
                await reader_task

                # Cancel others
                writer_task.cancel()
                my_silence_task.cancel()
                other_silence_task.cancel()

                try:
                    await writer_task
                    await my_silence_task
                    await other_silence_task
                except asyncio.CancelledError:
                    pass

        except Exception as e:
            logger.error(f"[{self.name}] Connection failed: {e}")
            raise

    async def _perform_handshake(self, websocket):
        """Send the start event."""
        start_event_msg = {
            "event": "start",
            "sequenceNumber": "1",
            "start": {
                "accountSid": "AC_SIMULATION",
                "callSid": self.call_sid,
                "streamSid": f"stream_{self.call_sid}",
                "tracks": ["inbound"],
                "customParameters": {},
            },
            "streamSid": f"stream_{self.call_sid}",
        }
        await websocket.send(json.dumps(start_event_msg))
        logger.info(f"[{self.name}] Sent start event")

    async def _synchronize_ready_state(self) -> bool:
        """Signal readiness and wait for the partner connection."""
        self.my_ready.set()
        try:
            await asyncio.wait_for(self.other_ready.wait(), timeout=self.sync_timeout)
            logger.info(f"[{self.name}] Both connections ready, starting audio flow")
            return True
        except asyncio.TimeoutError:
            logger.error(
                f"[{self.name}] Other connection not ready in {self.sync_timeout}s, aborting"
            )
            return False

    async def _read_loop(self, websocket):
        """Read messages from this agent (Audio IN)."""
        try:
            async for message in websocket:
                if self.stop_event.is_set():
                    break

                data = json.loads(message)
                event_type = data.get("event")

                if event_type == "media":
                    payload = data["media"]["payload"]
                    audio_bytes = base64.b64decode(payload)
                    current_time = time.time()

                    # Log when this agent starts speaking
                    if not self.my_speaking:
                        gap = ""
                        if self.my_last_stop_time is not None:
                            gap = f" (gap: {current_time - self.my_last_stop_time:.3f}s)"
                        logger.info(f"[{self.name}] Started speaking{gap}")
                        self.my_speaking = True

                    self.my_last_audio_time = current_time

                    # Send to the other agent's queue
                    await self.outgoing_queue.put(audio_bytes)

                    # Monitor queue depth
                    queue_depth = self.outgoing_queue.qsize()
                    if queue_depth > 10:
                        logger.warning(f"[{self.name}-READ] Queue backing up: {queue_depth} items")

                elif event_type == "mark":
                    logger.info(f"[{self.name}] Mark: {data.get('mark', {}).get('name')}")
                elif event_type == "clear":
                    logger.info(f"[{self.name}] Clear received")
                    # Clear incoming queue (audio destined for this agent)
                    while not self.incoming_queue.empty():
                        try:
                            self.incoming_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                elif event_type == "stop":
                    logger.info(f"[{self.name}] Stop received")
                    break
        except Exception as e:
            logger.error(f"[{self.name}] Read error: {e}")

    async def _write_loop(self, websocket):
        """Write audio to this agent (Audio OUT) from the incoming queue."""
        stream_sid = f"stream_{self.call_sid}"
        next_tick = time.time()
        audio_chunks_sent = 0
        silence_chunks_sent = 0

        # Determine the name of the "Other" agent for logging purposes
        other_name = "User" if self.name == "Target" else "Target"

        try:
            while not self.stop_event.is_set():
                # Maintain configurable cadence
                next_tick += self.chunk_duration_seconds
                sleep_time = next_tick - time.time()
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

                # Check cadence drift
                actual_time = time.time()
                drift_ms = (actual_time - next_tick) * 1000
                if abs(drift_ms) > 3:
                    logger.warning(f"[{self.name}-WRITE] Cadence drift: {drift_ms:.2f}ms")

                payload_to_send = None
                queue_depth_before = self.incoming_queue.qsize()

                try:
                    # Try to get audio from the other agent
                    audio_data = await asyncio.wait_for(
                        self.incoming_queue.get(), timeout=self.chunk_duration_seconds + 0.01
                    )

                    if self.record_sent_callback:
                        self.record_sent_callback(audio_data)

                    payload_to_send = base64.b64encode(audio_data).decode("utf-8")
                    audio_chunks_sent += 1
                    current_time = time.time()

                    # Log when the OTHER agent (whose audio we are sending) starts speaking
                    if not self.other_speaking:
                        gap = ""
                        if self.other_last_stop_time is not None:
                            gap = f" (gap: {current_time - self.other_last_stop_time:.3f}s)"
                        logger.info(f"[{other_name}] Started speaking{gap}")
                        self.other_speaking = True

                    self.other_last_audio_time = current_time

                except asyncio.TimeoutError:
                    # Timeout - send silence
                    # Only log generic warning if queue had items but we timed out?
                    # Actually existing logic logged this warning on timeout if items existed
                    if queue_depth_before > 0:
                        logger.warning(
                            f"[{self.name}-WRITE] ⚠️ TIMEOUT! Queue had {queue_depth_before} items. "
                            f"Sending SILENCE."
                        )
                    pass
                except Exception as queue_error:
                    logger.warning(
                        f"[{self.name}] Queue access error: {queue_error}, sending silence"
                    )
                    payload_to_send = None

                # Send silence if no audio
                if not payload_to_send:
                    chunk_size = int(8000 * self.chunk_duration_seconds)
                    silence = b"\xff" * chunk_size
                    silence_chunks_sent += 1

                    if self.record_sent_callback:
                        self.record_sent_callback(silence)

                    payload_to_send = base64.b64encode(silence).decode("utf-8")

                    if silence_chunks_sent % 50 == 0:  # Reduced frequency of silence logging
                        logger.debug(
                            f"[{self.name}-WRITE] Sent {silence_chunks_sent} silence chunks"
                        )

                try:
                    media_event = {
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": payload_to_send},
                    }
                    await websocket.send(json.dumps(media_event))
                except (ConnectionClosedOK, ConnectionClosedError):
                    logger.info(f"[{self.name}] WebSocket connection closed during write")
                    break
                except Exception as send_error:
                    logger.warning(f"[{self.name}] Send error: {send_error}")

        except asyncio.CancelledError:
            pass
        except (ConnectionClosedOK, ConnectionClosedError):
            logger.debug(f"[{self.name}] WebSocket closed normally")
        except Exception as e:
            logger.error(f"[{self.name}] Write error: {e}")

    async def _monitor_my_silence(self):
        """Monitor if THIS agent stops speaking."""
        while not self.stop_event.is_set():
            await asyncio.sleep(0.1)
            if self.my_speaking and self.my_last_audio_time:
                current_time = time.time()
                if current_time - self.my_last_audio_time > self.silence_timeout:
                    logger.info(f"[{self.name}] Stopped speaking")
                    self.my_speaking = False
                    self.my_last_stop_time = current_time
                    self.my_last_audio_time = None

    async def _monitor_other_silence(self):
        """Monitor if the OTHER agent stops speaking (based on what we write)."""
        other_name = "User" if self.name == "Target" else "Target"
        while not self.stop_event.is_set():
            await asyncio.sleep(0.1)
            if self.other_speaking and self.other_last_audio_time:
                current_time = time.time()
                if current_time - self.other_last_audio_time > self.silence_timeout:
                    logger.info(f"[{other_name}] Stopped speaking")
                    self.other_speaking = False
                    self.other_last_stop_time = current_time
                    self.other_last_audio_time = None
