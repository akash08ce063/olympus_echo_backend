"""
Web scaled testing service for concurrent WebSocket connections.

This module provides functionality to test multiple concurrent WebSocket connections
between target agent (media-stream, 8k mulaw) and user agent (web-media-stream, 16k PCM).
Copies the exact logic from test.py.
"""

import asyncio
import base64
import json
import time
import uuid
import wave
import audioop
from pathlib import Path
from typing import Optional

import websockets
from websockets.exceptions import ConnectionClosed

from services.audio_converter import AudioConverter
from telemetrics.logger import logger


class WebScaledTestingService:
    """Service for managing scaled WebSocket testing with media-stream and web-media-stream."""

    def __init__(
        self,
        target_agent_uri: str,
        user_agent_id: str,
        ws_url_base: str = "ws://localhost:5050",
        recording_path: str = "test_suite_recordings",
    ):
        """
        Initialize the web scaled testing service.

        Args:
            target_agent_uri: Full WebSocket URL for target agent (media-stream endpoint)
            user_agent_id: Agent ID for user agent (will use web-media-stream endpoint)
            ws_url_base: Base WebSocket URL (default: ws://localhost:5050)
            recording_path: Base directory for storing recordings
        """
        self.target_agent_uri = target_agent_uri
        self.user_agent_id = user_agent_id
        self.ws_url_base = ws_url_base.rstrip("/")
        self.recording_path = Path(recording_path)
        self.recording_path.mkdir(parents=True, exist_ok=True)

        # Audio settings for media-stream (target agent)
        self.target_chunk_size = 160  # 20ms @ 8kHz
        self.target_sample_rate = 8000
        self.target_silence_byte = b"\xff"  # μ-law silence

        # Audio settings for web-media-stream (user agent)
        self.user_chunk_size = 320  # 20ms @ 16kHz = 320 samples
        self.user_sample_rate = 16000
        # PCM16 silence: 320 samples * 2 bytes per sample = 640 bytes of zeros
        self.user_silence_bytes = bytes(640)  # 640 bytes of zeros

    async def run_concurrent_test(
        self,
        concurrent_requests: int,
        timeout: int,
        test_id: Optional[str] = None,
    ) -> dict:
        """
        Run concurrent WebSocket connections for scaled testing.

        Args:
            concurrent_requests: Number of parallel WebSocket connections
            timeout: Timeout in seconds after which all connections close
            test_id: Optional test ID for organizing recordings

        Returns:
            Dictionary with test results and statistics
        """
        test_id = test_id or str(uuid.uuid4())
        test_dir = self.recording_path / test_id
        test_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Starting web scaled test: {concurrent_requests} concurrent connections, "
            f"timeout: {timeout}s, test_id: {test_id}"
        )

        # Create tasks for all concurrent connections
        tasks = []
        for conn_num in range(concurrent_requests):
            task = asyncio.create_task(
                self._run_single_connection(conn_num, test_dir, timeout)
            )
            tasks.append(task)

        # Wait for all tasks or timeout
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        # Process results
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        failed = len(results) - successful

        test_summary = {
            "test_id": test_id,
            "concurrent_requests": concurrent_requests,
            "timeout": timeout,
            "successful_connections": successful,
            "failed_connections": failed,
            "duration_seconds": end_time - start_time,
            "recording_path": str(test_dir),
            "results": results,
        }

        logger.info(
            f"Web scaled test completed: {successful}/{concurrent_requests} successful, "
            f"duration: {test_summary['duration_seconds']:.2f}s"
        )

        return test_summary

    async def _run_single_connection(
        self,
        conn_num: int,
        test_dir: Path,
        timeout: int,
    ) -> dict:
        """
        Run a single WebSocket connection between target and user agents.

        Args:
            conn_num: Connection number (for identification)
            test_dir: Directory to save recordings
            timeout: Timeout in seconds

        Returns:
            Dictionary with connection results
        """
        call_sid_target = str(uuid.uuid4())

        # Recording storage (PCM16 at 8kHz from target agent)
        pcm_frames = bytearray()
        stop_event = asyncio.Event()

        def record_ulaw_to_pcm(ulaw_audio: bytes):
            """Convert μ-law → PCM16 and store. Recording ONLY happens here (read side)."""
            pcm = audioop.ulaw2lin(ulaw_audio, 2)  # 16-bit PCM
            pcm_frames.extend(pcm)

        try:
            # Create queues for audio routing
            target_to_user_queue = asyncio.Queue()
            user_to_target_queue = asyncio.Queue()

            # Start both agent connections
            target_task = asyncio.create_task(
                self._target_agent_connection(
                    f"Target-{conn_num}",
                    call_sid_target,
                    incoming_queue=user_to_target_queue,
                    outgoing_queue=target_to_user_queue,
                    stop_event=stop_event,
                    record_callback=record_ulaw_to_pcm,
                )
            )

            user_task = asyncio.create_task(
                self._user_agent_connection(
                    f"User-{conn_num}",
                    incoming_queue=target_to_user_queue,
                    outgoing_queue=user_to_target_queue,
                    stop_event=stop_event,
                )
            )

            # Wait for timeout or until one connection fails
            try:
                await asyncio.wait_for(
                    asyncio.gather(target_task, user_task, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.info(f"[Conn-{conn_num}] Timeout reached, stopping connection")
            finally:
                stop_event.set()
                target_task.cancel()
                user_task.cancel()
                await asyncio.gather(target_task, user_task, return_exceptions=True)

            # Save recording
            recording_file = test_dir / f"conn_{conn_num}.wav"
            if pcm_frames:
                with wave.open(str(recording_file), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit PCM
                    wf.setframerate(self.target_sample_rate)  # 8kHz
                    wf.writeframes(pcm_frames)

                logger.info(
                    f"[Conn-{conn_num}] Recording saved: {recording_file} "
                    f"({len(pcm_frames)} bytes)"
                )

            return {
                "success": True,
                "connection_number": conn_num,
                "recording_file": str(recording_file),
                "audio_bytes": len(pcm_frames),
            }

        except Exception as e:
            logger.error(f"[Conn-{conn_num}] Connection failed: {e}")
            return {
                "success": False,
                "connection_number": conn_num,
                "error": str(e),
            }

    async def _target_agent_connection(
        self,
        name: str,
        call_sid: str,
        incoming_queue: asyncio.Queue,
        outgoing_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        record_callback: callable,
    ):
        """
        Manage target agent WebSocket connection (media-stream endpoint).

        This follows the exact same pattern as test.py:
        - Reads audio from agent -> outgoing_queue (8k mulaw)
        - Writes audio from incoming_queue -> agent (20ms cadence, 8k mulaw)
        - Converts incoming PCM16 (16k) to mulaw (8k) before sending

        Args:
            name: Connection name for logging
            call_sid: Call SID for this connection
            incoming_queue: Queue for audio to send to agent (PCM16 16k)
            outgoing_queue: Queue for audio received from agent (mulaw 8k)
            stop_event: Event to signal stop
            record_callback: Callback to record audio
        """
        # Build URL - use target_agent_uri if it's a full URL, otherwise construct it
        if self.target_agent_uri.startswith("ws://") or self.target_agent_uri.startswith("wss://"):
            ws_url = self.target_agent_uri
            # Append call_sid if not present
            if "call_sid=" not in ws_url:
                separator = "&" if "?" in ws_url else "?"
                ws_url = f"{ws_url}{separator}call_sid={call_sid}"
        else:
            # Assume it's an agent ID and construct URL
            ws_url = f"{self.ws_url_base}/api/call/media-stream/agents/{self.target_agent_uri}?call_sid={call_sid}"

        logger.info(f"[{name}] Connecting to {ws_url}")

        try:
            async with websockets.connect(ws_url) as websocket:
                # -------- START EVENT --------
                start_event = {
                    "event": "start",
                    "sequenceNumber": "1",
                    "start": {
                        "accountSid": "AC_SIMULATION",
                        "callSid": call_sid,
                        "streamSid": f"stream_{call_sid}",
                        "tracks": ["inbound"],
                        "customParameters": {},
                    },
                    "streamSid": f"stream_{call_sid}",
                }

                await websocket.send(json.dumps(start_event))
                logger.info(f"[{name}] Sent start event")

                # -------- READ FROM AGENT --------
                async def read_from_ws():
                    try:
                        async for message in websocket:
                            if stop_event.is_set():
                                break

                            data = json.loads(message)
                            event_type = data.get("event")

                            if event_type == "media":
                                payload = data["media"]["payload"]
                                ulaw_audio = base64.b64decode(payload)

                                # ✅ RECORD ONLY ON READ (NO DUPLICATION)
                                record_callback(ulaw_audio)

                                # Convert mulaw (8k) to PCM16 (16k) for user agent
                                pcm_8k = audioop.ulaw2lin(ulaw_audio, 2)
                                pcm_16k = AudioConverter.resample_audio(
                                    pcm_8k,
                                    from_sample_rate=self.target_sample_rate,
                                    to_sample_rate=self.user_sample_rate,
                                    sample_width=2,
                                    encoding="pcm16",
                                )
                                await outgoing_queue.put(pcm_16k)

                            elif event_type == "mark":
                                logger.info(
                                    f"[{name}] Mark: {data.get('mark', {}).get('name')}"
                                )

                            elif event_type == "clear":
                                logger.info(f"[{name}] Clear received")
                                while not incoming_queue.empty():
                                    try:
                                        incoming_queue.get_nowait()
                                    except asyncio.QueueEmpty:
                                        break

                            elif event_type == "stop":
                                logger.info(f"[{name}] Stop received")
                                break

                    except ConnectionClosed:
                        logger.info(f"[{name}] WebSocket connection closed")
                    except Exception as e:
                        logger.error(f"[{name}] Read error: {e}")

                # -------- WRITE TO AGENT (20ms) --------
                async def write_to_ws():
                    stream_sid = f"stream_{call_sid}"
                    next_tick = time.time()

                    try:
                        while not stop_event.is_set():
                            next_tick += 0.02
                            sleep_time = next_tick - time.time()
                            if sleep_time > 0:
                                await asyncio.sleep(sleep_time)

                            payload_b64 = None

                            if not incoming_queue.empty():
                                try:
                                    # Get PCM16 (16k) from user agent
                                    pcm_16k = incoming_queue.get_nowait()

                                    # Convert PCM16 (16k) to mulaw (8k) for target agent
                                    pcm_8k = AudioConverter.resample_audio(
                                        pcm_16k,
                                        from_sample_rate=self.user_sample_rate,
                                        to_sample_rate=self.target_sample_rate,
                                        sample_width=2,
                                        encoding="pcm16",
                                    )
                                    ulaw_audio = audioop.lin2ulaw(pcm_8k, 2)
                                    payload_b64 = base64.b64encode(ulaw_audio).decode()

                                except asyncio.QueueEmpty:
                                    pass

                            if not payload_b64:
                                silence = self.target_silence_byte * self.target_chunk_size
                                payload_b64 = base64.b64encode(silence).decode()

                            media_event = {
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {"payload": payload_b64},
                            }

                            await websocket.send(json.dumps(media_event))

                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"[{name}] Write error: {e}")

                reader_task = asyncio.create_task(read_from_ws())
                writer_task = asyncio.create_task(write_to_ws())

                await reader_task
                writer_task.cancel()
                try:
                    await writer_task
                except asyncio.CancelledError:
                    pass

        except Exception as e:
            logger.error(f"[{name}] Connection failed: {e}")
            raise

    async def _user_agent_connection(
        self,
        name: str,
        incoming_queue: asyncio.Queue,
        outgoing_queue: asyncio.Queue,
        stop_event: asyncio.Event,
    ):
        """
        Manage user agent WebSocket connection (web-media-stream endpoint).

        This handles raw PCM16 bytes (16kHz) - no JSON wrapping:
        - Reads raw PCM16 bytes from agent -> outgoing_queue
        - Writes raw PCM16 bytes from incoming_queue -> agent

        Args:
            name: Connection name for logging
            incoming_queue: Queue for audio to send to agent (PCM16 16k)
            outgoing_queue: Queue for audio received from agent (PCM16 16k)
            stop_event: Event to signal stop
        """
        # Build web-media-stream URL
        ws_url = f"{self.ws_url_base}/api/call/web-media-stream?agent_id={self.user_agent_id}"

        logger.info(f"[{name}] Connecting to {ws_url}")

        try:
            async with websockets.connect(ws_url) as websocket:
                # Wait for start_media_streaming message
                try:
                    start_message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    if isinstance(start_message, str):
                        start_data = json.loads(start_message)
                        if start_data.get("event_type") == "start_media_streaming":
                            logger.info(f"[{name}] Received start_media_streaming")
                except asyncio.TimeoutError:
                    logger.warning(f"[{name}] Timeout waiting for start_media_streaming")

                # -------- READ FROM AGENT (raw PCM16 bytes) --------
                async def read_from_ws():
                    try:
                        while not stop_event.is_set():
                            try:
                                # Receive raw PCM16 bytes (16kHz)
                                audio_bytes = await asyncio.wait_for(
                                    websocket.recv(), timeout=1.0
                                )
                                if isinstance(audio_bytes, bytes):
                                    await outgoing_queue.put(audio_bytes)
                            except asyncio.TimeoutError:
                                continue
                            except ConnectionClosed:
                                logger.info(f"[{name}] WebSocket connection closed")
                                break
                    except Exception as e:
                        logger.error(f"[{name}] Read error: {e}")

                # -------- WRITE TO AGENT (raw PCM16 bytes, 20ms chunks) --------
                async def write_to_ws():
                    next_tick = time.time()

                    try:
                        while not stop_event.is_set():
                            next_tick += 0.02
                            sleep_time = next_tick - time.time()
                            if sleep_time > 0:
                                await asyncio.sleep(sleep_time)

                            audio_data = None

                            if not incoming_queue.empty():
                                try:
                                    audio_data = incoming_queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    pass

                            if not audio_data:
                                # Send silence (PCM16, 16kHz, 20ms = 320 samples = 640 bytes)
                                audio_data = self.user_silence_bytes

                            # Send raw PCM16 bytes
                            await websocket.send(audio_data)

                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"[{name}] Write error: {e}")

                reader_task = asyncio.create_task(read_from_ws())
                writer_task = asyncio.create_task(write_to_ws())

                await reader_task
                writer_task.cancel()
                try:
                    await writer_task
                except asyncio.CancelledError:
                    pass

        except Exception as e:
            logger.error(f"[{name}] Connection failed: {e}")
            raise

