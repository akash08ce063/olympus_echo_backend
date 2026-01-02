"""
Scaled testing service for concurrent WebSocket connections.

This module provides functionality to test multiple concurrent WebSocket
connections between target and user agents with audio conversion support.
"""

import asyncio
import audioop
import base64
import json
import time
import uuid
import wave
from pathlib import Path
from typing import Optional

import websockets
from websockets.exceptions import ConnectionClosed
from services.audio_converter import AudioConverter, EncodingType, SampleRateType
from services.recording_storage_service import RecordingStorageService
from telemetrics.logger import logger


class ScaledTestingService:
    """Service for managing scaled WebSocket testing with audio conversion."""

    def __init__(
        self,
        target_agent_uri: str,
        user_agent_uri: str,
        sample_rate: SampleRateType,
        encoding: EncodingType,
        recording_path: str = "test_suite_recordings",
    ):
        """
        Initialize the scaled testing service.

        Args:
            target_agent_uri: WebSocket URL for the target agent
            user_agent_uri: WebSocket URL for the user agent
            sample_rate: Sample rate of audio from target agent
            encoding: Encoding of audio from target agent (mulaw, pcm16, pcm)
            recording_path: Base directory for storing recordings (deprecated - now uses Supabase)
        """
        self.target_agent_uri = target_agent_uri
        self.user_agent_uri = user_agent_uri
        self.sample_rate = sample_rate
        self.encoding = encoding.lower()  # Normalize to lowercase
        self.recording_path = Path(recording_path)
        self.recording_path.mkdir(parents=True, exist_ok=True)
        self.recording_service = RecordingStorageService()

        # Audio settings based on encoding
        self.chunk_size = 160  # 20ms @ 8kHz default
        if sample_rate == 8000:
            self.chunk_size = 160
        elif sample_rate == 16000:
            self.chunk_size = 320
        else:
            # Calculate chunk size for 20ms
            self.chunk_size = int(sample_rate * 0.02)

        # Silence byte based on encoding
        if encoding == "mulaw":
            self.silence_byte = b"\xff"  # μ-law silence
        else:
            self.silence_byte = b"\x00"  # PCM silence

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
            f"Starting scaled test: {concurrent_requests} concurrent connections, "
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
            f"Scaled test completed: {successful}/{concurrent_requests} successful, "
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
        call_sid_user = str(uuid.uuid4())

        # Recording storage - ONLY for target agent audio
        pcm_frames = bytearray()
        stop_event = asyncio.Event()

        def record_ulaw_to_pcm(audio_data: bytes):
            """
            Convert audio to PCM16 and record.
            Recording ONLY happens here (read side from target agent).
            """
            try:
                # Convert to PCM16 for recording based on encoding
                if self.encoding == "mulaw":
                    pcm = audioop.ulaw2lin(audio_data, 2)  # 16-bit PCM
                elif self.encoding == "pcm16":
                    pcm = audio_data  # Already PCM16
                else:
                    # For other PCM formats, assume it's already in correct format
                    pcm = audio_data
                
                pcm_frames.extend(pcm)
            except Exception as e:
                logger.error(f"[Conn-{conn_num}] Recording error: {e}")

        try:
            # Create queues for audio routing
            target_to_user_queue = asyncio.Queue()
            user_to_target_queue = asyncio.Queue()

            # Start both agent connections
            target_task = asyncio.create_task(
                self._agent_connection(
                    f"Target-{conn_num}",
                    self.target_agent_uri,
                    call_sid_target,
                    incoming_queue=user_to_target_queue,
                    outgoing_queue=target_to_user_queue,
                    stop_event=stop_event,
                    record_callback=record_ulaw_to_pcm,  # Record from target
                )
            )

            user_task = asyncio.create_task(
                self._agent_connection(
                    f"User-{conn_num}",
                    self.user_agent_uri,
                    call_sid_user,
                    incoming_queue=target_to_user_queue,
                    outgoing_queue=user_to_target_queue,
                    stop_event=stop_event,
                    record_callback=None,  # Don't record from user
                )
            )

            # Wait for timeout
            await asyncio.sleep(timeout)
            logger.info(f"[Conn-{conn_num}] Timeout reached, shutting down connection")

            # Stop connections
            stop_event.set()
            target_task.cancel()
            user_task.cancel()
            await asyncio.gather(target_task, user_task, return_exceptions=True)

            # Upload recording to Supabase storage
            recording_file_id = None
            if pcm_frames:
                # Convert PCM frames to WAV format
                import io
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(pcm_frames)

                wav_data = wav_buffer.getvalue()

                # Upload to Supabase storage
                recording_file_id = await self.recording_service.upload_recording_file(
                    file_content=wav_data,
                    file_name=f"connection_{conn_num}.wav",
                    content_type="audio/wav"
                )

                if recording_file_id:
                    logger.info(
                        f"[Conn-{conn_num}] Recording uploaded to Supabase: {recording_file_id} "
                        f"({len(pcm_frames)} PCM bytes)"
                    )
                else:
                    logger.warning(f"[Conn-{conn_num}] Failed to upload recording to Supabase")
            else:
                logger.warning(f"[Conn-{conn_num}] No audio recorded")

            return {
                "success": True,
                "connection_number": conn_num,
                "recording_file_id": str(recording_file_id) if recording_file_id else None,
                "audio_bytes": len(pcm_frames),
            }

        except Exception as e:
            logger.error(f"[Conn-{conn_num}] Connection failed: {e}")
            return {
                "success": False,
                "connection_number": conn_num,
                "error": str(e),
            }

    async def _agent_connection(
        self,
        name: str,
        ws_url: str,
        call_sid: str,
        incoming_queue: asyncio.Queue,
        outgoing_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        record_callback: Optional[callable] = None,
    ):
        """
        Manage a single agent WebSocket connection.

        Pattern from test.py:
        - Reads audio from agent -> outgoing_queue (and records if callback provided)
        - Writes audio from incoming_queue -> agent (20ms cadence)

        Args:
            name: Connection name for logging
            ws_url: WebSocket URL to connect to
            call_sid: Call SID for this connection
            incoming_queue: Queue for audio to send to agent
            outgoing_queue: Queue for audio received from agent
            stop_event: Event to signal stop
            record_callback: Optional callback to record audio (target agent only)
        """
        # Handle URL with or without call_sid parameter
        if "call_sid=" not in ws_url:
            separator = "&" if "?" in ws_url else "?"
            ws_url = f"{ws_url}{separator}call_sid={call_sid}"

        logger.info(f"[{name}] Connecting to {ws_url}")

        try:
            async with websockets.connect(ws_url) as websocket:
                # Send start event
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
                                audio_bytes = base64.b64decode(payload)

                                # ✅ RECORD ONLY ON READ (from target agent)
                                if record_callback:
                                    record_callback(audio_bytes)

                                # Put audio in outgoing queue (no duplication)
                                await outgoing_queue.put(audio_bytes)

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

                # -------- WRITE TO AGENT (20ms cadence) --------
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
                                    audio = incoming_queue.get_nowait()
                                    payload_b64 = base64.b64encode(audio).decode()
                                except asyncio.QueueEmpty:
                                    pass

                            if not payload_b64:
                                silence = self.silence_byte * self.chunk_size
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