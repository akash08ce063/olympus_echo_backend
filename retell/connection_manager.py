"""
Retell AI connection manager for the main stream bridge.

Uses Retell REST API to create a web call with WebSocket transport (similar to Vapi):
create call via POST, connect to websocket URL, then bridge audio bidirectionally
with PCM 16k 16le on the Retell side and µ-law 8k on the queue side (to match Twilio/Pranthora).
"""

import asyncio
import json
import time
from typing import Optional, Callable

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

from services.audio_converter import AudioConverter
from telemetrics.logger import logger

# Retell WebSocket: PCM 16-bit 16kHz (similar to Vapi)
RETELL_SAMPLE_RATE = 16000
RETELL_BYTES_PER_MS = (RETELL_SAMPLE_RATE * 2) // 1000  # 32 bytes/ms
# Twilio/Pranthora queue: µ-law 8kHz, 20ms chunks = 160 bytes
QUEUE_SAMPLE_RATE = 8000
CHUNK_DURATION_SEC = 0.02
QUEUE_CHUNK_BYTES = int(QUEUE_SAMPLE_RATE * CHUNK_DURATION_SEC)  # 160


class RetellConnectionManager:
    """
    Connects to a Retell agent via WebSocket transport and bridges audio
    to/from the other agent's queues (µ-law 8k). Data flows both ways so
    Pranthora can hear Retell and Retell can hear Pranthora.
    """

    def __init__(
        self,
        name: str,
        agent_id: str,
        api_key: str,
        call_sid: str,
        incoming_queue: asyncio.Queue,
        outgoing_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        my_ready: asyncio.Event,
        other_ready: asyncio.Event,
        chunk_duration_seconds: float = 0.02,
        sync_timeout: float = 10.0,
        record_sent_callback: Optional[Callable[[bytes], None]] = None,
    ):
        self.name = name
        self.agent_id = agent_id
        self.api_key = api_key
        self.call_sid = call_sid
        self.incoming_queue = incoming_queue
        self.outgoing_queue = outgoing_queue
        self.stop_event = stop_event
        self.my_ready = my_ready
        self.other_ready = other_ready
        self.chunk_duration_seconds = chunk_duration_seconds
        self.sync_timeout = sync_timeout
        self.record_sent_callback = record_sent_callback
        self.silence_timeout = 0.3
        self.my_speaking = False
        self.my_last_audio_time = None
        self.other_speaking = False
        self.other_last_audio_time = None
        self.other_last_stop_time = None
        self.retell_ws = None
        self.retell_call_id: Optional[str] = None

    async def _create_retell_call(self) -> str:
        """
        Register a Retell call and return the audio WebSocket URL.

        For audio bridging (Twilio/custom telephony) Retell recommends using the
        register-call + audio-websocket flow rather than the web-call SDK flow.
        """
        url = "https://api.retellai.com/register-call"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "agent_id": self.agent_id,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status not in (200, 201):
                    text = await resp.text()
                    raise RuntimeError(
                        f"Retell register-call failed: {resp.status} - {text}"
                    )
                data = await resp.json()

        self.retell_call_id = data.get("call_id")
        if not self.retell_call_id:
            raise RuntimeError("Retell response missing call_id")

        # Retell audio WebSocket URL format: wss://api.retellai.com/audio-websocket/{call_id}
        ws_url = f"wss://api.retellai.com/audio-websocket/{self.retell_call_id}"
        logger.info(f"[{self.name}] Retell call registered: {self.retell_call_id}")
        return ws_url

    async def connect(self) -> None:
        """Create Retell call, connect WebSocket, run read/write loops until stop."""
        logger.info(
            f"[{self.name}] Creating Retell call (agent {self.agent_id[:8]}...)"
        )
        ws_url = await self._create_retell_call()
        logger.info(f"[{self.name}] Connecting to Retell WebSocket...")
        self.retell_ws = await websockets.connect(ws_url)
        logger.info(f"[{self.name}] Connected to Retell WebSocket")

        try:
            if not await self._synchronize_ready_state():
                return

            reader_task = asyncio.create_task(self._read_loop())
            writer_task = asyncio.create_task(self._write_loop())

            await self.stop_event.wait()

            reader_task.cancel()
            writer_task.cancel()
            for t in (reader_task, writer_task):
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        finally:
            if self.retell_ws:
                try:
                    # Send end call message if Retell protocol supports it
                    await asyncio.sleep(0.2)
                except Exception:
                    pass
                try:
                    await self.retell_ws.close()
                except Exception:
                    pass
                self.retell_ws = None
            logger.info(f"[{self.name}] Retell call stopped")

    async def _synchronize_ready_state(self) -> bool:
        self.my_ready.set()
        try:
            await asyncio.wait_for(
                self.other_ready.wait(), timeout=self.sync_timeout
            )
            logger.info(f"[{self.name}] Both connections ready")
            return True
        except asyncio.TimeoutError:
            logger.error(
                f"[{self.name}] Other connection not ready in "
                f"{self.sync_timeout}s"
            )
            return False

    def _pcm_16k_to_ulaw_8k(self, pcm_bytes: bytes) -> bytes:
        """Convert PCM 16-bit 16kHz to µ-law 8kHz for queue/Twilio."""
        if not pcm_bytes:
            return b""
        return AudioConverter.convert_and_resample(
            pcm_bytes,
            from_encoding="pcm16",
            to_encoding="mulaw",
            from_sample_rate=RETELL_SAMPLE_RATE,
            to_sample_rate=QUEUE_SAMPLE_RATE,
            sample_width=2,
        )

    def _ulaw_8k_to_pcm_16k(self, ulaw_bytes: bytes) -> bytes:
        """Convert µ-law 8kHz to PCM 16-bit 16kHz for Retell."""
        if not ulaw_bytes:
            return b""
        return AudioConverter.convert_and_resample(
            ulaw_bytes,
            from_encoding="mulaw",
            to_encoding="pcm16",
            from_sample_rate=QUEUE_SAMPLE_RATE,
            to_sample_rate=RETELL_SAMPLE_RATE,
            sample_width=2,
        )

    async def _read_loop(self) -> None:
        """Read from Retell WebSocket (binary audio) and push audio to outgoing_queue."""
        try:
            async for message in self.retell_ws:
                if self.stop_event.is_set():
                    break

                if isinstance(message, bytes):
                    # Retell agent audio: PCM 16k 16le → convert to µ-law 8k for queue
                    ulaw_chunk = self._pcm_16k_to_ulaw_8k(message)
                    if ulaw_chunk:
                        self.my_speaking = True
                        self.my_last_audio_time = time.time()
                        await self.outgoing_queue.put(ulaw_chunk)
                else:
                    try:
                        data = json.loads(message)
                        # Handle JSON messages from Retell (transcripts, status updates, etc.)
                        logger.debug(f"[{self.name}] Retell message: {data}")
                    except json.JSONDecodeError:
                        pass
        except (ConnectionClosedOK, ConnectionClosedError):
            logger.info(f"[{self.name}] Retell WebSocket closed")
        except Exception as e:
            logger.error(f"[{self.name}] Read error: {e}")
        finally:
            self.stop_event.set()

    async def _write_loop(self) -> None:
        """Read from incoming_queue (µ-law 8k) and send PCM 16k to Retell WebSocket."""
        next_tick = time.time()
        try:
            while not self.stop_event.is_set():
                next_tick += self.chunk_duration_seconds
                sleep_time = next_tick - time.time()
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

                try:
                    ulaw_chunk = await asyncio.wait_for(
                        self.incoming_queue.get(),
                        timeout=self.chunk_duration_seconds + 0.01,
                    )
                except asyncio.TimeoutError:
                    # Send silence (µ-law 0xff) so Retell gets continuous stream
                    ulaw_chunk = b"\xff" * QUEUE_CHUNK_BYTES

                if self.record_sent_callback:
                    self.record_sent_callback(ulaw_chunk)

                pcm_chunk = self._ulaw_8k_to_pcm_16k(ulaw_chunk)
                if not pcm_chunk:
                    continue

                try:
                    await self.retell_ws.send(pcm_chunk)
                except (ConnectionClosedOK, ConnectionClosedError):
                    logger.info(
                        f"[{self.name}] Retell WebSocket closed during write"
                    )
                    break
                except Exception as send_err:
                    logger.warning(
                        f"[{self.name}] Send error: {send_err}"
                    )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[{self.name}] Write error: {e}")
        finally:
            self.stop_event.set()
