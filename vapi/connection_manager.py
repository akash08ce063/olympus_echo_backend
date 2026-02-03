"""
Vapi connection manager for the main stream bridge.

Uses Vapi REST API to create a call with WebSocket transport (same approach as
the working VapiToTwilioBridge): create call via POST, connect to websocketCallUrl,
then bridge audio bidirectionally with PCM 16k 16le on the Vapi side and
µ-law 8k on the queue side (to match Twilio/Pranthora).
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

# Vapi WebSocket: PCM 16-bit 16kHz
VAPI_SAMPLE_RATE = 16000
VAPI_BYTES_PER_MS = (VAPI_SAMPLE_RATE * 2) // 1000  # 32 bytes/ms
# Twilio/Pranthora queue: µ-law 8kHz, 20ms chunks = 160 bytes
QUEUE_SAMPLE_RATE = 8000
CHUNK_DURATION_SEC = 0.02
QUEUE_CHUNK_BYTES = int(QUEUE_SAMPLE_RATE * CHUNK_DURATION_SEC)  # 160


class VapiConnectionManager:
    """
    Connects to a Vapi assistant via WebSocket transport and bridges audio
    to/from the other agent's queues (µ-law 8k). Data flows both ways so
    Pranthora can hear Vapi and Vapi can hear Pranthora.
    """

    def __init__(
        self,
        name: str,
        assistant_id: str,
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
        self.assistant_id = assistant_id
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
        self.vapi_ws = None
        self.vapi_call_id: Optional[str] = None

    async def _create_vapi_call(self) -> str:
        """Create Vapi call with WebSocket transport; return websocket URL."""
        url = "https://api.vapi.ai/call"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "assistantId": self.assistant_id,
            "transport": {
                "provider": "vapi.websocket",
                "audioFormat": {
                    "format": "pcm_s16le",
                    "container": "raw",
                    "sampleRate": VAPI_SAMPLE_RATE,
                },
            },
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status not in (200, 201):
                    text = await resp.text()
                    raise RuntimeError(
                        f"Vapi create call failed: {resp.status} - {text}"
                    )
                data = await resp.json()
        self.vapi_call_id = data.get("id")
        ws_url = data.get("transport", {}).get("websocketCallUrl")
        if not ws_url:
            raise RuntimeError("Vapi response missing websocketCallUrl")
        logger.info(f"[{self.name}] Vapi call created: {self.vapi_call_id}")
        return ws_url

    async def connect(self) -> None:
        """Create Vapi call, connect WebSocket, run read/write loops until stop."""
        logger.info(
            f"[{self.name}] Creating Vapi call (assistant {self.assistant_id[:8]}...)"
        )
        ws_url = await self._create_vapi_call()
        logger.info(f"[{self.name}] Connecting to Vapi WebSocket...")
        self.vapi_ws = await websockets.connect(ws_url)
        logger.info(f"[{self.name}] Connected to Vapi WebSocket")

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
            if self.vapi_ws:
                try:
                    await self.vapi_ws.send(
                        json.dumps({"type": "end-call"})
                    )
                    await asyncio.sleep(0.2)
                except Exception:
                    pass
                try:
                    await self.vapi_ws.close()
                except Exception:
                    pass
                self.vapi_ws = None
            logger.info(f"[{self.name}] Vapi call stopped")

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
            from_sample_rate=VAPI_SAMPLE_RATE,
            to_sample_rate=QUEUE_SAMPLE_RATE,
            sample_width=2,
        )

    def _ulaw_8k_to_pcm_16k(self, ulaw_bytes: bytes) -> bytes:
        """Convert µ-law 8kHz to PCM 16-bit 16kHz for Vapi."""
        if not ulaw_bytes:
            return b""
        return AudioConverter.convert_and_resample(
            ulaw_bytes,
            from_encoding="mulaw",
            to_encoding="pcm16",
            from_sample_rate=QUEUE_SAMPLE_RATE,
            to_sample_rate=VAPI_SAMPLE_RATE,
            sample_width=2,
        )

    async def _read_loop(self) -> None:
        """Read from Vapi WebSocket (binary + JSON) and push audio to outgoing_queue."""
        try:
            async for message in self.vapi_ws:
                if self.stop_event.is_set():
                    break

                if isinstance(message, bytes):
                    # Vapi agent audio: PCM 16k 16le → convert to µ-law 8k for queue
                    ulaw_chunk = self._pcm_16k_to_ulaw_8k(message)
                    if ulaw_chunk:
                        self.my_speaking = True
                        self.my_last_audio_time = time.time()
                        await self.outgoing_queue.put(ulaw_chunk)
                else:
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type", "")
                        if msg_type == "transcript" and data.get("isFinal"):
                            logger.info(
                                f"[{self.name}] Vapi: {data.get('transcript', '')}"
                            )
                        elif msg_type == "speech-update":
                            logger.debug(
                                f"[{self.name}] speech-update: "
                                f"{data.get('status', '')}"
                            )
                    except json.JSONDecodeError:
                        pass
        except (ConnectionClosedOK, ConnectionClosedError):
            logger.info(f"[{self.name}] Vapi WebSocket closed")
        except Exception as e:
            logger.error(f"[{self.name}] Read error: {e}")
        finally:
            self.stop_event.set()

    async def _write_loop(self) -> None:
        """Read from incoming_queue (µ-law 8k) and send PCM 16k to Vapi WebSocket."""
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
                    # Send silence (µ-law 0xff) so Vapi gets continuous stream
                    ulaw_chunk = b"\xff" * QUEUE_CHUNK_BYTES

                if self.record_sent_callback:
                    self.record_sent_callback(ulaw_chunk)

                pcm_chunk = self._ulaw_8k_to_pcm_16k(ulaw_chunk)
                if not pcm_chunk:
                    continue

                try:
                    await self.vapi_ws.send(pcm_chunk)
                except (ConnectionClosedOK, ConnectionClosedError):
                    logger.info(
                        f"[{self.name}] Vapi WebSocket closed during write"
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
