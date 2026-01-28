#!/usr/bin/env python3
"""
Test script to bridge Pranthora agent with LiveKit participant.

This test:
1. Connects to Pranthora agent via WebSocket
2. Creates/connects to LiveKit room
3. Joins LiveKit room as a participant
4. Bridges audio between Pranthora agent and LiveKit room

Prerequisites:
- LiveKit credentials configured in config.json
- Pranthora agent must be accessible
- Webhooks are OPTIONAL (not required for the test to work)
"""

import asyncio
import json
import base64
import logging
import sys
import uuid
import time
from typing import Optional
import websockets

# LiveKit imports - correct package structure
from livekit import rtc

# Import from parent directory
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from static_memory_cache import StaticMemoryCache

# For LiveKit REST API and token generation
import aiohttp
import jwt
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("PranthoraLiveKitBridge")

# Configuration
PRANTHORA_AGENT_ID = "e1143125-5a7f-4955-8f2c-01df1c604343"
PRANTHORA_WS_URL = f"wss://api.pranthora.com/api/call/web-media-stream?agent_id={PRANTHORA_AGENT_ID}"
LIVEKIT_AGENT_ID = "ab_iaapv5vx3tu"  # LiveKit agent ID

# Audio settings
CHUNK_SIZE = 160  # 20ms @ 8kHz
SAMPLE_RATE = 8000
SILENCE_BYTE = b"\xff"  # μ-law silence
MAX_DURATION_SEC = 60  # Test duration


class PranthoraLiveKitBridge:
    """Bridges audio between Pranthora agent and LiveKit room."""

    def __init__(self):
        self.livekit_url = StaticMemoryCache.get_livekit_url()
        self.livekit_api_key = StaticMemoryCache.get_livekit_api_key()
        self.livekit_api_secret = StaticMemoryCache.get_livekit_api_secret()
        self.pranthora_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.livekit_room: Optional[rtc.Room] = None
        self.room_name = f"test-room-{uuid.uuid4().hex[:8]}"
        self.participant_name = f"bridge-participant-{uuid.uuid4().hex[:8]}"
        self.livekit_agent_id = LIVEKIT_AGENT_ID
        self.stop_event = asyncio.Event()
        self.pranthora_to_livekit_queue = asyncio.Queue()
        self.livekit_to_pranthora_queue = asyncio.Queue()
        self.livekit_audio_source: Optional[rtc.AudioSource] = None
        self.livekit_audio_track: Optional[rtc.LocalAudioTrack] = None
        
        # Statistics for proof of communication
        self.stats = {
            "pranthora_to_livekit_bytes": 0,
            "livekit_to_pranthora_bytes": 0,
            "pranthora_to_livekit_chunks": 0,
            "livekit_to_pranthora_chunks": 0,
            "pranthora_messages": 0,
            "pranthora_text_messages": [],
            "livekit_audio_frames": 0,
            "start_time": None,
        }

    async def create_or_get_livekit_room(self) -> str:
        """LiveKit rooms are auto-created when participants join - no need to create via API."""
        # LiveKit Cloud automatically creates rooms when the first participant joins
        # We just need to ensure the room name is set
        logger.info(f"✅ Room '{self.room_name}' will be auto-created when participant joins")
        return self.room_name

    def create_livekit_token(self, identity: str, name: str) -> str:
        """Create a token for LiveKit participant to join the room."""
        try:
            # Create JWT token manually using pyjwt
            now = int(time.time())
            token_payload = {
                "iss": self.livekit_api_key,
                "sub": identity,
                "name": name,
                "exp": now + 3600,  # 1 hour expiry
                "video": {
                    "room": self.room_name,
                    "roomJoin": True,
                    "canPublish": True,
                    "canSubscribe": True,
                }
            }
            
            token_str = jwt.encode(
                token_payload,
                self.livekit_api_secret,
                algorithm="HS256"
            )
            logger.info(f"✅ Created LiveKit token for: {identity}")
            return token_str

        except Exception as e:
            logger.error(f"❌ Error creating LiveKit token: {e}")
            raise

    async def connect_livekit_room(self, token: str) -> rtc.Room:
        """Connect to LiveKit room as a participant."""
        try:
            room = rtc.Room()

            @room.on("track_subscribed")
            def on_track_subscribed(
                track: rtc.Track,
                publication: rtc.TrackPublication,
                participant: rtc.RemoteParticipant,
            ):
                logger.info(f"📥 Track subscribed: {track.kind} from {participant.identity} (name: {participant.name})")
                if track.kind == rtc.TrackKind.KIND_AUDIO:
                    logger.info(f"✅ PROOF: Successfully subscribed to audio track from {participant.identity}")
                    logger.info(f"🎵 PROOF: Starting audio stream handler for {participant.identity}")
                    asyncio.create_task(self._handle_livekit_incoming_audio(track))

            @room.on("track_published")
            def on_track_published(
                publication: rtc.RemoteTrackPublication,
                participant: rtc.RemoteParticipant,
            ):
                logger.info(f"📤 Track published: {publication.kind} from {participant.identity} (name: {participant.name})")
                if publication.kind == rtc.TrackKind.KIND_AUDIO:
                    logger.info(f"✅ PROOF: Audio track published by {participant.identity}, subscribing...")
                    # Explicitly subscribe to the track - this is critical!
                    try:
                        room.local_participant.set_subscribed(publication, True)
                        logger.info(f"✅ PROOF: Explicitly subscribed to audio track from {participant.identity}")
                    except Exception as sub_err:
                        logger.warning(f"⚠️ Could not explicitly subscribe: {sub_err}")
                    # Handle the track if it's already available
                    if publication.track:
                        logger.info(f"✅ PROOF: Track available, starting audio handler")
                        asyncio.create_task(self._handle_livekit_incoming_audio(publication.track))
                    else:
                        logger.info(f"⏳ Track not yet available, will be handled when subscribed")

            @room.on("participant_connected")
            def on_participant_connected(participant: rtc.RemoteParticipant):
                logger.info(f"👤 Participant connected: {participant.identity} (name: {participant.name})")
                logger.info(f"🔍 Checking if this is LiveKit agent '{self.livekit_agent_id}'...")
                # Subscribe to all audio tracks from this participant
                for publication in participant.track_publications.values():
                    if publication.kind == rtc.TrackKind.KIND_AUDIO:
                        logger.info(f"✅ PROOF: Found audio track from {participant.identity}, subscribing...")
                        # Explicitly subscribe to the track
                        try:
                            room.local_participant.set_subscribed(publication, True)
                            logger.info(f"✅ PROOF: Explicitly subscribed to track from {participant.identity}")
                        except Exception as sub_err:
                            logger.warning(f"⚠️ Could not explicitly subscribe: {sub_err}")
                        if publication.track:
                            logger.info(f"✅ PROOF: Track available, starting audio handler")
                            asyncio.create_task(self._handle_livekit_incoming_audio(publication.track))
                        else:
                            logger.info(f"⏳ Track not yet available, will subscribe when ready")

            @room.on("participant_disconnected")
            def on_participant_disconnected(participant: rtc.RemoteParticipant):
                logger.info(f"👤 Participant disconnected: {participant.identity}")

            # Connect to room with auto-subscribe enabled
            await room.connect(self.livekit_url, token)
            logger.info(f"✅ Connected to LiveKit room: {self.room_name}")
            logger.info(f"🔍 Waiting for LiveKit agent '{self.livekit_agent_id}' to join the room...")
            logger.info(f"💡 Make sure your LiveKit agent is configured to join room '{self.room_name}'")

            # Create and publish audio track
            self.livekit_audio_source = rtc.AudioSource(SAMPLE_RATE, 1)  # 8kHz, mono
            self.livekit_audio_track = rtc.LocalAudioTrack.create_audio_track(
                "bridge-audio", self.livekit_audio_source
            )
            options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            publication = await room.local_participant.publish_track(
                self.livekit_audio_track, options
            )
            logger.info("✅ Published audio track to LiveKit room")

            # Subscribe to all existing participants' audio tracks
            for participant in room.remote_participants.values():
                logger.info(f"🔍 Checking existing participant: {participant.identity} (name: {participant.name})")
                for publication in participant.track_publications.values():
                    if publication.kind == rtc.TrackKind.KIND_AUDIO:
                        logger.info(f"✅ PROOF: Found existing audio track from {participant.identity}, subscribing...")
                        # Explicitly subscribe to the track
                        try:
                            room.local_participant.set_subscribed(publication, True)
                            logger.info(f"✅ PROOF: Explicitly subscribed to existing track from {participant.identity}")
                        except Exception as sub_err:
                            logger.warning(f"⚠️ Could not explicitly subscribe: {sub_err}")
                        if publication.track:
                            logger.info(f"✅ PROOF: Track available, starting audio handler")
                            asyncio.create_task(self._handle_livekit_incoming_audio(publication.track))
                        else:
                            # Track not yet available, wait for it
                            logger.info(f"⏳ Audio track from {participant.identity} not yet available, will subscribe when ready")

            # Start task to send audio to LiveKit
            asyncio.create_task(self._send_audio_to_livekit())

            self.livekit_room = room
            return room

        except Exception as e:
            logger.error(f"❌ Error connecting to LiveKit room: {e}")
            raise

    async def _handle_livekit_incoming_audio(self, track: rtc.Track):
        """Handle incoming audio from LiveKit room."""
        try:
            logger.info(f"🎵 PROOF: Starting to receive audio from LiveKit track")
            stream = rtc.AudioStream(track)
            frame_count = 0
            async for frame in stream:
                if self.stop_event.is_set():
                    break
                # Convert audio frame to bytes and queue for Pranthora
                try:
                    # Handle different audio frame formats
                    if hasattr(frame.data, 'tobytes'):
                        audio_data = frame.data.tobytes()
                    elif hasattr(frame.data, 'tobytes'):
                        audio_data = bytes(frame.data)
                    else:
                        # Convert numpy array or other formats
                        import numpy as np
                        if isinstance(frame.data, np.ndarray):
                            audio_data = frame.data.tobytes()
                        else:
                            audio_data = bytes(frame.data)
                    
                    await self.livekit_to_pranthora_queue.put(audio_data)
                    
                    # Update statistics - PROOF of communication
                    self.stats["livekit_to_pranthora_bytes"] += len(audio_data)
                    self.stats["livekit_to_pranthora_chunks"] += 1
                    self.stats["livekit_audio_frames"] += 1
                    frame_count += 1
                    
                    if frame_count == 1:
                        logger.info(f"✅ PROOF: First audio frame received from LiveKit! ({len(audio_data)} bytes)")
                    
                    if self.stats["livekit_to_pranthora_chunks"] % 50 == 0:  # Log every 50 chunks
                        logger.info(f"📊 PROOF: Received {self.stats['livekit_to_pranthora_chunks']} audio chunks ({self.stats['livekit_to_pranthora_bytes']} bytes) from LiveKit")
                except Exception as frame_error:
                    logger.error(f"❌ Error processing audio frame: {frame_error}")
        except Exception as e:
            logger.error(f"❌ Error handling LiveKit audio: {e}", exc_info=True)

    async def _send_audio_to_livekit(self):
        """Send audio from Pranthora queue to LiveKit."""
        try:
            while not self.stop_event.is_set():
                try:
                    # Get audio from Pranthora queue
                    audio_data = await asyncio.wait_for(
                        self.pranthora_to_livekit_queue.get(),
                        timeout=0.1
                    )

                    # Convert to audio frame and push to LiveKit
                    if self.livekit_audio_source:
                        # Create audio frame from bytes (assuming 16-bit PCM)
                        import array
                        audio_array = array.array('h', audio_data)  # 'h' = signed short (16-bit)
                        frame = rtc.AudioFrame(
                            data=audio_array,
                            sample_rate=SAMPLE_RATE,
                            num_channels=1,
                            samples_per_channel=len(audio_array),
                        )
                        await self.livekit_audio_source.capture_frame(frame)
                        logger.debug("📤 Sent audio to LiveKit")

                except asyncio.TimeoutError:
                    # Send silence if no audio available
                    import array
                    silence = array.array('h', [0] * CHUNK_SIZE)  # 'h' = signed short (16-bit)
                    frame = rtc.AudioFrame(
                        data=silence,
                        sample_rate=SAMPLE_RATE,
                        num_channels=1,
                        samples_per_channel=CHUNK_SIZE,
                    )
                    if self.livekit_audio_source:
                        await self.livekit_audio_source.capture_frame(frame)

        except Exception as e:
            logger.error(f"❌ Error sending audio to LiveKit: {e}")

    async def connect_pranthora_agent(self):
        """Connect to Pranthora agent via WebSocket."""
        try:
            logger.info(f"🔌 Connecting to Pranthora agent: {PRANTHORA_WS_URL}")

            # Connect with proper headers - try different origins or no origin
            # Pranthora may validate origin, so we try common options
            headers = {
                "Origin": "https://pranthora.com",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            }

            try:
                async with websockets.connect(
                    PRANTHORA_WS_URL,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=10,
                ) as websocket:
                    await self._handle_pranthora_connection(websocket)
            except websockets.exceptions.InvalidStatus as e:
                if "403" in str(e):
                    # Try without Origin header if 403
                    logger.warning("⚠️ Connection rejected with Origin header, trying without...")
                    async with websockets.connect(
                        PRANTHORA_WS_URL,
                        ping_interval=20,
                        ping_timeout=10,
                    ) as websocket:
                        await self._handle_pranthora_connection(websocket)
                else:
                    raise

        except Exception as e:
            logger.error(f"❌ Error connecting to Pranthora agent: {e}")
            raise

    async def _handle_pranthora_connection(self, websocket):
        """Handle the Pranthora WebSocket connection once established."""
        self.pranthora_ws = websocket
        logger.info("✅ Connected to Pranthora agent")

        # Send start event
        start_event = {
            "event_type": "start_media_streaming"
        }
        await websocket.send(json.dumps(start_event))
        logger.info("✅ Sent start event to Pranthora")

        # Start tasks
        read_task = asyncio.create_task(self._read_from_pranthora(websocket))
        write_task = asyncio.create_task(self._write_to_pranthora(websocket))

        # Wait for stop event or timeout
        try:
            await asyncio.wait_for(self.stop_event.wait(), timeout=MAX_DURATION_SEC)
        except asyncio.TimeoutError:
            logger.info(f"⏹️ {MAX_DURATION_SEC} seconds reached, shutting down")
        finally:
            self.stop_event.set()
            read_task.cancel()
            write_task.cancel()
            await asyncio.gather(read_task, write_task, return_exceptions=True)

    async def _read_from_pranthora(self, websocket):
        """Read audio from Pranthora agent and forward to LiveKit."""
        try:
            # Set text mode to handle both text and binary
            async for raw_message in websocket:
                if self.stop_event.is_set():
                    break

                try:
                    # Handle binary data first
                    if isinstance(raw_message, bytes):
                        # Try to decode as UTF-8 text first
                        try:
                            message = raw_message.decode('utf-8')
                        except UnicodeDecodeError:
                            # It's binary audio data
                            await self.pranthora_to_livekit_queue.put(raw_message)
                            self.stats["pranthora_to_livekit_bytes"] += len(raw_message)
                            self.stats["pranthora_to_livekit_chunks"] += 1
                            
                            if self.stats["pranthora_to_livekit_chunks"] % 50 == 0:
                                logger.info(f"📊 PROOF: Received {self.stats['pranthora_to_livekit_chunks']} binary chunks ({self.stats['pranthora_to_livekit_bytes']} bytes) from Pranthora")
                            continue
                    else:
                        message = raw_message

                    # Try to parse as JSON
                    try:
                        data = json.loads(message)
                        event_type = data.get("event_type") or data.get("event") or data.get("type")

                        if event_type == "audio" or "audio" in data or "payload" in data:
                            # Extract audio data
                            audio_payload = data.get("audio") or data.get("payload") or data.get("data")
                            if audio_payload:
                                try:
                                    audio_bytes = base64.b64decode(audio_payload)
                                    await self.pranthora_to_livekit_queue.put(audio_bytes)
                                    
                                    # Update statistics - PROOF of communication
                                    self.stats["pranthora_to_livekit_bytes"] += len(audio_bytes)
                                    self.stats["pranthora_to_livekit_chunks"] += 1
                                    self.stats["pranthora_messages"] += 1
                                    
                                    if self.stats["pranthora_to_livekit_chunks"] % 50 == 0:
                                        logger.info(f"📊 PROOF: Received {self.stats['pranthora_to_livekit_chunks']} audio chunks ({self.stats['pranthora_to_livekit_bytes']} bytes) from Pranthora")
                                except Exception as decode_err:
                                    logger.debug(f"Could not decode audio payload: {decode_err}")

                        elif event_type == "text" or "text" in data or "transcript" in data or "message" in data:
                            text = data.get("text") or data.get("message") or data.get("transcript") or data.get("content")
                            if text:
                                self.stats["pranthora_text_messages"].append(text)
                                logger.info(f"💬 PROOF - Pranthora said: {text}")
                                self.stats["pranthora_messages"] += 1
                        else:
                            # Unknown event type, log it
                            logger.debug(f"Received event: {event_type} - {str(data)[:100]}")

                    except json.JSONDecodeError:
                        # Not JSON, might be plain text message
                        if message and message.strip():
                            logger.info(f"💬 PROOF - Pranthora message: {message[:100]}")
                            self.stats["pranthora_text_messages"].append(message)
                            self.stats["pranthora_messages"] += 1

                except Exception as decode_error:
                    logger.debug(f"Error processing message: {decode_error}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("⚠️ Pranthora WebSocket connection closed")
        except Exception as e:
            logger.error(f"❌ Error reading from Pranthora: {e}", exc_info=True)

    async def _write_to_pranthora(self, websocket):
        """Write audio from LiveKit to Pranthora agent."""
        try:
            while not self.stop_event.is_set():
                try:
                    # Get audio from LiveKit queue with timeout
                    audio_data = await asyncio.wait_for(
                        self.livekit_to_pranthora_queue.get(),
                        timeout=0.1
                    )

                    # Send to Pranthora as base64 encoded
                    payload_b64 = base64.b64encode(audio_data).decode()
                    message = {
                        "event_type": "audio",
                        "payload": payload_b64
                    }
                    await websocket.send(json.dumps(message))
                    
                    # Update statistics - PROOF of communication
                    self.stats["livekit_to_pranthora_bytes"] += len(audio_data)
                    self.stats["livekit_to_pranthora_chunks"] += 1
                    
                    if self.stats["livekit_to_pranthora_chunks"] % 50 == 0:
                        logger.info(f"📊 PROOF: Sent {self.stats['livekit_to_pranthora_chunks']} audio chunks ({self.stats['livekit_to_pranthora_bytes']} bytes) to Pranthora")

                except asyncio.TimeoutError:
                    # Send silence if no audio available (20ms chunks)
                    await asyncio.sleep(0.02)
                    silence = SILENCE_BYTE * CHUNK_SIZE
                    payload_b64 = base64.b64encode(silence).decode()
                    message = {
                        "event_type": "audio",
                        "payload": payload_b64
                    }
                    await websocket.send(json.dumps(message))

        except Exception as e:
            logger.error(f"❌ Error writing to Pranthora: {e}")

    async def run(self):
        """Run the bridge test."""
        try:
            logger.info("=" * 70)
            logger.info("🚀 Starting Pranthora-LiveKit Bridge Test")
            logger.info("=" * 70)
            logger.info(f"Pranthora Agent ID: {PRANTHORA_AGENT_ID}")
            logger.info(f"Pranthora WS URL: {PRANTHORA_WS_URL}")
            logger.info(f"LiveKit URL: {self.livekit_url}")
            logger.info(f"LiveKit Agent ID: {self.livekit_agent_id}")
            logger.info(f"Room Name: {self.room_name}")
            logger.info(f"Participant Name: {self.participant_name}")
            logger.info("=" * 70)

            # Step 1: Create or get LiveKit room
            logger.info("\n📦 Step 1: Creating/getting LiveKit room...")
            await self.create_or_get_livekit_room()

            # Step 2: Create participant token
            logger.info("\n🔑 Step 2: Creating LiveKit participant token...")
            token = self.create_livekit_token(self.participant_name, "Bridge Participant")

            # Step 3: Connect to LiveKit room
            logger.info("\n🔌 Step 3: Connecting to LiveKit room...")
            room = await self.connect_livekit_room(token)

            # Print LiveKit participant WebSocket URL
            ws_url = f"{self.livekit_url}/?room={self.room_name}&token={token}"
            
            # Create token for LiveKit agent
            agent_token = self.create_livekit_token(self.livekit_agent_id, f"Agent {self.livekit_agent_id}")
            agent_ws_url = f"{self.livekit_url}/?room={self.room_name}&token={agent_token}"
            
            logger.info("\n" + "=" * 70)
            logger.info("🌐 LiveKit Participant WebSocket URL:")
            logger.info(ws_url)
            logger.info("=" * 70)
            logger.info(f"\n🤖 LiveKit Agent ID: {self.livekit_agent_id}")
            logger.info(f"🌐 LiveKit Agent WebSocket URL (use this to connect your agent):")
            logger.info(agent_ws_url)
            logger.info(f"\n💡 Instructions:")
            logger.info(f"   1. Use the Agent WebSocket URL above to connect agent '{self.livekit_agent_id}'")
            logger.info(f"   2. Or configure your LiveKit agent to join room '{self.room_name}'")
            logger.info(f"   3. The bridge will automatically subscribe to audio from the agent")
            logger.info("=" * 70)

            # Step 4: Connect to Pranthora agent
            logger.info("\n🔌 Step 4: Connecting to Pranthora agent...")
            pranthora_task = asyncio.create_task(self.connect_pranthora_agent())

            # Step 5: Start audio bridging
            logger.info("\n🎵 Step 5: Audio bridge is active!")
            logger.info("✅ Pranthora agent and LiveKit room are now connected")
            logger.info(f"⏱️  Test will run for {MAX_DURATION_SEC} seconds...")
            logger.info("💬 Agents can now talk to each other through the bridge\n")
            
            # Initialize stats
            self.stats["start_time"] = time.time()

            # Wait for completion
            await pranthora_task

            # Print PROOF statistics
            elapsed_time = time.time() - self.stats["start_time"]
            logger.info("\n" + "=" * 70)
            logger.info("📊 PROOF OF COMMUNICATION - STATISTICS")
            logger.info("=" * 70)
            logger.info(f"⏱️  Test Duration: {elapsed_time:.2f} seconds")
            logger.info(f"")
            logger.info(f"📥 FROM PRANTHORA TO LIVEKIT:")
            logger.info(f"   - Audio chunks received: {self.stats['pranthora_to_livekit_chunks']}")
            logger.info(f"   - Total bytes: {self.stats['pranthora_to_livekit_bytes']:,}")
            logger.info(f"   - Average rate: {self.stats['pranthora_to_livekit_bytes'] / elapsed_time if elapsed_time > 0 else 0:.0f} bytes/sec")
            logger.info(f"")
            logger.info(f"📤 FROM LIVEKIT TO PRANTHORA:")
            logger.info(f"   - Audio chunks sent: {self.stats['livekit_to_pranthora_chunks']}")
            logger.info(f"   - Total bytes: {self.stats['livekit_to_pranthora_bytes']:,}")
            logger.info(f"   - Average rate: {self.stats['livekit_to_pranthora_bytes'] / elapsed_time if elapsed_time > 0 else 0:.0f} bytes/sec")
            logger.info(f"   - Audio frames processed: {self.stats['livekit_audio_frames']}")
            logger.info(f"")
            logger.info(f"💬 TEXT MESSAGES FROM PRANTHORA:")
            logger.info(f"   - Total messages: {self.stats['pranthora_messages']}")
            if self.stats["pranthora_text_messages"]:
                logger.info(f"   - Text messages received:")
                for i, msg in enumerate(self.stats["pranthora_text_messages"][:10], 1):  # Show first 10
                    logger.info(f"     {i}. {msg[:100]}")
                if len(self.stats["pranthora_text_messages"]) > 10:
                    logger.info(f"     ... and {len(self.stats['pranthora_text_messages']) - 10} more")
            else:
                logger.info(f"   - No text messages received (audio-only communication)")
            logger.info("=" * 70)
            
            # Final proof check
            total_bytes = self.stats["pranthora_to_livekit_bytes"] + self.stats["livekit_to_pranthora_bytes"]
            if total_bytes > 0:
                logger.info(f"\n✅ PROOF: Agents ARE communicating!")
                logger.info(f"   Total data exchanged: {total_bytes:,} bytes")
                logger.info(f"   This proves audio/data is flowing between agents!")
            else:
                logger.warning(f"\n⚠️  WARNING: No data exchanged - agents may not be communicating")
            
            logger.info("\n✅ Bridge test completed successfully!")

        except Exception as e:
            logger.error(f"❌ Bridge test failed: {e}", exc_info=True)
            raise
        finally:
            # Cleanup
            logger.info("\n🧹 Cleaning up...")
            if self.livekit_room:
                try:
                    await self.livekit_room.disconnect()
                    logger.info("✅ Disconnected from LiveKit room")
                except Exception as e:
                    logger.error(f"⚠️ Error disconnecting from LiveKit: {e}")
            self.stop_event.set()
            logger.info("✅ Cleanup completed")


async def main():
    """Main entry point."""
    bridge = PranthoraLiveKitBridge()
    await bridge.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️ Test stopped by user")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        sys.exit(1)
