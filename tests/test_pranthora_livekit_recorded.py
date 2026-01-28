#!/usr/bin/env python3
"""
Test script to connect Pranthora agent with LiveKit agent and record the conversation.

This test:
1. Connects Pranthora agent (Customer)
2. Connects LiveKit agent (Support)
3. Bridges audio between them
4. Records the entire conversation to a single WAV file

Prerequisites:
- LiveKit credentials configured in config.json
- Pranthora agent WebSocket URL
"""

import asyncio
import json
import logging
import sys
import uuid
import time
import wave
import numpy as np
from typing import Optional
from livekit import rtc
import jwt
import websockets
import base64
from datetime import datetime

# Import from parent directory
import sys as sys_module
import os
sys_module.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from static_memory_cache import StaticMemoryCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("PranthoraLiveKitRecorded")

# Audio settings
CHUNK_SIZE = 160  # 20ms @ 8kHz
SAMPLE_RATE = 8000
MAX_DURATION_SEC = 120  # 2 minutes for conversation

# Pranthora agent URL
PRANTHORA_AGENT_URL = "wss://51cd35808ed9.ngrok-free.app/api/call/web-media-stream?agent_id=e1143125-5a7f-4955-8f2c-01df1c604343"


class PranthoraLiveKitRecorder:
    """Bridges audio between Pranthora and LiveKit agents and records the conversation."""

    def __init__(self):
        self.livekit_url = StaticMemoryCache.get_livekit_url()
        self.livekit_api_key = StaticMemoryCache.get_livekit_api_key()
        self.livekit_api_secret = StaticMemoryCache.get_livekit_api_secret()
        
        # Room and participant names
        self.room_name = f"pranthora-livekit-room-{uuid.uuid4().hex[:8]}"
        self.livekit_agent_id = f"livekit-agent-{uuid.uuid4().hex[:8]}"
        
        # LiveKit room
        self.livekit_room: Optional[rtc.Room] = None
        
        # Audio sources and tracks
        self.livekit_audio_source: Optional[rtc.AudioSource] = None
        self.livekit_audio_track: Optional[rtc.LocalAudioTrack] = None
        
        # Queues for audio bridging
        self.pranthora_to_livekit_queue = asyncio.Queue()
        self.livekit_to_pranthora_queue = asyncio.Queue()
        
        # Audio recording buffers
        self.pranthora_audio_buffer = []  # List of audio bytes from Pranthora
        self.livekit_audio_buffer = []  # List of audio bytes from LiveKit
        self.recording_lock = asyncio.Lock()
        
        self.stop_event = asyncio.Event()
        self.pranthora_websocket = None
        
        # Statistics for proof of communication
        self.stats = {
            "pranthora_to_livekit_bytes": 0,
            "livekit_to_pranthora_bytes": 0,
            "pranthora_to_livekit_chunks": 0,
            "livekit_to_pranthora_chunks": 0,
            "pranthora_audio_frames": 0,
            "livekit_audio_frames": 0,
            "start_time": None,
        }
        
        # Output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = f"pranthora_livekit_conversation_{timestamp}.wav"

    def create_livekit_token(self, identity: str, name: str) -> str:
        """Create a token for LiveKit participant to join the room."""
        try:
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
            logger.info(f"✅ Created LiveKit token for: {identity} ({name})")
            return token_str

        except Exception as e:
            logger.error(f"❌ Error creating LiveKit token: {e}")
            raise

    async def connect_livekit_room(self, token: str) -> rtc.Room:
        """Connect LiveKit agent to room."""
        try:
            room = rtc.Room()

            @room.on("track_subscribed")
            def on_track_subscribed(
                track: rtc.Track,
                publication: rtc.TrackPublication,
                participant: rtc.RemoteParticipant,
            ):
                logger.info(f"📥 [LiveKit] Track subscribed: {track.kind} from {participant.identity} ({participant.name})")
                if track.kind == rtc.TrackKind.KIND_AUDIO:
                    logger.info(f"✅ PROOF: [LiveKit] Subscribed to audio from {participant.identity}")
                    asyncio.create_task(self._handle_livekit_incoming_audio(track))

            @room.on("track_published")
            def on_track_published(
                publication: rtc.RemoteTrackPublication,
                participant: rtc.RemoteParticipant,
            ):
                logger.info(f"📤 [LiveKit] Track published: {publication.kind} from {participant.identity}")
                if publication.kind == rtc.TrackKind.KIND_AUDIO:
                    try:
                        room.local_participant.set_subscribed(publication, True)
                        logger.info(f"✅ PROOF: [LiveKit] Subscribed to published track from {participant.identity}")
                        if publication.track:
                            asyncio.create_task(self._handle_livekit_incoming_audio(publication.track))
                    except Exception as e:
                        logger.warning(f"⚠️ [LiveKit] Could not subscribe: {e}")

            @room.on("participant_connected")
            def on_participant_connected(participant: rtc.RemoteParticipant):
                logger.info(f"👤 [LiveKit] Participant connected: {participant.identity} ({participant.name})")
                # Subscribe to all audio tracks
                for pub in participant.track_publications.values():
                    if pub.kind == rtc.TrackKind.KIND_AUDIO:
                        try:
                            room.local_participant.set_subscribed(pub, True)
                            if pub.track:
                                asyncio.create_task(self._handle_livekit_incoming_audio(pub.track))
                        except Exception as e:
                            logger.warning(f"⚠️ [LiveKit] Could not subscribe: {e}")

            # Connect to room
            await room.connect(self.livekit_url, token)
            logger.info(f"✅ [LiveKit] Connected to LiveKit room: {self.room_name}")

            # Create and publish audio track for LiveKit agent
            self.livekit_audio_source = rtc.AudioSource(SAMPLE_RATE, 1)  # 8kHz, mono
            self.livekit_audio_track = rtc.LocalAudioTrack.create_audio_track(
                "livekit-audio", self.livekit_audio_source
            )
            options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            await room.local_participant.publish_track(self.livekit_audio_track, options)
            logger.info("✅ [LiveKit] Published audio track to room")

            # Start task to send audio from LiveKit to Pranthora
            asyncio.create_task(self._send_livekit_audio_to_pranthora())

            # Subscribe to existing participants
            for participant in room.remote_participants.values():
                logger.info(f"🔍 [LiveKit] Found existing participant: {participant.identity}")
                for pub in participant.track_publications.values():
                    if pub.kind == rtc.TrackKind.KIND_AUDIO:
                        try:
                            room.local_participant.set_subscribed(pub, True)
                            if pub.track:
                                asyncio.create_task(self._handle_livekit_incoming_audio(pub.track))
                        except Exception as e:
                            logger.warning(f"⚠️ [LiveKit] Could not subscribe: {e}")

            self.livekit_room = room
            return room

        except Exception as e:
            logger.error(f"❌ [LiveKit] Error connecting to room: {e}")
            raise

    async def _handle_livekit_incoming_audio(self, track: rtc.Track):
        """Handle incoming audio from LiveKit room (from other participants)."""
        try:
            logger.info(f"🎵 PROOF: [LiveKit] Starting to receive audio from track")
            stream = rtc.AudioStream(track)
            frame_count = 0
            async for frame_event in stream:
                if self.stop_event.is_set():
                    break
                try:
                    # AudioStream yields AudioFrameEvent - access the frame property
                    frame = frame_event.frame if hasattr(frame_event, 'frame') else frame_event
                    
                    # Get audio data from frame
                    audio_data = None
                    
                    if hasattr(frame, 'data'):
                        frame_data = frame.data
                        if hasattr(frame_data, 'tobytes'):
                            audio_data = frame_data.tobytes()
                        elif isinstance(frame_data, np.ndarray):
                            audio_data = frame_data.tobytes()
                        elif hasattr(frame_data, '__iter__'):
                            # Convert array.array or similar to bytes
                            try:
                                import array
                                if isinstance(frame_data, array.array):
                                    audio_data = frame_data.tobytes()
                                else:
                                    audio_data = bytes(frame_data)
                            except:
                                audio_data = None
                        else:
                            try:
                                audio_data = bytes(frame_data)
                            except:
                                audio_data = None
                    
                    if audio_data is None or len(audio_data) == 0:
                        if frame_count == 0:
                            attrs = [a for a in dir(frame_event) if not a.startswith('_')]
                            logger.warning(f"⚠️ [LiveKit] Could not extract audio data. Frame type: {type(frame_event)}, Frame attrs: {attrs}")
                        continue
                    
                    # Ensure audio data is valid (even length for 16-bit samples)
                    if len(audio_data) % 2 != 0:
                        audio_data = audio_data[:len(audio_data)//2*2]
                    
                    if len(audio_data) == 0:
                        continue
                    
                    # Queue for sending to Pranthora
                    await self.livekit_to_pranthora_queue.put(audio_data)
                    
                    # Record audio to buffer
                    async with self.recording_lock:
                        self.livekit_audio_buffer.append(audio_data)
                    
                    # Update statistics
                    self.stats["livekit_to_pranthora_bytes"] += len(audio_data)
                    self.stats["livekit_to_pranthora_chunks"] += 1
                    self.stats["livekit_audio_frames"] += 1
                    frame_count += 1
                    
                    if frame_count == 1:
                        logger.info(f"✅ PROOF: [LiveKit] First audio frame received! ({len(audio_data)} bytes)")
                    
                    if self.stats["livekit_to_pranthora_chunks"] % 50 == 0:
                        logger.info(f"📊 PROOF: [LiveKit] Received {self.stats['livekit_to_pranthora_chunks']} chunks ({self.stats['livekit_to_pranthora_bytes']} bytes)")
                except Exception as frame_error:
                    if frame_count == 0:
                        logger.error(f"❌ [LiveKit] Error processing frame: {frame_error}", exc_info=True)
        except Exception as e:
            logger.error(f"❌ [LiveKit] Error handling audio: {e}", exc_info=True)

    async def _send_livekit_audio_to_pranthora(self):
        """Send audio from LiveKit queue to Pranthora agent."""
        try:
            while not self.stop_event.is_set():
                try:
                    # Get audio from queue - only send actual audio, no silence
                    audio_data = await asyncio.wait_for(
                        self.livekit_to_pranthora_queue.get(),
                        timeout=0.1
                    )

                    # Ensure audio data is valid and not empty
                    if not audio_data or len(audio_data) == 0:
                        continue

                    # Convert to proper format if needed (16-bit PCM)
                    if len(audio_data) % 2 != 0:
                        # Pad to even length for 16-bit samples
                        audio_data = audio_data + b'\x00'

                    # Send to Pranthora as base64 encoded JSON
                    if self.pranthora_websocket:
                        try:
                            payload_b64 = base64.b64encode(audio_data).decode()
                            message = {
                                "event_type": "audio",
                                "payload": payload_b64
                            }
                            await self.pranthora_websocket.send(json.dumps(message))
                            
                            # Update statistics
                            self.stats["livekit_to_pranthora_bytes"] += len(audio_data)
                            self.stats["livekit_to_pranthora_chunks"] += 1
                            
                            if self.stats["livekit_to_pranthora_chunks"] % 50 == 0:
                                logger.info(f"📊 PROOF: [LiveKit] Sent {self.stats['livekit_to_pranthora_chunks']} chunks ({self.stats['livekit_to_pranthora_bytes']} bytes) to Pranthora")
                        except Exception as send_err:
                            logger.warning(f"⚠️ [LiveKit] Error sending audio to Pranthora: {send_err}")

                except asyncio.TimeoutError:
                    # Don't send silence - just wait for actual audio
                    await asyncio.sleep(0.01)
                    continue

        except Exception as e:
            logger.error(f"❌ [LiveKit] Error sending audio to Pranthora: {e}", exc_info=True)

    async def _send_pranthora_audio_to_livekit(self):
        """Send audio from Pranthora queue to LiveKit room."""
        try:
            while not self.stop_event.is_set():
                try:
                    # Get audio from queue - only send actual audio
                    audio_data = await asyncio.wait_for(
                        self.pranthora_to_livekit_queue.get(),
                        timeout=0.1
                    )

                    # Ensure audio data is valid
                    if not audio_data or len(audio_data) == 0:
                        continue

                    # Send to LiveKit room
                    if self.livekit_audio_source:
                        import array
                        try:
                            # Ensure even length for 16-bit samples
                            if len(audio_data) % 2 != 0:
                                audio_data = audio_data[:len(audio_data)//2*2]
                            
                            # Convert bytes to int16 array
                            audio_array = np.frombuffer(audio_data, dtype=np.int16)
                            audio_array = array.array('h', audio_array.tolist())
                            
                            if len(audio_array) > 0:
                                frame = rtc.AudioFrame(
                                    data=audio_array,
                                    sample_rate=SAMPLE_RATE,
                                    num_channels=1,
                                    samples_per_channel=len(audio_array),
                                )
                                await self.livekit_audio_source.capture_frame(frame)
                                logger.debug("📤 [Pranthora] Sent audio to LiveKit")
                        except (ValueError, OverflowError) as conv_err:
                            logger.warning(f"⚠️ [Pranthora] Audio conversion error: {conv_err}, skipping chunk")
                            continue
                        except Exception as frame_err:
                            logger.warning(f"⚠️ [Pranthora] Error creating frame: {frame_err}")

                except asyncio.TimeoutError:
                    # Don't send silence - just wait for actual audio
                    await asyncio.sleep(0.01)
                    continue

        except Exception as e:
            logger.error(f"❌ [Pranthora] Error sending audio to LiveKit: {e}", exc_info=True)

    async def connect_pranthora_agent(self):
        """Connect to Pranthora agent WebSocket."""
        try:
            logger.info(f"🔌 Connecting to Pranthora agent: {PRANTHORA_AGENT_URL}")
            
            # Connect without Origin header (as requested)
            self.pranthora_websocket = await websockets.connect(
                PRANTHORA_AGENT_URL,
                ping_interval=20,
                ping_timeout=10,
            )
            
            logger.info("✅ Connected to Pranthora agent")
            
            # Send start event
            start_message = {
                "event_type": "start",
                "agent_id": "e1143125-5a7f-4955-8f2c-01df1c604343"
            }
            await self.pranthora_websocket.send(json.dumps(start_message))
            logger.info("✅ Sent start event to Pranthora")
            
            # Start reading from Pranthora
            asyncio.create_task(self._read_from_pranthora(self.pranthora_websocket))
            # Start sending to Pranthora
            asyncio.create_task(self._send_pranthora_audio_to_livekit())
            
        except Exception as e:
            logger.error(f"❌ Error connecting to Pranthora: {e}")
            raise

    async def _read_from_pranthora(self, websocket):
        """Read audio from Pranthora agent and forward to LiveKit."""
        try:
            async for raw_message in websocket:
                if self.stop_event.is_set():
                    break

                try:
                    # Handle binary data first
                    if isinstance(raw_message, bytes):
                        # Check if it's valid audio data (should be even length for 16-bit samples)
                        if len(raw_message) > 0 and len(raw_message) % 2 == 0:
                            # It's binary audio data
                            await self.pranthora_to_livekit_queue.put(raw_message)
                            
                            # Record audio to buffer
                            async with self.recording_lock:
                                self.pranthora_audio_buffer.append(raw_message)
                            
                            self.stats["pranthora_to_livekit_bytes"] += len(raw_message)
                            self.stats["pranthora_to_livekit_chunks"] += 1
                            
                            if self.stats["pranthora_to_livekit_chunks"] % 50 == 0:
                                logger.info(f"📊 PROOF: Received {self.stats['pranthora_to_livekit_chunks']} binary chunks ({self.stats['pranthora_to_livekit_bytes']} bytes) from Pranthora")
                            continue
                        else:
                            # Try to decode as UTF-8 text
                            try:
                                message = raw_message.decode('utf-8')
                            except UnicodeDecodeError:
                                # Invalid binary data, skip it
                                logger.debug(f"⚠️ [Pranthora] Skipping invalid binary data: {len(raw_message)} bytes")
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
                                    
                                    # Record audio to buffer
                                    async with self.recording_lock:
                                        self.pranthora_audio_buffer.append(audio_bytes)
                                    
                                    # Update statistics
                                    self.stats["pranthora_to_livekit_bytes"] += len(audio_bytes)
                                    self.stats["pranthora_to_livekit_chunks"] += 1
                                    self.stats["pranthora_audio_frames"] += 1
                                    
                                    if self.stats["pranthora_to_livekit_chunks"] % 50 == 0:
                                        logger.info(f"📊 PROOF: Received {self.stats['pranthora_to_livekit_chunks']} audio chunks ({self.stats['pranthora_to_livekit_bytes']} bytes) from Pranthora")
                                except Exception as decode_err:
                                    logger.debug(f"Could not decode audio payload: {decode_err}")

                        elif event_type == "text" or "text" in data or "transcript" in data or "message" in data:
                            text = data.get("text") or data.get("message") or data.get("transcript") or data.get("content")
                            if text:
                                logger.info(f"💬 PROOF - Pranthora said: {text}")
                                self.stats["pranthora_audio_frames"] += 1

                    except json.JSONDecodeError:
                        # Not JSON, might be plain text message
                        if message and message.strip():
                            logger.info(f"💬 PROOF - Pranthora message: {message[:100]}")

                except Exception as e:
                    logger.error(f"❌ Error processing message from Pranthora: {e}", exc_info=True)

        except websockets.exceptions.ConnectionClosed:
            logger.info("⚠️ Pranthora WebSocket connection closed")
        except Exception as e:
            logger.error(f"❌ Error reading from Pranthora: {e}", exc_info=True)

    def save_conversation_to_wav(self):
        """Save the recorded conversation to a WAV file."""
        try:
            logger.info(f"💾 Saving conversation to {self.output_file}...")
            
            # Combine all audio buffers
            async def combine_buffers():
                async with self.recording_lock:
                    pranthora_combined = b''.join(self.pranthora_audio_buffer)
                    livekit_combined = b''.join(self.livekit_audio_buffer)
                return pranthora_combined, livekit_combined
            
            # Run in sync context
            loop = asyncio.get_event_loop()
            pranthora_audio, livekit_audio = loop.run_until_complete(combine_buffers())
            
            # Convert bytes to numpy arrays
            pranthora_samples = np.frombuffer(pranthora_audio, dtype=np.int16).astype(np.float32) / 32768.0
            livekit_samples = np.frombuffer(livekit_audio, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Pad shorter array to match longer one
            max_len = max(len(pranthora_samples), len(livekit_samples))
            if len(pranthora_samples) < max_len:
                pranthora_samples = np.pad(pranthora_samples, (0, max_len - len(pranthora_samples)), mode='constant')
            if len(livekit_samples) < max_len:
                livekit_samples = np.pad(livekit_samples, (0, max_len - len(livekit_samples)), mode='constant')
            
            # Mix both audio streams (simple addition, then normalize)
            mixed_audio = pranthora_samples + livekit_samples
            
            # Normalize to prevent clipping
            peak = np.max(np.abs(mixed_audio))
            if peak > 1.0:
                mixed_audio = mixed_audio / peak
            
            # Convert back to int16
            mixed_audio_int16 = (mixed_audio * 32767).astype(np.int16)
            mixed_audio_bytes = mixed_audio_int16.tobytes()
            
            # Save to WAV file
            with wave.open(self.output_file, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(SAMPLE_RATE)  # 8kHz
                wav_file.writeframes(mixed_audio_bytes)
            
            logger.info(f"✅ Conversation saved to {self.output_file}")
            logger.info(f"   Pranthora audio: {len(pranthora_audio)} bytes ({len(pranthora_samples)} samples)")
            logger.info(f"   LiveKit audio: {len(livekit_audio)} bytes ({len(livekit_samples)} samples)")
            logger.info(f"   Mixed audio: {len(mixed_audio_bytes)} bytes ({len(mixed_audio_int16)} samples)")
            
        except Exception as e:
            logger.error(f"❌ Error saving conversation: {e}", exc_info=True)

    async def run(self):
        """Run the bridge test with recording."""
        try:
            logger.info("=" * 70)
            logger.info("🚀 Starting Pranthora-LiveKit Bridge Test with Recording")
            logger.info("=" * 70)
            logger.info(f"Pranthora Agent URL: {PRANTHORA_AGENT_URL}")
            logger.info(f"LiveKit URL: {self.livekit_url}")
            logger.info(f"Room Name: {self.room_name}")
            logger.info(f"LiveKit Agent ID: {self.livekit_agent_id}")
            logger.info(f"Output File: {self.output_file}")
            logger.info("=" * 70)

            # Step 1: Create LiveKit token
            logger.info("\n🔑 Step 1: Creating LiveKit token...")
            livekit_token = self.create_livekit_token(self.livekit_agent_id, "LiveKit Support Agent")

            # Step 2: Connect to LiveKit room
            logger.info("\n🔌 Step 2: Connecting to LiveKit room...")
            livekit_room = await self.connect_livekit_room(livekit_token)
            
            # Small delay
            await asyncio.sleep(1)

            # Step 3: Connect to Pranthora agent
            logger.info("\n🔌 Step 3: Connecting to Pranthora agent...")
            await self.connect_pranthora_agent()

            # Print WebSocket URLs
            livekit_ws_url = f"{self.livekit_url}/?room={self.room_name}&token={livekit_token}"
            
            logger.info("\n" + "=" * 70)
            logger.info("🌐 WebSocket URLs:")
            logger.info(f"LiveKit Agent: {livekit_ws_url}")
            logger.info("=" * 70)

            # Step 4: Start recording
            logger.info("\n🎤 Step 4: Recording conversation...")
            logger.info(f"⏱️  Test will run for {MAX_DURATION_SEC} seconds...")
            logger.info("💬 Agents are now connected and talking\n")
            self.stats["start_time"] = time.time()

            # Wait for completion
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=MAX_DURATION_SEC)
            except asyncio.TimeoutError:
                logger.info(f"⏹️ {MAX_DURATION_SEC} seconds reached, ending conversation")
            finally:
                self.stop_event.set()

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
            logger.info("=" * 70)
            
            # Save conversation to file
            logger.info("\n💾 Saving conversation to audio file...")
            self.save_conversation_to_wav()
            
            # Final proof check
            total_bytes = self.stats["pranthora_to_livekit_bytes"] + self.stats["livekit_to_pranthora_bytes"]
            if total_bytes > 0:
                logger.info(f"\n✅ PROOF: Agents ARE communicating!")
                logger.info(f"   Total data exchanged: {total_bytes:,} bytes")
                logger.info(f"   Conversation recorded to: {self.output_file}")
            else:
                logger.warning(f"\n⚠️  WARNING: No data exchanged - agents may not be communicating")
            
            logger.info("\n✅ Bridge test with recording completed successfully!")

        except Exception as e:
            logger.error(f"❌ Test failed: {e}", exc_info=True)
            raise
        finally:
            # Cleanup
            logger.info("\n🧹 Cleaning up...")
            self.stop_event.set()
            if self.pranthora_websocket:
                try:
                    await self.pranthora_websocket.close()
                    logger.info("✅ Disconnected from Pranthora")
                except Exception as e:
                    logger.error(f"⚠️ Error disconnecting from Pranthora: {e}")
            if self.livekit_room:
                try:
                    await self.livekit_room.disconnect()
                    logger.info("✅ Disconnected from LiveKit room")
                except Exception as e:
                    logger.error(f"⚠️ Error disconnecting from LiveKit: {e}")
            logger.info("✅ Cleanup completed")


async def main():
    """Main entry point."""
    bridge = PranthoraLiveKitRecorder()
    await bridge.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️ Test stopped by user")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        sys.exit(1)
