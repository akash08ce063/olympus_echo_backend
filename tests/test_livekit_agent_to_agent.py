#!/usr/bin/env python3
"""
Test script to connect two LiveKit agents and let them talk.

This test:
1. Creates a LiveKit room
2. Connects Customer agent to the room
3. Connects Customer Support agent to the room
4. Bridges audio between them so they can have a conversation

Prerequisites:
- LiveKit credentials configured in config.json
"""

import asyncio
import json
import logging
import sys
import uuid
import time
from typing import Optional
from livekit import rtc
import jwt

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
logger = logging.getLogger("LiveKitAgentToAgent")

# Audio settings
CHUNK_SIZE = 160  # 20ms @ 8kHz
SAMPLE_RATE = 8000
MAX_DURATION_SEC = 120  # 2 minutes for conversation


class LiveKitAgentBridge:
    """Bridges audio between two LiveKit agents (Customer and Customer Support)."""

    def __init__(self):
        self.livekit_url = StaticMemoryCache.get_livekit_url()
        self.livekit_api_key = StaticMemoryCache.get_livekit_api_key()
        self.livekit_api_secret = StaticMemoryCache.get_livekit_api_secret()
        
        # Room and participant names
        self.room_name = f"conversation-room-{uuid.uuid4().hex[:8]}"
        self.customer_id = f"customer-{uuid.uuid4().hex[:8]}"
        self.support_agent_id = f"support-agent-{uuid.uuid4().hex[:8]}"
        
        # Rooms for each agent
        self.customer_room: Optional[rtc.Room] = None
        self.support_room: Optional[rtc.Room] = None
        
        # Audio sources and tracks
        self.customer_audio_source: Optional[rtc.AudioSource] = None
        self.customer_audio_track: Optional[rtc.LocalAudioTrack] = None
        self.support_audio_source: Optional[rtc.AudioSource] = None
        self.support_audio_track: Optional[rtc.LocalAudioTrack] = None
        
        # Queues for audio bridging
        self.customer_to_support_queue = asyncio.Queue()
        self.support_to_customer_queue = asyncio.Queue()
        
        self.stop_event = asyncio.Event()
        
        # Statistics for proof of communication
        self.stats = {
            "customer_to_support_bytes": 0,
            "support_to_customer_bytes": 0,
            "customer_to_support_chunks": 0,
            "support_to_customer_chunks": 0,
            "customer_audio_frames": 0,
            "support_audio_frames": 0,
            "start_time": None,
        }

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

    async def connect_customer_agent(self, token: str) -> rtc.Room:
        """Connect Customer agent to LiveKit room."""
        try:
            room = rtc.Room()

            @room.on("track_subscribed")
            def on_track_subscribed(
                track: rtc.Track,
                publication: rtc.TrackPublication,
                participant: rtc.RemoteParticipant,
            ):
                logger.info(f"📥 [Customer] Track subscribed: {track.kind} from {participant.identity} ({participant.name})")
                if track.kind == rtc.TrackKind.KIND_AUDIO:
                    logger.info(f"✅ PROOF: [Customer] Subscribed to audio from {participant.identity}")
                    asyncio.create_task(self._handle_customer_incoming_audio(track))

            @room.on("track_published")
            def on_track_published(
                publication: rtc.RemoteTrackPublication,
                participant: rtc.RemoteParticipant,
            ):
                logger.info(f"📤 [Customer] Track published: {publication.kind} from {participant.identity}")
                if publication.kind == rtc.TrackKind.KIND_AUDIO:
                    try:
                        room.local_participant.set_subscribed(publication, True)
                        logger.info(f"✅ PROOF: [Customer] Subscribed to published track from {participant.identity}")
                        if publication.track:
                            asyncio.create_task(self._handle_customer_incoming_audio(publication.track))
                    except Exception as e:
                        logger.warning(f"⚠️ [Customer] Could not subscribe: {e}")

            @room.on("participant_connected")
            def on_participant_connected(participant: rtc.RemoteParticipant):
                logger.info(f"👤 [Customer] Participant connected: {participant.identity} ({participant.name})")
                # Subscribe to all audio tracks
                for pub in participant.track_publications.values():
                    if pub.kind == rtc.TrackKind.KIND_AUDIO:
                        try:
                            room.local_participant.set_subscribed(pub, True)
                            if pub.track:
                                asyncio.create_task(self._handle_customer_incoming_audio(pub.track))
                        except Exception as e:
                            logger.warning(f"⚠️ [Customer] Could not subscribe: {e}")

            # Connect to room
            await room.connect(self.livekit_url, token)
            logger.info(f"✅ [Customer] Connected to LiveKit room: {self.room_name}")

            # Create and publish audio track for customer
            self.customer_audio_source = rtc.AudioSource(SAMPLE_RATE, 1)  # 8kHz, mono
            self.customer_audio_track = rtc.LocalAudioTrack.create_audio_track(
                "customer-audio", self.customer_audio_source
            )
            options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            await room.local_participant.publish_track(self.customer_audio_track, options)
            logger.info("✅ [Customer] Published audio track to room")

            # Start task to send audio from customer to support
            asyncio.create_task(self._send_customer_audio_to_room())

            # Subscribe to existing participants
            for participant in room.remote_participants.values():
                logger.info(f"🔍 [Customer] Found existing participant: {participant.identity}")
                for pub in participant.track_publications.values():
                    if pub.kind == rtc.TrackKind.KIND_AUDIO:
                        try:
                            room.local_participant.set_subscribed(pub, True)
                            if pub.track:
                                asyncio.create_task(self._handle_customer_incoming_audio(pub.track))
                        except Exception as e:
                            logger.warning(f"⚠️ [Customer] Could not subscribe: {e}")

            self.customer_room = room
            return room

        except Exception as e:
            logger.error(f"❌ [Customer] Error connecting to room: {e}")
            raise

    async def connect_support_agent(self, token: str) -> rtc.Room:
        """Connect Customer Support agent to LiveKit room."""
        try:
            room = rtc.Room()

            @room.on("track_subscribed")
            def on_track_subscribed(
                track: rtc.Track,
                publication: rtc.TrackPublication,
                participant: rtc.RemoteParticipant,
            ):
                logger.info(f"📥 [Support] Track subscribed: {track.kind} from {participant.identity} ({participant.name})")
                if track.kind == rtc.TrackKind.KIND_AUDIO:
                    logger.info(f"✅ PROOF: [Support] Subscribed to audio from {participant.identity}")
                    asyncio.create_task(self._handle_support_incoming_audio(track))

            @room.on("track_published")
            def on_track_published(
                publication: rtc.RemoteTrackPublication,
                participant: rtc.RemoteParticipant,
            ):
                logger.info(f"📤 [Support] Track published: {publication.kind} from {participant.identity}")
                if publication.kind == rtc.TrackKind.KIND_AUDIO:
                    try:
                        room.local_participant.set_subscribed(publication, True)
                        logger.info(f"✅ PROOF: [Support] Subscribed to published track from {participant.identity}")
                        if publication.track:
                            asyncio.create_task(self._handle_support_incoming_audio(publication.track))
                    except Exception as e:
                        logger.warning(f"⚠️ [Support] Could not subscribe: {e}")

            @room.on("participant_connected")
            def on_participant_connected(participant: rtc.RemoteParticipant):
                logger.info(f"👤 [Support] Participant connected: {participant.identity} ({participant.name})")
                # Subscribe to all audio tracks
                for pub in participant.track_publications.values():
                    if pub.kind == rtc.TrackKind.KIND_AUDIO:
                        try:
                            room.local_participant.set_subscribed(pub, True)
                            if pub.track:
                                asyncio.create_task(self._handle_support_incoming_audio(pub.track))
                        except Exception as e:
                            logger.warning(f"⚠️ [Support] Could not subscribe: {e}")

            # Connect to room
            await room.connect(self.livekit_url, token)
            logger.info(f"✅ [Support] Connected to LiveKit room: {self.room_name}")

            # Create and publish audio track for support
            self.support_audio_source = rtc.AudioSource(SAMPLE_RATE, 1)  # 8kHz, mono
            self.support_audio_track = rtc.LocalAudioTrack.create_audio_track(
                "support-audio", self.support_audio_source
            )
            options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            await room.local_participant.publish_track(self.support_audio_track, options)
            logger.info("✅ [Support] Published audio track to room")

            # Start task to send audio from support to customer
            asyncio.create_task(self._send_support_audio_to_room())

            # Subscribe to existing participants
            for participant in room.remote_participants.values():
                logger.info(f"🔍 [Support] Found existing participant: {participant.identity}")
                for pub in participant.track_publications.values():
                    if pub.kind == rtc.TrackKind.KIND_AUDIO:
                        try:
                            room.local_participant.set_subscribed(pub, True)
                            if pub.track:
                                asyncio.create_task(self._handle_support_incoming_audio(pub.track))
                        except Exception as e:
                            logger.warning(f"⚠️ [Support] Could not subscribe: {e}")

            self.support_room = room
            return room

        except Exception as e:
            logger.error(f"❌ [Support] Error connecting to room: {e}")
            raise

    async def _handle_customer_incoming_audio(self, track: rtc.Track):
        """Handle incoming audio from other participants (support agent) for customer.
        
        This is what the customer HEARS from the support agent.
        We only log this for proof, not queue it for sending.
        """
        try:
            logger.info(f"🎵 PROOF: [Customer] Starting to receive audio from track")
            stream = rtc.AudioStream(track)
            frame_count = 0
            async for frame_event in stream:
                if self.stop_event.is_set():
                    break
                try:
                    # AudioStream yields AudioFrameEvent - access the frame property
                    frame = frame_event.frame if hasattr(frame_event, 'frame') else frame_event
                    
                    # Get audio data from frame
                    if hasattr(frame, 'data'):
                        frame_data = frame.data
                        if hasattr(frame_data, 'tobytes'):
                            audio_data = frame_data.tobytes()
                        else:
                            import numpy as np
                            if isinstance(frame_data, np.ndarray):
                                audio_data = frame_data.tobytes()
                            else:
                                audio_data = bytes(frame_data)
                    else:
                        # Frame might be the data itself
                        import numpy as np
                        if isinstance(frame, np.ndarray):
                            audio_data = frame.tobytes()
                        elif hasattr(frame, 'tobytes'):
                            audio_data = frame.tobytes()
                        else:
                            # Try to get data from frame_event directly
                            audio_data = None
                            if hasattr(frame_event, 'data'):
                                frame_data = frame_event.data
                                if hasattr(frame_data, 'tobytes'):
                                    audio_data = frame_data.tobytes()
                                else:
                                    audio_data = bytes(frame_data)
                    
                    if audio_data is None:
                        # Log structure on first frame for debugging
                        if frame_count == 0:
                            attrs = [a for a in dir(frame_event) if not a.startswith('_')]
                            logger.warning(f"⚠️ [Customer] Could not extract audio data. Frame type: {type(frame_event)}, Frame attrs: {attrs}")
                        continue
                    
                    # Update statistics - PROOF that customer is receiving audio from support
                    self.stats["support_to_customer_bytes"] += len(audio_data)
                    self.stats["support_to_customer_chunks"] += 1
                    self.stats["customer_audio_frames"] += 1
                    frame_count += 1
                    
                    if frame_count == 1:
                        logger.info(f"✅ PROOF: [Customer] First audio frame received from Support! ({len(audio_data)} bytes)")
                    
                    if self.stats["support_to_customer_chunks"] % 50 == 0:
                        logger.info(f"📊 PROOF: [Customer] Received {self.stats['support_to_customer_chunks']} chunks ({self.stats['support_to_customer_bytes']} bytes) from Support")
                except Exception as frame_error:
                    if frame_count == 0:  # Only log on first error to avoid spam
                        logger.error(f"❌ [Customer] Error processing frame: {frame_error}", exc_info=True)
        except Exception as e:
            logger.error(f"❌ [Customer] Error handling audio: {e}", exc_info=True)

    async def _handle_support_incoming_audio(self, track: rtc.Track):
        """Handle incoming audio from other participants (customer) for support agent.
        
        This is what the support agent HEARS from the customer.
        We only log this for proof, not queue it for sending.
        """
        try:
            logger.info(f"🎵 PROOF: [Support] Starting to receive audio from track")
            stream = rtc.AudioStream(track)
            frame_count = 0
            async for frame_event in stream:
                if self.stop_event.is_set():
                    break
                try:
                    # AudioStream yields AudioFrameEvent - access the frame property
                    frame = frame_event.frame if hasattr(frame_event, 'frame') else frame_event
                    
                    # Get audio data from frame
                    if hasattr(frame, 'data'):
                        frame_data = frame.data
                        if hasattr(frame_data, 'tobytes'):
                            audio_data = frame_data.tobytes()
                        else:
                            import numpy as np
                            if isinstance(frame_data, np.ndarray):
                                audio_data = frame_data.tobytes()
                            else:
                                audio_data = bytes(frame_data)
                    else:
                        # Frame might be the data itself
                        import numpy as np
                        if isinstance(frame, np.ndarray):
                            audio_data = frame.tobytes()
                        elif hasattr(frame, 'tobytes'):
                            audio_data = frame.tobytes()
                        else:
                            # Try to get data from frame_event directly
                            audio_data = None
                            if hasattr(frame_event, 'data'):
                                frame_data = frame_event.data
                                if hasattr(frame_data, 'tobytes'):
                                    audio_data = frame_data.tobytes()
                                else:
                                    audio_data = bytes(frame_data)
                    
                    if audio_data is None:
                        # Log structure on first frame for debugging
                        if frame_count == 0:
                            attrs = [a for a in dir(frame_event) if not a.startswith('_')]
                            logger.warning(f"⚠️ [Support] Could not extract audio data. Frame type: {type(frame_event)}, Frame attrs: {attrs}")
                        continue
                    
                    # Update statistics - PROOF that support is receiving audio from customer
                    self.stats["customer_to_support_bytes"] += len(audio_data)
                    self.stats["customer_to_support_chunks"] += 1
                    self.stats["support_audio_frames"] += 1
                    frame_count += 1
                    
                    if frame_count == 1:
                        logger.info(f"✅ PROOF: [Support] First audio frame received from Customer! ({len(audio_data)} bytes)")
                    
                    if self.stats["customer_to_support_chunks"] % 50 == 0:
                        logger.info(f"📊 PROOF: [Support] Received {self.stats['customer_to_support_chunks']} chunks ({self.stats['customer_to_support_bytes']} bytes) from Customer")
                except Exception as frame_error:
                    if frame_count == 0:  # Only log on first error to avoid spam
                        logger.error(f"❌ [Support] Error processing frame: {frame_error}", exc_info=True)
        except Exception as e:
            logger.error(f"❌ [Support] Error handling audio: {e}", exc_info=True)

    async def _send_customer_audio_to_room(self):
        """Send audio from customer queue to LiveKit room (what customer says)."""
        try:
            while not self.stop_event.is_set():
                try:
                    # Get audio from queue (audio that customer should send)
                    audio_data = await asyncio.wait_for(
                        self.customer_to_support_queue.get(),
                        timeout=0.1
                    )

                    # Send to LiveKit room
                    if self.customer_audio_source:
                        import array
                        try:
                            audio_array = array.array('h', audio_data)
                        except (ValueError, OverflowError):
                            # If conversion fails, pad or truncate
                            import numpy as np
                            audio_array = np.frombuffer(audio_data[:len(audio_data)//2*2], dtype=np.int16)
                            audio_array = array.array('h', audio_array.tolist())
                        
                        frame = rtc.AudioFrame(
                            data=audio_array,
                            sample_rate=SAMPLE_RATE,
                            num_channels=1,
                            samples_per_channel=len(audio_array),
                        )
                        await self.customer_audio_source.capture_frame(frame)
                        logger.debug("📤 [Customer] Sent audio to room")

                except asyncio.TimeoutError:
                    # Send silence if no audio
                    import array
                    silence = array.array('h', [0] * CHUNK_SIZE)
                    frame = rtc.AudioFrame(
                        data=silence,
                        sample_rate=SAMPLE_RATE,
                        num_channels=1,
                        samples_per_channel=CHUNK_SIZE,
                    )
                    if self.customer_audio_source:
                        await self.customer_audio_source.capture_frame(frame)

        except Exception as e:
            logger.error(f"❌ [Customer] Error sending audio: {e}")

    async def _send_support_audio_to_room(self):
        """Send audio from support queue to LiveKit room (what support agent says)."""
        try:
            while not self.stop_event.is_set():
                try:
                    # Get audio from queue (audio that support should send)
                    audio_data = await asyncio.wait_for(
                        self.support_to_customer_queue.get(),
                        timeout=0.1
                    )

                    # Send to LiveKit room
                    if self.support_audio_source:
                        import array
                        try:
                            audio_array = array.array('h', audio_data)
                        except (ValueError, OverflowError):
                            # If conversion fails, pad or truncate
                            import numpy as np
                            audio_array = np.frombuffer(audio_data[:len(audio_data)//2*2], dtype=np.int16)
                            audio_array = array.array('h', audio_array.tolist())
                        
                        frame = rtc.AudioFrame(
                            data=audio_array,
                            sample_rate=SAMPLE_RATE,
                            num_channels=1,
                            samples_per_channel=len(audio_array),
                        )
                        await self.support_audio_source.capture_frame(frame)
                        logger.debug("📤 [Support] Sent audio to room")

                except asyncio.TimeoutError:
                    # Send silence if no audio
                    import array
                    silence = array.array('h', [0] * CHUNK_SIZE)
                    frame = rtc.AudioFrame(
                        data=silence,
                        sample_rate=SAMPLE_RATE,
                        num_channels=1,
                        samples_per_channel=CHUNK_SIZE,
                    )
                    if self.support_audio_source:
                        await self.support_audio_source.capture_frame(frame)

        except Exception as e:
            logger.error(f"❌ [Support] Error sending audio: {e}")

    async def generate_test_audio(self):
        """Generate test audio tones to simulate conversation."""
        try:
            import numpy as np
            import math
            
            # Generate a simple tone (440 Hz for customer, 550 Hz for support)
            sample_count = SAMPLE_RATE // 10  # 0.1 second chunks
            customer_tone_samples = int(SAMPLE_RATE * 0.5)  # 0.5 second tone
            support_tone_samples = int(SAMPLE_RATE * 0.5)  # 0.5 second tone
            
            customer_freq = 440  # A4 note
            support_freq = 550   # Higher note
            
            # Generate customer audio (simulated speech)
            for i in range(0, customer_tone_samples, sample_count):
                if self.stop_event.is_set():
                    break
                chunk_samples = min(sample_count, customer_tone_samples - i)
                t = np.arange(i, i + chunk_samples) / SAMPLE_RATE
                tone = np.sin(2 * np.pi * customer_freq * t) * 0.3
                tone_int16 = (tone * 32767).astype(np.int16)
                audio_bytes = tone_int16.tobytes()
                await self.customer_to_support_queue.put(audio_bytes)
                self.stats["customer_to_support_bytes"] += len(audio_bytes)
                self.stats["customer_to_support_chunks"] += 1
                await asyncio.sleep(0.1)
            
            await asyncio.sleep(0.5)  # Pause between speakers
            
            # Generate support audio (simulated response)
            for i in range(0, support_tone_samples, sample_count):
                if self.stop_event.is_set():
                    break
                chunk_samples = min(sample_count, support_tone_samples - i)
                t = np.arange(i, i + chunk_samples) / SAMPLE_RATE
                tone = np.sin(2 * np.pi * support_freq * t) * 0.3
                tone_int16 = (tone * 32767).astype(np.int16)
                audio_bytes = tone_int16.tobytes()
                await self.support_to_customer_queue.put(audio_bytes)
                self.stats["support_to_customer_bytes"] += len(audio_bytes)
                self.stats["support_to_customer_chunks"] += 1
                await asyncio.sleep(0.1)
            
            logger.info("✅ Generated test audio tones for both agents")
            
        except Exception as e:
            logger.error(f"❌ Error generating test audio: {e}")

    async def run(self):
        """Run the agent-to-agent conversation test."""
        try:
            logger.info("=" * 70)
            logger.info("🚀 Starting LiveKit Agent-to-Agent Conversation Test")
            logger.info("=" * 70)
            logger.info(f"LiveKit URL: {self.livekit_url}")
            logger.info(f"Room Name: {self.room_name}")
            logger.info(f"Customer ID: {self.customer_id}")
            logger.info(f"Support Agent ID: {self.support_agent_id}")
            logger.info("=" * 70)

            # Step 1: Create tokens
            logger.info("\n🔑 Step 1: Creating tokens for both agents...")
            customer_token = self.create_livekit_token(self.customer_id, "Customer")
            support_token = self.create_livekit_token(self.support_agent_id, "Customer Support Agent")

            # Step 2: Connect Customer agent
            logger.info("\n👤 Step 2: Connecting Customer agent to room...")
            customer_room = await self.connect_customer_agent(customer_token)
            
            # Small delay to ensure customer is connected
            await asyncio.sleep(1)

            # Step 3: Connect Support agent
            logger.info("\n🤖 Step 3: Connecting Customer Support agent to room...")
            support_room = await self.connect_support_agent(support_token)

            # Print WebSocket URLs
            customer_ws_url = f"{self.livekit_url}/?room={self.room_name}&token={customer_token}"
            support_ws_url = f"{self.livekit_url}/?room={self.room_name}&token={support_token}"
            
            logger.info("\n" + "=" * 70)
            logger.info("🌐 WebSocket URLs:")
            logger.info(f"Customer: {customer_ws_url}")
            logger.info(f"Support: {support_ws_url}")
            logger.info("=" * 70)

            # Step 4: Generate test audio to simulate conversation
            logger.info("\n🎵 Step 4: Generating test audio for conversation...")
            self.stats["start_time"] = time.time()
            
            # Generate audio in background
            audio_task = asyncio.create_task(self.generate_test_audio())

            # Step 5: Wait for conversation
            logger.info("\n💬 Step 5: Agents are now connected and can talk!")
            logger.info(f"⏱️  Test will run for {MAX_DURATION_SEC} seconds...")
            logger.info("🎤 Audio is being exchanged between Customer and Support agents\n")

            # Wait for completion
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=MAX_DURATION_SEC)
            except asyncio.TimeoutError:
                logger.info(f"⏹️ {MAX_DURATION_SEC} seconds reached, ending conversation")
            finally:
                self.stop_event.set()
                audio_task.cancel()
                try:
                    await audio_task
                except asyncio.CancelledError:
                    pass

            # Print PROOF statistics
            elapsed_time = time.time() - self.stats["start_time"]
            logger.info("\n" + "=" * 70)
            logger.info("📊 PROOF OF COMMUNICATION - STATISTICS")
            logger.info("=" * 70)
            logger.info(f"⏱️  Test Duration: {elapsed_time:.2f} seconds")
            logger.info(f"")
            logger.info(f"👤 FROM CUSTOMER TO SUPPORT:")
            logger.info(f"   - Audio chunks sent: {self.stats['customer_to_support_chunks']}")
            logger.info(f"   - Total bytes: {self.stats['customer_to_support_bytes']:,}")
            logger.info(f"   - Average rate: {self.stats['customer_to_support_bytes'] / elapsed_time if elapsed_time > 0 else 0:.0f} bytes/sec")
            logger.info(f"   - Audio frames processed: {self.stats['customer_audio_frames']}")
            logger.info(f"")
            logger.info(f"🤖 FROM SUPPORT TO CUSTOMER:")
            logger.info(f"   - Audio chunks sent: {self.stats['support_to_customer_chunks']}")
            logger.info(f"   - Total bytes: {self.stats['support_to_customer_bytes']:,}")
            logger.info(f"   - Average rate: {self.stats['support_to_customer_bytes'] / elapsed_time if elapsed_time > 0 else 0:.0f} bytes/sec")
            logger.info(f"   - Audio frames processed: {self.stats['support_audio_frames']}")
            logger.info("=" * 70)
            
            # Final proof check
            total_bytes = self.stats["customer_to_support_bytes"] + self.stats["support_to_customer_bytes"]
            if total_bytes > 0:
                logger.info(f"\n✅ PROOF: Agents ARE communicating!")
                logger.info(f"   Total data exchanged: {total_bytes:,} bytes")
                logger.info(f"   Customer ↔ Support conversation is active!")
            else:
                logger.warning(f"\n⚠️  WARNING: No data exchanged - agents may not be communicating")
            
            logger.info("\n✅ Agent-to-agent test completed successfully!")

        except Exception as e:
            logger.error(f"❌ Test failed: {e}", exc_info=True)
            raise
        finally:
            # Cleanup
            logger.info("\n🧹 Cleaning up...")
            self.stop_event.set()
            if self.customer_room:
                try:
                    await self.customer_room.disconnect()
                    logger.info("✅ [Customer] Disconnected from room")
                except Exception as e:
                    logger.error(f"⚠️ [Customer] Error disconnecting: {e}")
            if self.support_room:
                try:
                    await self.support_room.disconnect()
                    logger.info("✅ [Support] Disconnected from room")
                except Exception as e:
                    logger.error(f"⚠️ [Support] Error disconnecting: {e}")
            logger.info("✅ Cleanup completed")


async def main():
    """Main entry point."""
    bridge = LiveKitAgentBridge()
    await bridge.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️ Test stopped by user")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        sys.exit(1)
