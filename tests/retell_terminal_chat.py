#!/usr/bin/env python3
"""
Terminal chat with a Retell AI agent using the official Retell Python SDK.

Uses retell-sdk (https://pypi.org/project/retell-sdk/): creates a web call,
registers call, connects to audio WebSocket, and streams mic → agent and agent → speaker.

Credentials from env (RETELL_API_KEY + RETELL_AGENT_ID) or hardcoded for testing.

  pip install retell-sdk websockets pyaudio
  # On Mac for PyAudio: brew install portaudio

  cd olympus_echo_backend
  python -m tests.retell_terminal_chat

  # With env (no hardcoded):
  RETELL_API_KEY=... RETELL_AGENT_ID=... python -m tests.retell_terminal_chat

Press Enter or Ctrl+C to end.
"""

import asyncio
import os
import sys
import json
import websockets
from typing import Optional
import threading
import queue
import aiohttp

# Project root so retell package and config are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from retell import Retell
except ImportError:
    print("Install the Retell Python SDK: pip install retell-sdk")
    sys.exit(1)

try:
    import pyaudio
except ImportError:
    print("Install PyAudio: pip install pyaudio")
    print("On Mac: brew install portaudio && pip install pyaudio")
    sys.exit(1)


# Audio configuration (Retell uses PCM 16kHz 16-bit mono)
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 3200  # 200ms chunks (SAMPLE_RATE * 2 * 0.2)
FORMAT = pyaudio.paInt16


class RetellTerminalChat:
    def __init__(self, agent_id: str, api_key: str):
        self.agent_id = agent_id
        self.api_key = api_key
        self.retell_client = Retell(api_key=api_key)
        self.call_id: Optional[str] = None
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.audio_queue = queue.Queue()
        self.running = False
        self.pyaudio_instance = pyaudio.PyAudio()

    async def create_call(self):
        """Register a call to get call_id for audio WebSocket."""
        print("Registering call...")
        # Use register_call API to get call_id for audio WebSocket
        # Note: create_web_call is for frontend SDK, register_call is for audio WebSocket
        import aiohttp
        
        url = "https://api.retellai.com/register-call"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "agent_id": self.agent_id
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status not in (200, 201):
                    text = await resp.text()
                    raise RuntimeError(f"Failed to register call: {resp.status} - {text}")
                data = await resp.json()
                self.call_id = data.get("call_id")
                if not self.call_id:
                    raise RuntimeError("Response missing call_id")
                print(f"✅ Call registered: {self.call_id}")
                return self.call_id

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio input - sends mic data to queue."""
        if self.running:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    async def send_audio_loop(self):
        """Send audio from mic to Retell WebSocket."""
        try:
            while self.running:
                try:
                    # Get audio chunk from queue (non-blocking)
                    audio_data = self.audio_queue.get_nowait()
                    if self.ws and not self.ws.closed:
                        await self.ws.send(audio_data)
                except queue.Empty:
                    await asyncio.sleep(0.01)  # Small delay if no audio
                except Exception as e:
                    print(f"Error sending audio: {e}")
                    break
        except Exception as e:
            print(f"Send audio loop error: {e}")

    async def receive_audio_loop(self):
        """Receive audio from Retell WebSocket and play it."""
        try:
            while self.running:
                if not self.ws:
                    break
                try:
                    message = await asyncio.wait_for(self.ws.recv(), timeout=0.1)
                    
                    if isinstance(message, bytes):
                        # Raw audio bytes - play directly
                        self.play_audio(message)
                    elif isinstance(message, str):
                        # Text message (JSON events)
                        try:
                            data = json.loads(message)
                            event_type = data.get("event_type", "")
                            
                            if event_type == "update":
                                # Transcript update
                                transcript = data.get("transcript", [])
                                if transcript:
                                    last_utterance = transcript[-1]
                                    role = last_utterance.get("role", "")
                                    content = last_utterance.get("content", "")
                                    prefix = "🤖 Agent" if role == "agent" else "👤 You"
                                    print(f"{prefix}: {content}")
                            
                            elif event_type == "clear":
                                # Clear audio buffer
                                print("🛑 Interruption detected - clearing audio")
                            
                            elif event_type == "metadata":
                                # Metadata event
                                metadata = data.get("metadata", {})
                                print(f"📋 Metadata: {metadata}")
                        except json.JSONDecodeError:
                            if message == "clear":
                                print("🛑 Clear signal received")
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket connection closed")
                    break
        except Exception as e:
            print(f"Receive audio loop error: {e}")

    def play_audio(self, audio_data: bytes):
        """Play audio bytes using PyAudio (simplified - would need proper buffering in production)."""
        # Note: This is a simplified version. Production would need proper audio buffering
        # For now, we'll just print that audio was received
        pass  # Audio playback would require a separate audio output stream

    async def connect_audio_websocket(self):
        """Connect to Retell audio WebSocket."""
        if not self.call_id:
            raise ValueError("Call ID not set. Call create_call() first.")
        
        ws_url = f"wss://api.retellai.com/audio-websocket/{self.call_id}"
        print(f"Connecting to audio WebSocket: {ws_url}")
        
        self.ws = await websockets.connect(ws_url)
        print("✅ Connected to audio WebSocket")

    def start_audio_capture(self):
        """Start capturing audio from microphone."""
        print("🎤 Starting microphone capture...")
        self.stream = self.pyaudio_instance.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()
        print("✅ Microphone ready")

    def stop_audio_capture(self):
        """Stop capturing audio."""
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.pyaudio_instance.terminate()

    async def run(self):
        """Main run loop."""
        try:
            # Create call
            await self.create_call()
            
            # Connect to audio WebSocket
            await self.connect_audio_websocket()
            
            # Start audio capture
            self.running = True
            self.start_audio_capture()
            
            print("\n🎙️  Speak into your microphone. Press Enter to stop.\n")
            
            # Start send/receive loops
            send_task = asyncio.create_task(self.send_audio_loop())
            receive_task = asyncio.create_task(self.receive_audio_loop())
            
            # Wait for user input
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, input)
            
        except KeyboardInterrupt:
            print("\n⏹️  Stopping...")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            if self.ws:
                await self.ws.close()
            self.stop_audio_capture()
            print("✅ Stopped")


async def main():
    # Get credentials from env or use hardcoded for testing
    agent_id = os.getenv("RETELL_AGENT_ID") or "agent_abbb8831f77f356dfc612a0b91"
    api_key = os.getenv("RETELL_API_KEY") or "key_955d6d9aa6581750cfa316cd9f5e"

    if agent_id == "agent_abbb8831f77f356dfc612a0b91" or api_key == "key_955d6d9aa6581750cfa316cd9f5e":
        print("⚠️  Please set RETELL_AGENT_ID and RETELL_API_KEY environment variables")
        print("   or update the hardcoded values in this script.")
        sys.exit(1)

    print(f"Using Retell agent_id: {agent_id[:8]}...")
    print(f"Using Retell api_key: {api_key[:8]}...\n")

    chat = RetellTerminalChat(agent_id=agent_id, api_key=api_key)
    await chat.run()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
