import asyncio
import websockets
import json
import pyaudio

WSS_URL = "wss://phone-call-websocket.aws-us-west-2-backend-production1.vapi.ai/019c1d04-15be-7778-a44c-ced4b650a725/listen"


# -------- Audio Setup --------
CHUNK = 1024
FORMAT = pyaudio.paInt16   # typical PCM 16bit
CHANNELS = 1
RATE = 16000               # common telephony sample rate

p = pyaudio.PyAudio()
audio_stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    output=True,
    frames_per_buffer=CHUNK,
)


async def connect_and_stream():
    print("🔌 Connecting to websocket...")

    async with websockets.connect(WSS_URL, ping_interval=None) as ws:
        print("✅ Connected. Listening in realtime...\n")

        try:
            async for message in ws:

                # -------- TEXT DATA --------
                if isinstance(message, str):
                    try:
                        parsed = json.loads(message)
                        print("📩 JSON:", json.dumps(parsed, indent=2))
                    except:
                        print("📩 TEXT:", message)

                # -------- AUDIO / BINARY DATA --------
                elif isinstance(message, bytes):
                    print(f"🔊 Audio chunk received ({len(message)} bytes)")
                    audio_stream.write(message)

        except websockets.exceptions.ConnectionClosed:
            print("❌ Connection closed")

        finally:
            audio_stream.stop_stream()
            audio_stream.close()
            p.terminate()


if __name__ == "__main__":
    asyncio.run(connect_and_stream())

