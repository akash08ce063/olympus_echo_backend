#!/usr/bin/env python3
"""
Terminal chat with a Vapi agent using the official Vapi Python SDK.

Uses vapi_python (https://github.com/VapiAI/client-sdk-python): starts a web call,
joins via Daily.co, and streams mic → agent and agent → speaker (16 kHz, SDK default).

Credentials from shared vapi package (env VAPI_API_KEY + VAPI_ASSISTANT_ID, or DB via TARGET_AGENT_ID).

  pip install vapi_python
  # On Mac for PyAudio: brew install portaudio

  cd olympus_echo_backend
  python -m tests.vapi_terminal_chat

  # With env (no DB):
  VAPI_API_KEY=... VAPI_ASSISTANT_ID=... python -m tests.vapi_terminal_chat

Press Enter or Ctrl+C to end.
"""

import asyncio
import os
import sys

# Project root so vapi package and config are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from vapi_python import Vapi
except ImportError:
    print("Install the Vapi Python SDK: pip install vapi_python")
    print("On Mac you may need: brew install portaudio")
    sys.exit(1)


async def main():

    assistant_id, api_key = "4badd12f-fa01-4021-9160-9f920292fc32", "295f2fae-2ce2-4857-a888-a19dc056e4c9"
 

    print(f"Using Vapi assistant_id: {assistant_id[:8]}...")
    print(f"Using Vapi api_key: {api_key[:8]}...")
    print("Starting call (Vapi SDK web call via Daily.co)...\n")

    vapi = Vapi(api_key=api_key)
    vapi.start(assistant_id=assistant_id)

    print("Speak into the mic. Press Enter to stop.")
    try:
        input()
    except KeyboardInterrupt:
        pass

    vapi.stop()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
