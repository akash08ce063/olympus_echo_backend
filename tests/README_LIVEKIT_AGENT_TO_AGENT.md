# LiveKit Agent-to-Agent Conversation Test

This test script connects two LiveKit agents (Customer and Customer Support) in the same room and enables them to have a conversation.

## Overview

The script:
1. Creates a LiveKit room
2. Connects a **Customer** agent to the room
3. Connects a **Customer Support** agent to the room
4. Bridges audio between them so they can talk
5. Generates test audio tones to simulate a conversation
6. Provides detailed statistics as proof of communication

## Prerequisites

- LiveKit credentials configured in `config.json`:
  ```json
  {
    "livekit": {
      "url": "wss://your-livekit-server.livekit.cloud",
      "api_key": "your-api-key",
      "api_secret": "your-api-secret"
    }
  }
  ```

- Python dependencies installed:
  ```bash
  pip install livekit PyJWT numpy
  ```

## Usage

Run the test script:

```bash
cd olympus_echo_backend/tests
python test_livekit_agent_to_agent.py
```

## What It Does

1. **Creates a Room**: Generates a unique room name for the conversation
2. **Connects Customer Agent**: Creates a LiveKit participant with identity "customer-{uuid}"
3. **Connects Support Agent**: Creates a LiveKit participant with identity "support-agent-{uuid}"
4. **Publishes Audio Tracks**: Each agent publishes an audio track to the room
5. **Subscribes to Audio**: Each agent subscribes to the other's audio track
6. **Generates Test Audio**: Creates audio tones (440 Hz for customer, 550 Hz for support) to simulate speech
7. **Bridges Audio**: Audio flows bidirectionally between the two agents
8. **Logs Statistics**: Provides detailed proof of communication with byte counts and frame statistics

## Output

The script will:
- Print WebSocket URLs for both agents
- Log all connection events
- Show real-time statistics as audio flows
- Display final proof statistics at the end

## Proof of Communication

The script tracks:
- **Customer → Support**: Bytes and chunks sent from customer to support
- **Support → Customer**: Bytes and chunks sent from support to customer
- **Audio Frames**: Number of audio frames processed by each agent
- **Total Data**: Combined data exchanged between agents

## Duration

The test runs for **120 seconds** (2 minutes) by default. You can modify `MAX_DURATION_SEC` in the script to change this.

## Notes

- The script uses test audio tones to simulate conversation
- Both agents are created programmatically (no external agent configuration needed)
- The room is auto-created when the first participant joins
- All audio is logged for proof of bidirectional communication
