# Pranthora-LiveKit Bridge Test

This test bridges a Pranthora agent with a LiveKit room, allowing them to communicate via audio.

## Quick Start

### 1. Install Dependencies
```bash
cd olympus_echo_backend
pip install -r requirements.txt
```

### 2. Configure LiveKit Webhooks (OPTIONAL!)
**⚠️ Webhooks are OPTIONAL!** The test works without them.

Webhooks are only needed if you want to receive event notifications to an external service. If you don't have a webhook endpoint URL, you can skip this step entirely.

If you want to set up webhooks:
- Go to LiveKit Cloud Dashboard → Settings → Webhooks
- Click "Create Webhook" or "Add Webhook"
- Enter a Name and your webhook endpoint URL
- Select a Signing API key from the dropdown
- Click "Create"
- See `SETUP_STEPS.md` for detailed instructions

### 3. Run the Test
```bash
cd tests
python test_pranthora_livekit_bridge.py
```

## What It Does

1. ✅ Creates/connects to a LiveKit room
2. ✅ Connects to Pranthora agent via WebSocket
3. ✅ Bridges audio between Pranthora and LiveKit
4. ✅ Prints the LiveKit WebSocket URL for connecting additional agents

## Output

The test will print:
- LiveKit room name
- LiveKit participant WebSocket URL (use this to connect Agent 2)
- Connection status for both agents
- Audio bridge status

## Configuration

Edit `test_pranthora_livekit_bridge.py` to change:
- `PRANTHORA_AGENT_ID`: The Pranthora agent ID
- `MAX_DURATION_SEC`: Test duration (default: 60 seconds)
- Audio settings (sample rate, chunk size, etc.)

## Connecting Agent 2

After running the test, you'll get a LiveKit WebSocket URL. Use this URL to:
- Connect a LiveKit agent using the LiveKit Agents framework
- Connect a client application
- Test with multiple participants

## Files

- `test_pranthora_livekit_bridge.py`: Main test script
- `SETUP_STEPS.md`: Detailed setup instructions
- `README_PRANTHORA_LIVEKIT_BRIDGE.md`: This file

## Troubleshooting

See `SETUP_STEPS.md` for detailed troubleshooting steps.
