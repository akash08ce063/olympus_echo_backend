# Agent Bridge Service

A service that enables two voice agents to have real-time audio conversations by bridging their connections.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     Agent Bridge Service                       │
│                                                                │
│   ┌─────────────┐      ┌──────────┐      ┌─────────────┐       │
│   │  Transport  │◄────►│  Bridge  │◄────►│  Transport  │       │
│   │  (WS/Twilio)│      │  (Core)  │      │  (WS/Twilio)│       │
│   └─────────────┘      └────┬─────┘      └─────────────┘       │
│         ▲                   │                   ▲              │
│         │                   ▼                   │              │
│         │             ┌──────────┐              │              │
│         │             │ Recorder │              │              │
│         │             │  (Audio) │              │              │
│         │             └──────────┘              │              │
└─────────┼───────────────────────────────────────┼──────────────┘
          │                                       │
     ┌────▼────┐                             ┌────▼────┐
     │ Agent A │                             │ Agent B │
     └─────────┘                             └─────────┘
```

The bridge is **transport-agnostic**. It simply:
1. Receives audio from Transport A
2. Forwards it to Transport B
3. Receives audio from Transport B
4. Forwards it to Transport A

## Project Structure

```
agent_bridge/
├── api/
│   ├── app.py              # FastAPI app
│   └── routes/
│       ├── websocket.py    # WebSocket conversation routes
│       └── twilio.py       # Twilio conversation routes
├── transports/
│   ├── base.py             # AbstractTransport interface
│   ├── websocket.py        # WebSocket transport
│   └── twilio.py           # Twilio transport
├── bridge.py               # Core bridge logic (transport-agnostic)
├── audio_recorder.py       # Recording functionality
└── main.py                 # Entry point
```

## Installation

### 1. Set up virtual environment using uv

```bash
uv venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows
```

### 2. Install dependencies using uv

```bash
uv pip install -r requirements.txt
```

Alternatively, if you have a `pyproject.toml` file, you can use:

```bash
uv sync
```

## Running

### 1. Configure environment variables

Copy the example environment file and configure it with your values:

```bash
cp .env.example .env
```

Edit `.env` and set your Twilio credentials and ngrok URLs. **Important**:
- `TWILIO_WEBHOOK_BASE_URL` must use `https://` protocol
- `TWILIO_WEBSOCKET_BASE_URL` must use `wss://` protocol

### 2. Expose port 8000 using ngrok

The app requires a publicly accessible URL for webhooks (especially for Twilio). Run ngrok in a separate terminal:

```bash
ngrok http 8000
```

This will provide a public URL (e.g., `https://abc123.ngrok.io`). Update your `.env` file with the ngrok URLs:
- Use the `https://` URL for `TWILIO_WEBHOOK_BASE_URL`
- Use the `wss://` URL (same domain) for `TWILIO_WEBSOCKET_BASE_URL`

### 3. Run the application

```bash
uvicorn main:app --reload
```

The app will run on `http://localhost:8000` by default.

API docs: `http://localhost:8000/docs`

## API Endpoints

### WebSocket Conversations

```bash
# Start conversation
POST /conversations
{
  "backend_ws_url": "ws://localhost:8000",
  "agent_a_id": "agent_1",
  "agent_b_id": "agent_2",
  "recording_enabled": true
}

# Get status
GET /conversations/{id}

# Stop
POST /conversations/{id}/stop

# List all
GET /conversations
```

### Twilio Conversations

Requires environment variables (see `.env.example` for template):
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_PHONE_NUMBER`
- `TWILIO_WEBHOOK_BASE_URL` (must be publicly accessible, for HTTP webhooks - **must use `https://` protocol**, e.g., `https://abc123.ngrok.io`)
- `TWILIO_WEBSOCKET_BASE_URL` (must be publicly accessible, for WebSocket media streams - **must use `wss://` protocol**, e.g., `wss://abc123.ngrok.io`)

**Note**: Make sure to set these in your `.env` file as shown in `.env.example`. The webhook URL must use `https://` and the WebSocket URL must use `wss://`.

```bash
# Start call
POST /twilio/calls
{
  "agent_a_number": "+1234567890",
  "agent_b_number": "+1987654321",
  "recording_enabled": true
}

# Get status
GET /twilio/calls/{session_id}

# Stop
POST /twilio/calls/{session_id}/stop

# List all
GET /twilio/calls
```

### Health

```bash
GET /health
```

## How It Works

1. **API layer** creates the appropriate transports (WebSocket or Twilio)
2. **Transports** are injected into the **Bridge**
3. **Bridge** calls `transport.receive()` and `transport.send()` to route audio between agents
4. **Recorder** (if enabled) captures audio from both agents and saves it to WAV files
5. Bridge is completely unaware of transport implementation details

## Adding a New Transport

1. Create `transports/your_transport.py` implementing `AbstractTransport`
2. Create `api/routes/your_transport.py` with routes that:
   - Create your transport instances
   - Pass them to `AgentBridge`
3. Register the router in `api/app.py`

That's it. The bridge works with any transport that implements the interface.

## Requirements

- Python 3.10+
- For Twilio: `twilio` package and valid credentials
