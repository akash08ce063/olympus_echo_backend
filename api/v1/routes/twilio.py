"""Twilio-based conversation routes."""

import asyncio
import json
import os
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from twilio.twiml.voice_response import Connect, VoiceResponse

from call_bridges.base import BaseBridge
from models.api import StartCallRequest, StartCallResponse
from models.bridge import BridgeStatus
from recorders.twilio_recorder import TwilioConversationRecorder
from transports.twilio import TwilioTransport
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/twilio", tags=["Twilio Conversations"])

# Store active bridges and transports
active_bridges: dict[str, BaseBridge] = {}
twilio_transports: dict[str, TwilioTransport] = {}  # "session-agent" -> transport
twilio_call_sids: dict[str, dict] = {}  # session -> {"a": sid, "b": sid}

# Twilio client (lazy init)
_twilio_client = None


def _get_twilio_client():
    """Get or create Twilio client."""
    global _twilio_client
    if _twilio_client is None:
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        if account_sid and auth_token:
            from twilio.rest import Client

            _twilio_client = Client(account_sid, auth_token)
    return _twilio_client


def _create_twilio_transport(session_id: str, agent_type: str, phone_number: str):
    """Factory to create and store Twilio transports."""

    def factory(backend_url: str, agent_id: str) -> TwilioTransport:
        # For Twilio, agent_id is the phone number, backend_url is unused
        transport = TwilioTransport(agent_id=phone_number, phone_number=phone_number)
        twilio_transports[f"{session_id}-{agent_type}"] = transport
        return transport

    return factory


@router.post("/calls", response_model=StartCallResponse)
async def start_call(request: StartCallRequest):
    """Start a Twilio-based agent-to-agent call."""
    client = _get_twilio_client()
    twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
    webhook_base = os.getenv("TWILIO_WEBHOOK_BASE_URL")

    if not client or not twilio_number or not webhook_base:
        raise HTTPException(
            status_code=503,
            detail="Twilio not configured. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, TWILIO_WEBHOOK_BASE_URL, TWILIO_WEBSOCKET_BASE_URL",
        )

    session_id = request.session_id or f"twilio_{uuid4().hex[:8]}"

    if session_id in active_bridges:
        raise HTTPException(status_code=409, detail=f"Session '{session_id}' already exists")

    try:
        # Initiate calls to both agents (run blocking HTTP calls in thread pool)
        def _create_call_a():
            return client.calls.create(
                to=request.agent_a_number,
                from_=twilio_number,
                url=f"{webhook_base}/v1/twilio/twiml?session={session_id}&agent=a",
                status_callback=f"{webhook_base}/v1/twilio/status?session={session_id}&agent=a",
                status_callback_event=["initiated", "ringing", "answered", "completed", "no-answer", "busy", "failed"],
            )

        def _create_call_b():
            return client.calls.create(
                to=request.agent_b_number,
                from_=twilio_number,
                url=f"{webhook_base}/v1/twilio/twiml?session={session_id}&agent=b",
                status_callback=f"{webhook_base}/v1/twilio/status?session={session_id}&agent=b",
                status_callback_event=["initiated", "ringing", "answered", "completed", "no-answer", "busy", "failed"],
            )

        # Run both calls concurrently in thread pool
        call_a, call_b = await asyncio.gather(
            asyncio.to_thread(_create_call_a),
            asyncio.to_thread(_create_call_b),
        )

        # Validate that both calls have SIDs
        if not call_a.sid or not call_b.sid:
            raise HTTPException(status_code=500, detail="Twilio call creation failed: missing call SID")

        # Store call SIDs
        twilio_call_sids[session_id] = {"a": call_a.sid, "b": call_b.sid}

        # Create bridge with Twilio transports
        # Use phone numbers as agent IDs for identification
        bridge = BaseBridge(
            transport_factory=lambda url, aid: _create_twilio_transport(session_id, "a" if aid == request.agent_a_number else "b", aid)(url, aid),
            backend_url=request.agent_a_number,
            agent_a_id=request.agent_a_number,
            agent_b_id=request.agent_b_number,
            recording_path=request.recording_path,
            recorder_factory=lambda conv_id, path: TwilioConversationRecorder(conv_id, path),
            recording_enabled=request.recording_enabled,
            conversation_id=session_id,
            max_duration_seconds=request.max_duration_seconds,
        )

        # Store call SIDs in transports
        if t := twilio_transports.get(f"{session_id}-a"):
            t.call_sid = call_a.sid
        if t := twilio_transports.get(f"{session_id}-b"):
            t.call_sid = call_b.sid

        active_bridges[session_id] = bridge

        return StartCallResponse(
            success=True,
            session_id=session_id,
            call_sid_a=call_a.sid,
            call_sid_b=call_b.sid,
            message="Calls initiated. Waiting for agents to answer.",
        )

    except Exception as e:
        logger.error(f"Error initiating calls: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/twiml")
async def twiml_webhook(session: str, agent: str):
    """Return TwiML to connect call to media stream."""
    # Use WebSocket URL for media streams (wss:// or ws://)
    websocket_base = os.getenv("TWILIO_WEBSOCKET_BASE_URL")
    if not websocket_base:
        raise HTTPException(
            status_code=503,
            detail="Twilio not configured. Set TWILIO_WEBSOCKET_BASE_URL (must be wss:// or ws://)",
        )

    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=f"{websocket_base}/v1/twilio/stream?session={session}&agent={agent}")
    response.append(connect)

    return Response(content=str(response), media_type="application/xml")


@router.websocket("/stream")
async def media_stream(websocket: WebSocket):
    """Handle Twilio media stream WebSocket."""
    # Accept the websocket first (required by FastAPI)
    await websocket.accept()

    transport = None
    key = None
    session = None
    agent = None

    try:
        # Twilio Media Streams doesn't pass query params in WebSocket URL
        # We need to read messages until we get the "start" event which contains the call SID
        logger.info("Waiting for 'start' event from Twilio to get call SID")

        # Read messages until we get a "start" event
        # First message is usually "connected", second is "start"
        call_sid = None
        messages_received = []
        start_timeout = 5.0  # Total timeout for getting start event

        while call_sid is None:
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=start_timeout)
                messages_received.append(message)
                logger.info(f"Received message from Twilio: {message[:200]}")

                try:
                    data = json.loads(message)
                    event = data.get("event")

                    if event == "start":
                        # Extract call SID from start event
                        # Twilio can send it in different formats
                        call_sid = data.get("start", {}).get("callSid") or data.get("callSid")

                        if call_sid:
                            logger.info(f"Extracted call SID from 'start' event: {call_sid}")
                            break
                        else:
                            logger.warning(f"'start' event found but no callSid: {message[:200]}")
                    elif event == "connected":
                        logger.info("Received 'connected' event, waiting for 'start' event...")
                        # Continue reading - reduce timeout for next message
                        start_timeout = 2.0
                    else:
                        logger.warning(f"Unexpected event type '{event}' while waiting for 'start' event")
                        # Continue reading but reduce timeout
                        start_timeout = 2.0

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message as JSON: {e}, message: {message[:200]}")
                    # Continue reading in case next message is valid
                    start_timeout = 2.0

            except asyncio.TimeoutError:
                logger.error("Timeout waiting for 'start' event from Twilio")
                await websocket.close(code=1008, reason="Timeout waiting for start event")
                return

        if not call_sid:
            logger.error("Could not extract call SID from any message")
            await websocket.close(code=1008, reason="Missing call SID in start event")
            return

        # Look up session and agent from call SID
        for sess_id, sids in twilio_call_sids.items():
            if sids.get("a") == call_sid:
                session = sess_id
                agent = "a"
                break
            elif sids.get("b") == call_sid:
                session = sess_id
                agent = "b"
                break

        if not session or not agent:
            logger.error(f"Could not find session/agent for call SID: {call_sid}, available: {twilio_call_sids}")
            await websocket.close(code=1011, reason="Call SID not found in active sessions")
            return

        logger.info(f"Found session={session}, agent={agent} for call_sid={call_sid}")

        key = f"{session}-{agent}"
        logger.info(f"Looking up transport for key: {key}, available transports: {list(twilio_transports.keys())}")
        transport = twilio_transports.get(key)

        if not transport:
            logger.error(f"Transport not found for key: {key}")
            await websocket.close(code=1011, reason="Transport not found")
            return

        transport.set_websocket(websocket)
        transport._connected = True  # Mark as connected when WebSocket is established
        logger.info(
            f"WebSocket connected: {key}, transport._connected={transport._connected}, transport._websocket={transport._websocket is not None}"
        )

        # Process all messages we received (including "connected" and "start" events)
        for msg in messages_received:
            result = await transport._process_message(msg)
            if result is not None:
                transport.queue_message(result)

        # Check if both transports connected -> start bridge
        other = "b" if agent == "a" else "a"
        other_transport = twilio_transports.get(f"{session}-{other}")

        logger.info(
            f"Checking bridge start: current={agent}, other={other}, other_transport exists={other_transport is not None}, other_is_connected={other_transport.is_connected if other_transport else False}"
        )

        if other_transport and other_transport.is_connected:
            bridge = active_bridges.get(session)
            if bridge and bridge.status == BridgeStatus.IDLE:
                logger.info(f"Both agents connected, starting bridge: {session}")
                asyncio.create_task(_run_bridge(bridge))
        else:
            # Log if we're waiting for the other agent
            logger.info(
                f"Waiting for other agent to connect: session={session}, current={agent}, waiting_for={other}, other_transport_exists={other_transport is not None}"
            )

        # Continue reading messages
        while True:
            # Check if transport is still connected before reading
            if not transport.is_connected:
                logger.info(f"Transport disconnected, stopping message loop for {key}")
                break

            try:
                message = await websocket.receive_text()
            except Exception as e:
                # WebSocket closed or error reading
                logger.debug(f"Error reading from WebSocket for {key}: {e}")
                break

            result = await transport._process_message(message)
            if result is not None:
                transport.queue_message(result)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {key if key else 'unknown'}")
    except Exception as e:
        logger.error(f"WebSocket error {key if key else 'unknown'}: {e}")
    finally:
        if transport:
            await transport.disconnect()


async def _run_bridge(bridge: BaseBridge):
    """Start and run bridge until completion."""
    try:
        if await bridge.start():
            await asyncio.gather(*bridge._tasks, return_exceptions=True)
    finally:
        await bridge.stop()
        # Always terminate Twilio calls when bridge stops (timeout, error, or completion)
        # This ensures calls are hung up even if bridge stops due to max_duration
        session_id = bridge.conversation_id
        if session_id in active_bridges:
            # Terminate the actual Twilio phone calls (not just WebSockets)
            client = _get_twilio_client()
            if client and (sids := twilio_call_sids.get(session_id)):
                # Run blocking HTTP calls in thread pool
                async def _terminate_call(sid: str):
                    try:
                        await asyncio.to_thread(lambda: client.calls(sid).update(status="completed"))
                        logger.info(f"Terminated Twilio call {sid} for session {session_id}")
                    except Exception as e:
                        logger.error(f"Error terminating call {sid}: {e}")

                # Terminate all calls concurrently
                await asyncio.gather(*[_terminate_call(sid) for sid in sids.values()], return_exceptions=True)


@router.post("/status")
async def status_callback(request: Request, session: str, agent: str):
    """Handle Twilio call status updates."""
    form = await request.form()
    status = form.get("CallStatus")
    call_sid = form.get("CallSid")
    duration = form.get("CallDuration")  # Will be None until call completes

    logger.info(f"Call status: session={session}, agent={agent}, status={status}, call_sid={call_sid}, duration={duration}")

    # Log additional useful info
    if status in ["no-answer", "busy", "failed", "canceled"]:
        logger.warning(f"Call failed: session={session}, agent={agent}, status={status}")

    if status == "answered" or status == "in-progress":
        logger.info(f"Call answered: session={session}, agent={agent}, status={status}")
        # Mark transport as connected when answered (WebSocket may connect before or after)
        transport = twilio_transports.get(f"{session}-{agent}")
        if transport:
            transport._connected = True
            logger.info(
                f"Transport marked as connected: {session}-{agent}, _connected={transport._connected}, _websocket={transport._websocket is not None}, is_connected={transport.is_connected}"
            )

            # Don't start bridge from status callback - wait for WebSocket connections
            # The bridge will be started from the WebSocket handler when both WebSockets are connected
            # This ensures both transports have their WebSockets established before routing begins

    if status == "completed":
        bridge = active_bridges.get(session)
        if bridge and bridge.status == BridgeStatus.ACTIVE:
            asyncio.create_task(_stop_session(session))
        elif bridge and bridge.status == BridgeStatus.IDLE:
            # Call completed before bridge started - likely no-answer or failure
            logger.warning(f"Call completed before bridge started: session={session}, agent={agent}")

    return {"status": "ok"}


async def _stop_session(session_id: str):
    """Stop session and end calls."""
    client = _get_twilio_client()
    bridge = active_bridges.get(session_id)

    if bridge:
        await bridge.stop()

    # End Twilio calls (run blocking HTTP calls in thread pool)
    if client and (sids := twilio_call_sids.get(session_id)):

        async def _end_call(sid: str):
            try:
                await asyncio.to_thread(lambda: client.calls(sid).update(status="completed"))
            except Exception as e:
                logger.error(f"Error ending call {sid}: {e}")

        # End all calls concurrently
        await asyncio.gather(*[_end_call(sid) for sid in sids.values()], return_exceptions=True)

    # Cleanup
    twilio_call_sids.pop(session_id, None)
    twilio_transports.pop(f"{session_id}-a", None)
    twilio_transports.pop(f"{session_id}-b", None)


@router.post("/calls/{session_id}/stop")
async def stop_call(session_id: str):
    """Stop a Twilio call session."""
    if session_id not in active_bridges:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    await _stop_session(session_id)
    return {"success": True, "session_id": session_id, "message": "Session stopped"}


@router.get("/calls/{session_id}")
async def get_call_status(session_id: str):
    """Get call session status."""
    bridge = active_bridges.get(session_id)
    if not bridge:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    status = bridge.get_status()
    status["call_sids"] = twilio_call_sids.get(session_id, {})
    return status


@router.get("/calls")
async def list_calls():
    """List all Twilio call sessions."""
    return {
        "total": len(active_bridges),
        "sessions": [b.get_status() for b in active_bridges.values()],
    }
