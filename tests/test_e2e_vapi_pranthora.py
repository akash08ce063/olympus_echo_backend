"""
E2E test: Vapi target agent + Pranthora user agent conversation.

Uses target_agent_id (Vapi) and user_agent_id (Pranthora) from the Product Recommendation
test suite. Runs one conversation by connecting both agents and bridging audio until timeout.

Prerequisites:
- configurations.json with database and pranthora.base_url (e.g. http://localhost:5050)
- Target agent 09c928c6-5730-434b-9521-3654ca0094e3 (Vapi) with provider_config.assistant_id, api_key
- User agent a47835fd-edc1-47e6-85e9-ea0183dd8d99 with pranthora_agent_id set
- Pranthora backend running at pranthora.base_url

Run from repo root:
  cd olympus_echo_backend && python -m tests.test_e2e_vapi_pranthora
"""

import asyncio
import sys
import os

# Add project root so we can import from services, models, etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Config must be loaded before any service that uses StaticMemoryCache
from static_memory_cache import StaticMemoryCache
StaticMemoryCache.initialize()

from uuid import UUID
from data_layer.supabase_client import get_supabase_client
from services.target_connection_factory import create_target_connection
from services.agent_connection_manager import AgentConnectionManager


# IDs from the Product Recommendation test suite
TARGET_AGENT_ID = "09c928c6-5730-434b-9521-3654ca0094e3"  # Vapi
USER_AGENT_ID = "a47835fd-edc1-47e6-85e9-ea0183dd8d99"    # Pranthora
CONVERSATION_TIMEOUT_SECONDS = 60


def _build_target_agent_from_row(row: dict):
    """Build a target-agent-like object from DB row (no Pydantic)."""
    class T:
        pass
    t = T()
    t.id = row.get("id")
    t.agent_type = row.get("agent_type") or "custom"
    t.websocket_url = row.get("websocket_url") or ""
    t.connection_metadata = row.get("connection_metadata")
    t.provider_config = row.get("provider_config") or {}
    return t


def _build_user_agent_from_row(row: dict):
    """Build a user-agent-like object from DB row."""
    class U:
        pass
    u = U()
    u.pranthora_agent_id = row.get("pranthora_agent_id")
    return u


async def run_e2e():
    print("Loading config and DB...")
    pranthora_base_url = StaticMemoryCache.get_pranthora_base_url()
    if not pranthora_base_url:
        raise SystemExit("pranthora.base_url not set in configurations.json")

    client = await get_supabase_client()
    target_rows = await client.select("target_agents", filters={"id": TARGET_AGENT_ID})
    user_rows = await client.select("user_agents", filters={"id": USER_AGENT_ID})

    if not target_rows:
        raise SystemExit(f"Target agent {TARGET_AGENT_ID} not found in DB")
    if not user_rows:
        raise SystemExit(f"User agent {USER_AGENT_ID} not found in DB")

    target_agent = _build_target_agent_from_row(target_rows[0])
    user_agent = _build_user_agent_from_row(user_rows[0])

    if (target_agent.provider_config or {}).get("assistant_id") is None:
        raise SystemExit("Target agent missing provider_config.assistant_id (Vapi)")
    if not user_agent.pranthora_agent_id:
        raise SystemExit("User agent missing pranthora_agent_id")

    print(f"Target (Vapi) assistant_id: {target_agent.provider_config.get('assistant_id')}")
    print(f"User (Pranthora) agent_id: {user_agent.pranthora_agent_id}")
    print(f"Pranthora base URL: {pranthora_base_url}")

    if pranthora_base_url.startswith("https://"):
        base_ws_url = pranthora_base_url.replace("https://", "wss://")
    else:
        base_ws_url = pranthora_base_url.replace("http://", "ws://")
    user_ws_url = f"{base_ws_url}/api/call/media-stream/agents/{user_agent.pranthora_agent_id}"
    call_sid_target = "e2e-test-target"
    call_sid_user = "e2e-test-user"
    user_ws_url = f"{user_ws_url}?call_sid={call_sid_user}"

    target_to_user_queue = asyncio.Queue()
    user_to_target_queue = asyncio.Queue()
    stop_event = asyncio.Event()
    target_ready = asyncio.Event()
    user_ready = asyncio.Event()

    def noop_callback(_):
        pass

    print("Creating target (Vapi) connection...")
    target_manager = await create_target_connection(
        target_agent,
        call_sid=call_sid_target,
        incoming_queue=user_to_target_queue,
        outgoing_queue=target_to_user_queue,
        stop_event=stop_event,
        my_ready=target_ready,
        other_ready=user_ready,
        record_sent_callback=noop_callback,
        sync_timeout=10.0,
        pranthora_base_url=pranthora_base_url,
    )
    print("Creating user (Pranthora) connection...")
    user_manager = AgentConnectionManager(
        name="User",
        ws_url=user_ws_url,
        call_sid=call_sid_user,
        incoming_queue=target_to_user_queue,
        outgoing_queue=user_to_target_queue,
        stop_event=stop_event,
        my_ready=user_ready,
        other_ready=target_ready,
        sync_timeout=10.0,
        record_sent_callback=noop_callback,
    )

    print("Starting both connections (conversation will run until timeout)...")
    target_task = asyncio.create_task(target_manager.connect())
    user_task = asyncio.create_task(user_manager.connect())

    try:
        done, pending = await asyncio.wait(
            [target_task, user_task],
            timeout=CONVERSATION_TIMEOUT_SECONDS,
            return_when=asyncio.FIRST_EXCEPTION,
        )
        for t in pending:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        stop_event.set()

        for t in done:
            exc = t.exception()
            if exc:
                print(f"Task failed: {exc}")
                raise exc
        print("Both connections finished without exception.")
    except asyncio.TimeoutError:
        print(f"Conversation ran for {CONVERSATION_TIMEOUT_SECONDS}s (timeout). Stopping.")
        stop_event.set()
        target_task.cancel()
        user_task.cancel()
        await asyncio.gather(target_task, user_task, return_exceptions=True)

    print("E2E test done: Vapi target + Pranthora user conversation ran successfully.")


if __name__ == "__main__":
    asyncio.run(run_e2e())
