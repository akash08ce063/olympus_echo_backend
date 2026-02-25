"""
Shared Vapi helpers: credentials only (env or DB). No api.vapi.ai calls.

Start calls with the real SDK: Vapi(api_key=...).start(assistant_id=...) as in
vapi_terminal_chat.py and vapi/connection_manager.py.
"""

import os
from typing import Tuple


async def get_vapi_credentials(
    target_agent_id: str = "09c928c6-5730-434b-9521-3654ca0094e3",
) -> Tuple[str, str]:
    """
    Get (assistant_id, api_key) from env (VAPI_ASSISTANT_ID, VAPI_API_KEY) or from DB.
    """
    api_key = os.environ.get("VAPI_API_KEY")
    assistant_id = os.environ.get("VAPI_ASSISTANT_ID")
    if api_key and assistant_id:
        return assistant_id, api_key

    aid = os.environ.get("TARGET_AGENT_ID", target_agent_id)
    from static_memory_cache import StaticMemoryCache
    from data_layer.supabase_client import get_supabase_client
    StaticMemoryCache.initialize()
    client = await get_supabase_client()
    rows = await client.select("target_agents", filters={"id": aid})
    if not rows:
        raise ValueError("Target agent not found. Set VAPI_API_KEY and VAPI_ASSISTANT_ID.")
    pc = rows[0].get("provider_config") or {}
    assistant_id = pc.get("assistant_id") or pc.get("assistantId")
    api_key = pc.get("api_key") or (os.environ.get(pc.get("api_key_env") or "")) if pc.get("api_key_env") else None
    if not assistant_id or not api_key:
        raise ValueError("Target agent missing provider_config.assistant_id or api_key (or set api_key_env to an env var name)")
    return assistant_id, api_key
