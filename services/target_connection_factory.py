"""
Factory to create target connection: Vapi (credentials from target_agent.provider_config) or custom WebSocket.
"""

import re
from typing import Any, Optional

from target_agent_type import TargetAgentType
from services.agent_connection_manager import AgentConnectionManager
from services.agent_url_resolver import resolve_target_agent_websocket_url
from vapi import VapiConnectionManager
from telemetrics.logger import logger


async def create_target_connection(
    target_agent: Any,
    call_sid: str,
    incoming_queue,
    outgoing_queue,
    stop_event,
    my_ready,
    other_ready,
    record_sent_callback,
    sync_timeout: float,
    pranthora_base_url: Optional[str] = None,
):
    agent_type = (getattr(target_agent, "agent_type", None) or "custom").lower()
    logger.info(f"[TargetConnectionFactory] agent_type={agent_type}")

    if agent_type == TargetAgentType.RETELL.value:
        raise ValueError("Retell target agent is not supported yet")

    if agent_type == TargetAgentType.VAPI.value:
        pc = getattr(target_agent, "provider_config", None) or {}
        assistant_id = pc.get("assistant_id") or ""
        api_key = pc.get("api_key") or ""
        if not assistant_id or not api_key:
            raise ValueError(
                "Target agent (vapi) missing provider_config.assistant_id or "
                "provider_config.api_key in database"
            )
        return VapiConnectionManager(
            name="Target",
            assistant_id=assistant_id.strip(),
            api_key=api_key.strip(),
            call_sid=call_sid,
            incoming_queue=incoming_queue,
            outgoing_queue=outgoing_queue,
            stop_event=stop_event,
            my_ready=my_ready,
            other_ready=other_ready,
            sync_timeout=sync_timeout,
            record_sent_callback=record_sent_callback,
        )

    # custom: resolve URL (ws/wss or HTTP -> ws), apply port replacement
    agent_url = getattr(target_agent, "websocket_url", None) or ""
    connection_metadata = getattr(target_agent, "connection_metadata", None)
    target_ws_url = await resolve_target_agent_websocket_url(agent_url, connection_metadata)
    if not target_ws_url.startswith(("ws://", "wss://")):
        raise ValueError(f"Resolved target URL is not WebSocket: {target_ws_url[:80]}")

    if pranthora_base_url:
        port_match = re.search(r":(\d+)", pranthora_base_url)
        if port_match:
            pranthora_port = port_match.group(1)
            target_ws_url = re.sub(r":\d+", f":{pranthora_port}", target_ws_url)

    if "call_sid=" not in target_ws_url:
        separator = "&" if "?" in target_ws_url else "?"
        target_ws_url = f"{target_ws_url}{separator}call_sid={call_sid}"

    return AgentConnectionManager(
        name="Target",
        ws_url=target_ws_url,
        call_sid=call_sid,
        incoming_queue=incoming_queue,
        outgoing_queue=outgoing_queue,
        stop_event=stop_event,
        my_ready=my_ready,
        other_ready=other_ready,
        sync_timeout=sync_timeout,
        record_sent_callback=record_sent_callback,
    )
