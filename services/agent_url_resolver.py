"""
Resolve target agent WebSocket URL from either a direct ws/wss URL or an HTTP(S) endpoint.

When the stored URL is HTTP(S), this module calls the endpoint with optional payload/headers
and extracts the WebSocket URL from the JSON response using a configurable dot-notation path.
"""

from typing import Any, Dict, Optional

import httpx

from telemetrics.logger import logger


def _get_nested_value(data: Dict[str, Any], path: str) -> Any:
    """Get value from nested dict using dot-notation path (e.g. 'data.websocket_url')."""
    if not path or not path.strip():
        return None
    keys = [k.strip() for k in path.split(".") if k.strip()]
    current: Any = data
    for key in keys:
        if current is None or not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


async def resolve_target_agent_websocket_url(
    agent_url: str,
    connection_metadata: Optional[Dict[str, Any]] = None,
    timeout_seconds: float = 15.0,
) -> str:
    """
    Resolve the WebSocket URL for a target agent.

    - If agent_url starts with ws:// or wss://, return it as-is (direct WebSocket).
    - If agent_url starts with http:// or https://, call the endpoint using connection_metadata
      (method, headers, payload) and extract the WebSocket URL from the response using
      response_websocket_url_path (dot-notation JSON path).

    Args:
        agent_url: Either a direct WebSocket URL (ws/wss) or an HTTP(S) endpoint URL.
        connection_metadata: Optional dict with keys: method, headers, payload, response_websocket_url_path.
        timeout_seconds: HTTP request timeout.

    Returns:
        The WebSocket URL to connect to.

    Raises:
        ValueError: If URL is invalid or HTTP response does not yield a valid WebSocket URL.
    """
    agent_url = (agent_url or "").strip()
    if not agent_url:
        raise ValueError("Target agent URL is empty")

    lower = agent_url.lower()
    if lower.startswith("ws://") or lower.startswith("wss://"):
        logger.info(f"[AgentUrlResolver] Direct WebSocket URL: {agent_url[:80]}{'...' if len(agent_url) > 80 else ''}")
        return agent_url

    if not lower.startswith("http://") and not lower.startswith("https://"):
        raise ValueError(f"Target agent URL must be ws/wss or http/https: {agent_url[:80]}")

    meta = connection_metadata or {}
    method = (meta.get("method") or "POST").upper()
    if method not in ("GET", "POST", "PUT", "PATCH"):
        method = "POST"
    headers = meta.get("headers") or {}
    payload = meta.get("payload") or meta.get("body")
    response_path = meta.get("response_websocket_url_path") or "websocket_url"

    logger.info(
        f"[AgentUrlResolver] HTTP endpoint: method={method}, url={agent_url[:80]}, "
        f"response_path={response_path}"
    )

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        if method == "GET":
            resp = await client.get(agent_url, headers=headers)
        elif method == "PUT":
            resp = await client.put(agent_url, json=payload, headers=headers)
        elif method == "PATCH":
            resp = await client.patch(agent_url, json=payload, headers=headers)
        else:
            resp = await client.post(agent_url, json=payload, headers=headers)

        resp.raise_for_status()
        try:
            data = resp.json()
        except Exception as e:
            logger.error(f"[AgentUrlResolver] Response is not JSON: {e}")
            raise ValueError(f"HTTP response is not JSON: {e}") from e

    ws_url = _get_nested_value(data, response_path)
    if ws_url is None:
        logger.error(
            f"[AgentUrlResolver] Path '{response_path}' not found or null in response. "
            f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}"
        )
        raise ValueError(
            f"WebSocket URL not found at path '{response_path}' in HTTP response"
        )

    ws_url = str(ws_url).strip()
    if not ws_url.lower().startswith("ws://") and not ws_url.lower().startswith("wss://"):
        logger.error(f"[AgentUrlResolver] Extracted value is not a WebSocket URL: {ws_url[:80]}")
        raise ValueError(f"Extracted value is not a WebSocket URL: {ws_url[:80]}")

    logger.info(f"[AgentUrlResolver] Resolved WebSocket URL from HTTP: {ws_url[:80]}{'...' if len(ws_url) > 80 else ''}")
    return ws_url
