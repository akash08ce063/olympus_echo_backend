"""
Vapi integration for Olympus Echo.

- Terminal / human chat: use vapi_python SDK (web call via Daily.co) in tests/vapi_terminal_chat.py.
- Main stream (test execution): VapiConnectionManager uses real SDK (Vapi().start()) like vapi_terminal_chat.
"""

from vapi.client import get_vapi_credentials
from vapi.connection_manager import VapiConnectionManager

__all__ = ["VapiConnectionManager", "get_vapi_credentials"]
