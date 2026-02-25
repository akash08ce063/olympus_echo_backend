"""
Target agent type constants. Stored as string in DB (no enum type in database).
"""

from enum import Enum


class TargetAgentType(str, Enum):
    """Type of target agent; determines which connection implementation to use."""

    CUSTOM = "custom"   # WebSocket or HTTP URL (current behavior)
    VAPI = "vapi"       # Vapi assistant by ID (Vapi SDK/API)
    RETELL = "retell"   # Reserved for future Retell integration
    PHONE = "phone"     # Direct phone target using connection_metadata.phone_number
