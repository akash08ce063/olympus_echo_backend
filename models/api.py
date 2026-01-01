"""
API request and response models.

This module contains Pydantic models for API endpoints.
"""

from typing import Optional

from pydantic import BaseModel, Field


class StartConversationRequest(BaseModel):
    """Request model for starting a WebSocket-based conversation."""

    backend_ws_url: str = Field(..., description="WebSocket URL of the voice backend")
    agent_a_id: str = Field(..., description="ID of the first agent")
    agent_b_id: str = Field(..., description="ID of the second agent")
    conversation_id: Optional[str] = Field(None, description="Optional custom conversation ID")
    max_duration_seconds: Optional[int] = Field(None, gt=0)
    recording_enabled: bool = Field(True)
    recording_path: str = Field("recordings")


class ConversationResponse(BaseModel):
    """Response model for conversation operations."""

    success: bool
    conversation_id: str
    message: str
    status: dict


class StartCallRequest(BaseModel):
    """Request model for starting a Twilio-based call."""

    agent_a_number: str = Field(..., description="Phone number of agent A (E.164 format)")
    agent_b_number: str = Field(..., description="Phone number of agent B (E.164 format)")
    session_id: Optional[str] = Field(None)
    max_duration_seconds: Optional[int] = Field(None, gt=0)
    recording_enabled: bool = Field(True)
    recording_path: str = Field("recordings")


class StartCallResponse(BaseModel):
    """Response model for starting a Twilio call."""

    success: bool
    session_id: str
    call_sid_a: str
    call_sid_b: str
    message: str


class ScaledTestRequest(BaseModel):
    """Request model for starting a scaled WebSocket test."""

    target_agent_uri: str = Field(
        ..., description="WebSocket URL of the target agent"
    )
    user_agent_uri: str = Field(..., description="WebSocket URL of the user agent")
    concurrent_requests: int = Field(
        ..., ge=1, description="Number of parallel WebSocket connections"
    )
    sample_rate: int = Field(
        ...,
        ge=1000,
        le=48000,
        description="Sample rate of audio from target agent (Hz)",
    )
    encoding: str = Field(
        ...,
        pattern="^(mulaw|pcm16|pcm)$",
        description="Encoding of audio from target agent (mulaw, pcm16, pcm)",
    )
    timeout: int = Field(
        ..., ge=1, description="Timeout in seconds after which all connections close"
    )


class ScaledTestResponse(BaseModel):
    """Response model for scaled test operations."""

    success: bool
    test_id: str
    message: str
    status: dict


class WebScaledTestRequest(BaseModel):
    """Request model for starting a web scaled WebSocket test."""

    target_agent_uri: str = Field(
        ...,
        description="WebSocket URL or agent ID for target agent (media-stream endpoint)",
    )
    user_agent_id: str = Field(
        ..., description="Agent ID for user agent (web-media-stream endpoint)"
    )
    concurrent_requests: int = Field(
        ..., ge=1, description="Number of parallel WebSocket connections"
    )
    timeout: int = Field(
        ..., ge=1, description="Timeout in seconds after which all connections close"
    )
    ws_url_base: str = Field(
        default="ws://localhost:5050",
        description="Base WebSocket URL (default: ws://localhost:5050)",
    )


class WebScaledTestResponse(BaseModel):
    """Response model for web scaled test operations."""

    success: bool
    test_id: str
    message: str
    status: dict
