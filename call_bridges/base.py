"""
Base bridge implementation for agent-to-agent communication.

This module contains the core bridge logic that can be inherited
by transport-specific bridge implementations (WebSocket, Twilio, etc.).
"""

import asyncio
from uuid import uuid4
from datetime import datetime
from typing import Callable, Optional

from models.bridge import AgentConnection, BridgeStatus, ConversationStats
from recorders.conversation_recorder import ConversationRecorder
from transports.base import AbstractTransport
from utils.logger import get_logger

logger = get_logger(__name__)


# Type alias for transport factory function
TransportFactory = Callable[[str, str], AbstractTransport]

# Type alias for recorder factory function
RecorderFactory = Callable[[str, str], ConversationRecorder]


class BaseBridge:
    """
    Base bridge implementation for connecting two voice agents.

    This class contains the core logic for routing audio between agents
    via transports. Transport-specific bridge implementations should
    inherit from this class and override methods as needed.
    """

    def __init__(
        self,
        transport_factory: TransportFactory,
        backend_url: str,
        agent_a_id: str,
        agent_b_id: str,
        recording_path: str,
        recorder_factory: RecorderFactory,
        recording_enabled: bool = True,
        conversation_id: Optional[str] = None,
        max_duration_seconds: Optional[int] = None,
    ):
        """
        Initialize the agent bridge.

        Args:
            transport_factory: Factory function that creates transports.
                               Signature: (backend_url, agent_id) -> AbstractTransport
            backend_url: Base URL of the voice backend.
            agent_a_id: ID of the first agent.
            agent_b_id: ID of the second agent.
            conversation_id: Optional unique ID for this conversation.
            max_duration_seconds: Optional maximum duration for the call in seconds.
            recording_enabled: Whether to record the conversation audio.
            recording_path: Directory to save recordings.
            recorder_factory: Factory function that creates recorders.
                              Signature: (conversation_id, recording_path) -> ConversationRecorder
        """
        self.backend_url = backend_url
        self.conversation_id = conversation_id or f"conv_{uuid4().hex[:8]}"
        self.max_duration_seconds = max_duration_seconds

        # Create agent connections with injected transports
        self.agent_a = AgentConnection(
            agent_id=agent_a_id,
            transport=transport_factory(backend_url, agent_a_id),
        )
        self.agent_b = AgentConnection(
            agent_id=agent_b_id,
            transport=transport_factory(backend_url, agent_b_id),
        )

        self.status = BridgeStatus.IDLE
        self.stats = ConversationStats()
        self._stop_event = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._timeout_task: Optional[asyncio.Task] = None

        # Audio recording
        self.recording_enabled = recording_enabled
        self.recording_path = recording_path
        self._recorder_factory = recorder_factory
        self._recorder: Optional[ConversationRecorder] = None

    async def _connect_agent(self, agent: AgentConnection) -> bool:
        """Connect to a single agent via its transport."""
        logger.info(f"[{self.conversation_id}] Connecting to agent {agent.agent_id}")
        success = await agent.transport.connect()
        if success:
            logger.info(f"[{self.conversation_id}] Successfully connected to agent {agent.agent_id}")
        else:
            logger.error(f"[{self.conversation_id}] Failed to connect to agent {agent.agent_id}")
        return success

    async def _disconnect_agent(self, agent: AgentConnection) -> None:
        """Disconnect from a single agent."""
        await agent.transport.disconnect()
        logger.info(f"[{self.conversation_id}] Disconnected from agent {agent.agent_id}")

    async def _route_audio(
        self,
        source: AgentConnection,
        destination: AgentConnection,
        source_name: str,
    ) -> None:
        """Route audio from source agent to destination agent."""
        logger.info(f"[{self.conversation_id}] Starting audio routing: {source_name}")

        # Forward any buffered message first
        if source.buffered_message is not None:
            logger.info(f"[{self.conversation_id}] {source_name} forwarding buffered message ({len(source.buffered_message)} bytes)")
            try:
                await destination.transport.send(source.buffered_message)
                await self._record_and_track_audio(source.buffered_message, source_name)
            except Exception as e:
                logger.error(f"[{self.conversation_id}] Error forwarding buffered message in {source_name}: {e}")
            finally:
                source.buffered_message = None

        while not self._stop_event.is_set():
            try:
                if not source.is_connected or not destination.is_connected:
                    logger.warning(f"[{self.conversation_id}] Transport disconnected in {source_name} routing")
                    break

                message = await source.transport.receive(timeout=1.0)

                if message is None:
                    continue

                if isinstance(message, bytes):
                    # Log at INFO so we can see if/when raw audio frames flow
                    logger.info(f"[{self.conversation_id}] {source_name} received {len(message)} bytes")
                    await destination.transport.send(message)
                    await self._record_and_track_audio(message, source_name)
                elif isinstance(message, str):
                    # Log at info so we can see what kind of text frames the backend sends
                    truncated = message[:200]
                    logger.info(f"[{self.conversation_id}] {source_name} received text frame (len={len(message)}): {truncated}")

            except Exception as e:
                logger.error(f"[{self.conversation_id}] Error in {source_name} routing: {e}")
                break

        logger.info(f"[{self.conversation_id}] Stopped audio routing: {source_name}")

    async def _record_and_track_audio(self, audio_data: bytes, source_name: str) -> None:
        """Record audio and update statistics."""
        if self._recorder and self._recorder.is_open:
            await self._recorder.write_audio(audio_data, source_name)

        # Track first audio timestamp
        if self.stats.first_audio_at is None:
            self.stats.first_audio_at = datetime.now()
            if self.stats.started_at:
                self.stats.time_to_first_audio = (self.stats.first_audio_at - self.stats.started_at).total_seconds()
            logger.info(f"[{self.conversation_id}] First audio received after {self.stats.time_to_first_audio:.3f}s")

        # Update stats
        if source_name == "A->B":
            self.stats.agent_a_bytes_sent += len(audio_data)
            self.stats.agent_a_messages += 1
        else:
            self.stats.agent_b_bytes_sent += len(audio_data)
            self.stats.agent_b_messages += 1

    async def _monitor_timeout(self) -> None:
        """Monitor the conversation duration and force stop if max_duration_seconds is exceeded."""
        if not self.max_duration_seconds:
            return

        try:
            await asyncio.sleep(self.max_duration_seconds)
            logger.warning(f"[{self.conversation_id}] Maximum duration of {self.max_duration_seconds}s exceeded. Force terminating.")
            await self.stop()
        except asyncio.CancelledError:
            pass

    async def start(self) -> bool:
        """
        Start the agent-to-agent conversation.

        Returns:
            True if both agents connected successfully and conversation started.
        """
        if self.status not in (BridgeStatus.IDLE, BridgeStatus.STOPPED, BridgeStatus.ERROR):
            logger.warning(f"[{self.conversation_id}] Cannot start - current status: {self.status}")
            return False

        self.status = BridgeStatus.CONNECTING
        self._stop_event.clear()
        self.stats = ConversationStats(started_at=datetime.now())

        try:
            # Connect to both agents
            results = await asyncio.gather(
                self._connect_agent(self.agent_a),
                self._connect_agent(self.agent_b),
                return_exceptions=True,
            )

            if not all(r is True for r in results):
                logger.error(f"[{self.conversation_id}] Failed to connect to one or both agents")
                await self.stop()
                return False

            # Initialize audio recorder if enabled
            if self.recording_enabled:
                self._recorder = self._recorder_factory(
                    self.conversation_id,
                    self.recording_path,
                )
                if not await self._recorder.open():
                    logger.error(f"[{self.conversation_id}] Failed to start audio recording")
                    self._recorder = None

            # Start bidirectional audio routing
            self._tasks = [
                asyncio.create_task(self._route_audio(self.agent_a, self.agent_b, "A->B")),
                asyncio.create_task(self._route_audio(self.agent_b, self.agent_a, "B->A")),
            ]

            # Start timeout monitoring if max_duration is set
            if self.max_duration_seconds:
                self._timeout_task = asyncio.create_task(self._monitor_timeout())
                logger.info(f"[{self.conversation_id}] Maximum duration set to {self.max_duration_seconds}s")

            self.status = BridgeStatus.ACTIVE
            logger.info(f"[{self.conversation_id}] Bridge active - agents are now conversing")
            return True

        except Exception as e:
            logger.error(f"[{self.conversation_id}] Error starting bridge: {e}")
            self.status = BridgeStatus.ERROR
            await self.stop()
            return False

    async def stop(self) -> None:
        """Stop the agent-to-agent conversation and clean up resources."""
        if self.status == BridgeStatus.STOPPED:
            return

        logger.info(f"[{self.conversation_id}] Stopping bridge...")
        self.status = BridgeStatus.STOPPING
        self._stop_event.set()

        # Cancel timeout monitoring task
        if self._timeout_task:
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass
            self._timeout_task = None

        # Cancel routing tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()

        # Disconnect both agents
        await asyncio.gather(
            self._disconnect_agent(self.agent_a),
            self._disconnect_agent(self.agent_b),
        )

        # Close audio recorder
        if self._recorder is not None:
            await self._recorder.close()
            self._recorder = None

        self.stats.ended_at = datetime.now()
        self.status = BridgeStatus.STOPPED
        logger.info(f"[{self.conversation_id}] Bridge stopped")

    async def run_until_complete(self) -> None:
        """Run the bridge until it's stopped or a connection closes."""
        if not await self.start():
            return

        try:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        finally:
            await self.stop()

    def get_status(self) -> dict:
        """Get the current status of the bridge."""
        status_dict = {
            "conversation_id": self.conversation_id,
            "status": self.status.value,
            "agent_a": {
                "id": self.agent_a.agent_id,
                "connected": self.agent_a.is_connected,
                "bytes_sent": self.stats.agent_a_bytes_sent,
                "messages": self.stats.agent_a_messages,
            },
            "agent_b": {
                "id": self.agent_b.agent_id,
                "connected": self.agent_b.is_connected,
                "bytes_sent": self.stats.agent_b_bytes_sent,
                "messages": self.stats.agent_b_messages,
            },
            "duration_seconds": self.stats.duration_seconds,
            "total_bytes": self.stats.total_bytes,
            "time_to_first_audio": self.stats.time_to_first_audio,
            "first_audio_at": self.stats.first_audio_at.isoformat() if self.stats.first_audio_at else None,
            "started_at": self.stats.started_at.isoformat() if self.stats.started_at else None,
            "ended_at": self.stats.ended_at.isoformat() if self.stats.ended_at else None,
        }

        if self.recording_enabled:
            status_dict["recording"] = {
                "enabled": True,
                "file_path": str(self._recorder.filepath_merged) if self._recorder else None,
                "is_recording": self._recorder.is_open if self._recorder else False,
            }
        else:
            status_dict["recording"] = {"enabled": False}

        return status_dict
