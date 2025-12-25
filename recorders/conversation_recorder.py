"""
Audio recorder module for capturing agent-to-agent conversations.
"""

import asyncio
import time
import wave
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class ConversationRecorder:
    """
    Records audio from an agent-to-agent conversation to a WAV file.

    This class handles thread-safe recording of audio chunks from multiple
    concurrent sources (Agent A and Agent B) into a single mixed audio file.
    """

    def __init__(
        self,
        conversation_id: str,
        recording_path: str = "recordings",
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,  # 16-bit = 2 bytes
    ):
        """
        Initialize the conversation recorder.

        Args:
            conversation_id: Unique identifier for the conversation
            recording_path: Directory to save recordings (default: "recordings/")
            sample_rate: Audio sample rate in Hz (default: 16000)
            channels: Number of audio channels (default: 1 for mono)
            sample_width: Sample width in bytes (default: 2 for 16-bit)
        """
        self.conversation_id = conversation_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width

        # Set up recording path
        self.recording_dir = Path(recording_path)
        # Note: Directory creation will be done async in open() to avoid blocking

        # Generate filenames with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath_agent_a = self.recording_dir / f"{conversation_id}_{timestamp}_agent_a.wav"
        self.filepath_agent_b = self.recording_dir / f"{conversation_id}_{timestamp}_agent_b.wav"
        self.filepath_merged = self.recording_dir / f"{conversation_id}_{timestamp}.wav"
        # Keep filepath for backward compatibility (points to merged file)
        self.filepath = self.filepath_merged

        # Thread safety for concurrent writes
        self._lock = Lock()
        self._wav_file: Optional[wave.Wave_write] = None
        self._is_open = False

        # Buffer queue for audio chunks: (timestamp, audio_data, source)
        self._audio_buffer: deque = deque()
        self._buffer_lock = asyncio.Lock()
        self._buffer_size_limit = 8192  # 8KB default
        self._current_buffer_size = 0
        self._write_task: Optional[asyncio.Task] = None
        self._stop_writing = asyncio.Event()

        logger.info(f"[{conversation_id}] Recorder initialized: {self.filepath_merged}")

    async def open(self) -> bool:
        """
        Open the WAV file for writing.

        Returns:
            True if file opened successfully, False otherwise
        """
        try:
            # Create directory if it doesn't exist (non-blocking)
            await asyncio.to_thread(self.recording_dir.mkdir, parents=True, exist_ok=True)

            # File will be created on first flush by AudioUtils
            # We don't need to open it here anymore
            with self._lock:
                self._is_open = True
                self._stop_writing.clear()
                # Start background task to write audio chunks
                self._write_task = asyncio.create_task(self._write_audio_loop())
                logger.info(f"[{self.conversation_id}] Recording started: {self.filepath_merged}")
                return True
        except Exception as e:
            logger.error(f"[{self.conversation_id}] Failed to open recording file: {e}")
            return False

    async def write_audio(self, audio_data: bytes, source: Optional[str] = None) -> None:
        """
        Queue audio data for writing to the recording file.

        This method queues audio chunks which will be written by a background
        task to ensure proper ordering and prevent overlapping.

        Args:
            audio_data: Raw PCM audio bytes to write
            source: Source direction identifier (e.g., "A->B" or "B->A")
        """
        if not self._is_open:
            logger.warning(f"[{self.conversation_id}] Attempted to write to closed recorder")
            return

        try:
            # Append to buffer queue with timestamp and source
            timestamp = time.time()
            async with self._buffer_lock:
                self._audio_buffer.append((timestamp, audio_data, source))
                self._current_buffer_size += len(audio_data)

                # Trigger flush if buffer size limit reached
                if self._current_buffer_size >= self._buffer_size_limit:
                    # Signal the write loop to flush (non-blocking)
                    pass  # The write loop will check and flush
        except Exception as e:
            logger.error(f"[{self.conversation_id}] Error queuing audio data: {e}")

    async def close(self) -> None:
        """
        Close the WAV file and finalize the recording.
        """
        # Stop the write loop and flush remaining buffer
        if self._write_task:
            self._stop_writing.set()
            # Wait for write task to finish flushing
            try:
                await self._write_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"[{self.conversation_id}] Error in write task: {e}")
            self._write_task = None

        # Final flush of any remaining chunks
        await self._flush_buffer()

        # Merge the separate audio files into a single mixed recording
        try:
            from utils.audio_utils import AudioUtils

            # Check if both separate files exist
            if self.filepath_agent_a.exists() or self.filepath_agent_b.exists():
                await asyncio.to_thread(
                    AudioUtils.merge_audio_files,
                    str(self.filepath_agent_a),
                    str(self.filepath_agent_b),
                    str(self.filepath_merged),
                    self.sample_rate,
                    self.sample_width,
                )
                logger.info(f"[{self.conversation_id}] Merged audio files into: {self.filepath_merged}")
        except Exception as e:
            logger.error(f"[{self.conversation_id}] Error merging audio files: {e}. Separate files preserved.")

        try:
            # File operations are handled by AudioUtils, just mark as closed
            def _close_file():
                with self._lock:
                    if self._wav_file is not None:
                        self._wav_file.close()
                        self._wav_file = None
                    self._is_open = False

                    # Log file size of merged file
                    if self.filepath_merged.exists():
                        size_mb = self.filepath_merged.stat().st_size / (1024 * 1024)
                        logger.info(f"[{self.conversation_id}] Recording saved: {self.filepath_merged} ({size_mb:.2f} MB)")

            await asyncio.to_thread(_close_file)
        except Exception as e:
            logger.error(f"[{self.conversation_id}] Error closing recording file: {e}")

    async def _flush_buffer(self) -> None:
        """
        Flush buffered audio chunks to file.

        This method extracts chunks from the buffer, groups them by source,
        and writes them to separate files for each direction.
        """
        async with self._buffer_lock:
            if not self._audio_buffer:
                return

            # Extract all chunks and clear buffer
            chunks = list(self._audio_buffer)
            self._audio_buffer.clear()
            self._current_buffer_size = 0

        # Sort chunks by timestamp to maintain chronological order
        chunks.sort(key=lambda x: x[0])  # Sort by timestamp

        # Group chunks by source direction
        agent_a_chunks = [(t, data) for t, data, src in chunks if src == "A->B"]
        agent_b_chunks = [(t, data) for t, data, src in chunks if src == "B->A"]

        # Write each direction to separate files
        try:
            from utils.audio_utils import AudioUtils

            # Close the file handle if it's open (AudioUtils will handle file operations)
            if self._wav_file is not None:
                with self._lock:
                    if self._wav_file is not None:
                        self._wav_file.close()
                        self._wav_file = None

            # Write Agent A audio (A->B direction)
            if agent_a_chunks:
                raw_audio_chunks_a = [audio_data for _, audio_data in agent_a_chunks]
                await asyncio.to_thread(
                    AudioUtils.to_thread_flush_audio_frames,
                    raw_audio_chunks_a,
                    str(self.filepath_agent_a),
                    self.sample_rate,
                    self.channels,
                    self.sample_width,
                    convert_mulaw=False,
                )

            # Write Agent B audio (B->A direction)
            if agent_b_chunks:
                raw_audio_chunks_b = [audio_data for _, audio_data in agent_b_chunks]
                await asyncio.to_thread(
                    AudioUtils.to_thread_flush_audio_frames,
                    raw_audio_chunks_b,
                    str(self.filepath_agent_b),
                    self.sample_rate,
                    self.channels,
                    self.sample_width,
                    convert_mulaw=False,
                )
        except Exception as e:
            logger.error(f"[{self.conversation_id}] Error writing buffered audio: {e}")

    async def _convert_audio(self, audio_data: bytes) -> bytes:
        """
        Convert audio data if needed.

        Base class implementation just returns the data as-is.
        Subclasses (like TwilioConversationRecorder) override this to convert mulaw to PCM.

        Args:
            audio_data: Raw audio bytes

        Returns:
            Converted audio bytes (or original if no conversion needed)
        """
        return audio_data

    async def _write_audio_loop(self) -> None:
        """
        Background task that monitors the buffer and flushes when size limit is reached.
        """
        try:
            while not self._stop_writing.is_set() or self._audio_buffer:
                # Check if buffer needs flushing
                should_flush = False
                async with self._buffer_lock:
                    if self._current_buffer_size >= self._buffer_size_limit:
                        should_flush = True

                if should_flush:
                    await self._flush_buffer()
                elif not self._stop_writing.is_set():
                    # Wait a bit before checking again
                    await asyncio.sleep(0.01)
                else:
                    # Stop event set, flush remaining and exit
                    await self._flush_buffer()
                    break
        except asyncio.CancelledError:
            # Flush any remaining chunks before exiting
            await self._flush_buffer()
            raise
        except Exception as e:
            logger.error(f"[{self.conversation_id}] Error in write audio loop: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        return False

    @property
    def is_open(self) -> bool:
        """Check if the recorder is currently open."""
        return self._is_open
