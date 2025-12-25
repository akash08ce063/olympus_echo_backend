"""
Audio utility functions for thread-safe file operations.
"""

import audioop
import wave
from pathlib import Path


class AudioUtils:
    """Utility class for audio file operations."""

    @staticmethod
    def to_thread_flush_audio_frames(
        audio_buffer: list[bytes],
        filename: str,
        sample_rate: int = 8000,
        channels: int = 1,
        sample_width: int = 2,
        convert_mulaw: bool = False,
    ) -> None:
        """
        Flush audio frames to a WAV file in a thread-safe manner.

        This method combines all audio chunks, optionally converts them,
        and writes to file in one atomic operation. Designed to be called
        via asyncio.to_thread to prevent blocking the event loop.

        Based on the pattern: combine all chunks -> convert all at once -> write all at once

        Args:
            audio_buffer: List of audio frame bytes to write
            filename: Path to the WAV file
            sample_rate: Audio sample rate in Hz (default: 8000)
            channels: Number of audio channels (default: 1 for mono)
            sample_width: Sample width in bytes (default: 2 for 16-bit)
            convert_mulaw: If True, convert mulaw to linear PCM (default: False)
        """
        if not audio_buffer:
            return

        filepath = Path(filename)

        try:
            # Combine all audio chunks (like the sample: b''.join(audio_buffer))
            combined_audio = b"".join(audio_buffer)

            # Convert if needed (like the sample: audioop.ulaw2lin(mulaw_audio, 2))
            if convert_mulaw:
                # Convert mulaw (8-bit) to linear PCM (16-bit)
                combined_audio = audioop.ulaw2lin(combined_audio, sample_width)

            # Check if file exists - if so, we need to append to it
            if filepath.exists():
                # Read existing audio data
                with wave.open(str(filepath), "rb") as rf:
                    params = rf.getparams()
                    existing_audio = rf.readframes(rf.getnframes())

                # Combine existing and new audio
                all_audio = existing_audio + combined_audio

                # Write everything back with same parameters
                with wave.open(str(filepath), "wb") as wf:
                    wf.setnchannels(params.nchannels)
                    wf.setsampwidth(params.sampwidth)
                    wf.setframerate(params.framerate)
                    wf.writeframes(all_audio)
            else:
                # Write new file
                with wave.open(str(filepath), "wb") as wav_file:
                    wav_file.setnchannels(channels)
                    wav_file.setsampwidth(sample_width)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(combined_audio)

        except Exception as e:
            raise RuntimeError(f"Error flushing audio frames to {filename}: {e}")

    @staticmethod
    def merge_audio_files(
        file_a: str,
        file_b: str,
        output_file: str,
        sample_rate: int = 8000,
        sample_width: int = 2,
    ) -> None:
        """
        Merge two WAV files by time-aligning and mixing overlapping samples.

        This method reads both audio files, aligns them by sample position,
        and mixes overlapping samples using audioop.add() with clipping protection.

        Args:
            file_a: Path to first WAV file (Agent A audio)
            file_b: Path to second WAV file (Agent B audio)
            output_file: Path to output merged WAV file
            sample_rate: Audio sample rate in Hz (default: 8000)
            sample_width: Sample width in bytes (default: 2 for 16-bit)
        """
        filepath_a = Path(file_a)
        filepath_b = Path(file_b)
        filepath_out = Path(output_file)

        try:
            # Check if files exist
            if not filepath_a.exists() and not filepath_b.exists():
                raise RuntimeError(f"Neither input file exists: {file_a}, {file_b}")

            # Handle case where one file is empty or missing
            if not filepath_a.exists():
                # Only file_b exists, copy it to output
                with wave.open(str(filepath_b), "rb") as rf:
                    params = rf.getparams()
                    audio_data = rf.readframes(rf.getnframes())
                with wave.open(str(filepath_out), "wb") as wf:
                    wf.setnchannels(params.nchannels)
                    wf.setsampwidth(params.sampwidth)
                    wf.setframerate(params.framerate)
                    wf.writeframes(audio_data)
                return

            if not filepath_b.exists():
                # Only file_a exists, copy it to output
                with wave.open(str(filepath_a), "rb") as rf:
                    params = rf.getparams()
                    audio_data = rf.readframes(rf.getnframes())
                with wave.open(str(filepath_out), "wb") as wf:
                    wf.setnchannels(params.nchannels)
                    wf.setsampwidth(params.sampwidth)
                    wf.setframerate(params.framerate)
                    wf.writeframes(audio_data)
                return

            # Read both files
            with wave.open(str(filepath_a), "rb") as rf_a:
                params_a = rf_a.getparams()
                audio_a = rf_a.readframes(rf_a.getnframes())
                frames_a = rf_a.getnframes()

            with wave.open(str(filepath_b), "rb") as rf_b:
                params_b = rf_b.getparams()
                audio_b = rf_b.readframes(rf_b.getnframes())
                frames_b = rf_b.getnframes()

            # Verify both files have compatible parameters
            if params_a.sampwidth != params_b.sampwidth:
                raise RuntimeError(f"Sample width mismatch: {params_a.sampwidth} vs {params_b.sampwidth}")
            if params_a.framerate != params_b.framerate:
                raise RuntimeError(f"Sample rate mismatch: {params_a.framerate} vs {params_b.framerate}")

            # Determine max duration (in frames)
            max_frames = max(frames_a, frames_b)
            bytes_per_frame = params_a.sampwidth * params_a.nchannels

            # Mix the audio samples
            mixed_audio = bytearray()
            for i in range(max_frames):
                frame_start = i * bytes_per_frame
                frame_end = frame_start + bytes_per_frame

                # Get frame from file_a (or silence if exhausted)
                if frame_start < len(audio_a):
                    frame_a = audio_a[frame_start:frame_end]
                    # Pad if incomplete frame
                    if len(frame_a) < bytes_per_frame:
                        frame_a += b"\x00" * (bytes_per_frame - len(frame_a))
                else:
                    frame_a = b"\x00" * bytes_per_frame

                # Get frame from file_b (or silence if exhausted)
                if frame_start < len(audio_b):
                    frame_b = audio_b[frame_start:frame_end]
                    # Pad if incomplete frame
                    if len(frame_b) < bytes_per_frame:
                        frame_b += b"\x00" * (bytes_per_frame - len(frame_b))
                else:
                    frame_b = b"\x00" * bytes_per_frame

                # Mix frames using audioop.add() with clipping protection
                # Scale down to 50% each to prevent clipping
                mixed_frame = audioop.add(
                    audioop.mul(frame_a, params_a.sampwidth, 0.5),
                    audioop.mul(frame_b, params_a.sampwidth, 0.5),
                    params_a.sampwidth,
                )
                mixed_audio.extend(mixed_frame)

            # Write merged result
            with wave.open(str(filepath_out), "wb") as wf:
                wf.setnchannels(params_a.nchannels)
                wf.setsampwidth(params_a.sampwidth)
                wf.setframerate(params_a.framerate)
                wf.writeframes(bytes(mixed_audio))

        except Exception as e:
            raise RuntimeError(f"Error merging audio files {file_a} and {file_b}: {e}")
