"""
Audio conversion service for handling encoding and sample rate conversions.

This module provides utilities to convert audio data between different
encodings (mulaw, pcm) and sample rates.
"""

import audioop
import numpy as np
from typing import Literal

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from telemetrics.logger import logger

EncodingType = Literal["mulaw", "pcm16", "pcm"]
SampleRateType = int


class AudioConverter:
    """Service class for converting audio between different encodings and sample rates."""

    @staticmethod
    def convert_encoding(
        audio_data: bytes,
        from_encoding: EncodingType,
        to_encoding: EncodingType,
        sample_width: int = 2,
    ) -> bytes:
        """
        Convert audio data from one encoding to another.

        Args:
            audio_data: Raw audio bytes
            from_encoding: Source encoding type ("mulaw", "pcm16", "pcm")
            to_encoding: Target encoding type ("mulaw", "pcm16", "pcm")
            sample_width: Sample width in bytes (default: 2 for 16-bit)

        Returns:
            Converted audio bytes
        """
        if from_encoding == to_encoding:
            return audio_data

        # Convert to PCM16 as intermediate format
        if from_encoding == "mulaw":
            pcm_data = audioop.ulaw2lin(audio_data, sample_width)
        elif from_encoding in ("pcm16", "pcm"):
            pcm_data = audio_data
        else:
            raise ValueError(f"Unsupported source encoding: {from_encoding}")

        # Convert from PCM16 to target encoding
        if to_encoding == "mulaw":
            return audioop.lin2ulaw(pcm_data, sample_width)
        elif to_encoding in ("pcm16", "pcm"):
            return pcm_data
        else:
            raise ValueError(f"Unsupported target encoding: {to_encoding}")

    @staticmethod
    def resample_audio(
        audio_data: bytes,
        from_sample_rate: SampleRateType,
        to_sample_rate: SampleRateType,
        sample_width: int = 2,
        encoding: EncodingType = "pcm16",
    ) -> bytes:
        """
        Resample audio data from one sample rate to another.

        Args:
            audio_data: Raw audio bytes
            from_sample_rate: Source sample rate in Hz
            to_sample_rate: Target sample rate in Hz
            sample_width: Sample width in bytes (default: 2 for 16-bit)
            encoding: Audio encoding type (default: "pcm16")

        Returns:
            Resampled audio bytes
        """
        if from_sample_rate == to_sample_rate:
            return audio_data

        if not LIBROSA_AVAILABLE:
            logger.warning("librosa not available, using simple resampling")
            return AudioConverter._simple_resample(
                audio_data, from_sample_rate, to_sample_rate, sample_width
            )

        try:
            # Convert bytes to numpy array
            if encoding == "mulaw":
                # Convert mulaw to PCM16 first
                pcm_data = audioop.ulaw2lin(audio_data, sample_width)
            else:
                pcm_data = audio_data

            # Convert to numpy array
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)

            # Convert to float32 normalized
            audio_float = audio_array.astype(np.float32) / 32768.0

            # Resample using librosa
            resampled_float = librosa.resample(
                audio_float,
                orig_sr=from_sample_rate,
                target_sr=to_sample_rate,
            )

            # Convert back to int16
            resampled_int16 = (resampled_float * 32767.0).astype(np.int16)
            resampled_bytes = resampled_int16.tobytes()

            # Convert back to original encoding if needed
            if encoding == "mulaw":
                return audioop.lin2ulaw(resampled_bytes, sample_width)
            else:
                return resampled_bytes

        except Exception as e:
            logger.error(f"Error resampling audio: {e}, falling back to simple resampling")
            return AudioConverter._simple_resample(
                audio_data, from_sample_rate, to_sample_rate, sample_width
            )

    @staticmethod
    def _simple_resample(
        audio_data: bytes,
        from_sample_rate: SampleRateType,
        to_sample_rate: SampleRateType,
        sample_width: int = 2,
    ) -> bytes:
        """
        Simple resampling using decimation/interpolation.

        This is a fallback when librosa is not available.
        Only works well for integer ratios (e.g., 8kHz -> 16kHz).
        """
        if from_sample_rate == to_sample_rate:
            return audio_data

        ratio = to_sample_rate / from_sample_rate

        if ratio == int(ratio):
            # Integer ratio - simple decimation/interpolation
            if ratio > 1:
                # Upsample: repeat samples
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                upsampled = np.repeat(audio_array, int(ratio))
                return upsampled.tobytes()
            else:
                # Downsample: take every Nth sample
                step = int(1 / ratio)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                downsampled = audio_array[::step]
                return downsampled.tobytes()
        else:
            # Non-integer ratio - use linear interpolation approximation
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            original_length = len(audio_array)
            new_length = int(original_length * ratio)
            indices = np.linspace(0, original_length - 1, new_length)
            resampled = np.interp(indices, np.arange(original_length), audio_array)
            return resampled.astype(np.int16).tobytes()

    @staticmethod
    def convert_and_resample(
        audio_data: bytes,
        from_encoding: EncodingType,
        to_encoding: EncodingType,
        from_sample_rate: SampleRateType,
        to_sample_rate: SampleRateType,
        sample_width: int = 2,
    ) -> bytes:
        """
        Convert audio encoding and resample in one operation.

        Args:
            audio_data: Raw audio bytes
            from_encoding: Source encoding type
            to_encoding: Target encoding type
            from_sample_rate: Source sample rate in Hz
            to_sample_rate: Target sample rate in Hz
            sample_width: Sample width in bytes (default: 2)

        Returns:
            Converted and resampled audio bytes
        """
        # First convert encoding to PCM16 if needed
        if from_encoding != "pcm16":
            pcm_data = AudioConverter.convert_encoding(
                audio_data, from_encoding, "pcm16", sample_width
            )
        else:
            pcm_data = audio_data

        # Resample
        resampled_pcm = AudioConverter.resample_audio(
            pcm_data, from_sample_rate, to_sample_rate, sample_width, "pcm16"
        )

        # Convert to target encoding if needed
        if to_encoding != "pcm16":
            return AudioConverter.convert_encoding(
                resampled_pcm, "pcm16", to_encoding, sample_width
            )
        else:
            return resampled_pcm

