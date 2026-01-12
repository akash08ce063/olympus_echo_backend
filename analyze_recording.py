#!/usr/bin/env python3
"""
Analyze WAV recording to detect breaks, gaps, and timing issues.

Usage: python analyze_recording.py <recording.wav>
"""

import sys
import wave
import numpy as np
from pathlib import Path


def analyze_wav_file(wav_path: str):
    """Analyze WAV file for breaks, gaps, and timing issues."""
    print(f"Analyzing: {wav_path}\n")
    print("=" * 80)

    with wave.open(wav_path, "rb") as wf:
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        duration = n_frames / sample_rate

        print(f"Sample Rate: {sample_rate} Hz")
        print(f"Sample Width: {sample_width} bytes")
        print(f"Channels: {n_channels}")
        print(f"Total Frames: {n_frames}")
        print(f"Duration: {duration:.3f} seconds ({duration/60:.2f} minutes)")
        print()

        # Read all audio data
        audio_data = wf.readframes(n_frames)

        # Convert to numpy array for analysis
        if sample_width == 2:  # 16-bit
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
        elif sample_width == 1:  # 8-bit
            audio_array = np.frombuffer(audio_data, dtype=np.int8)
        else:
            print(f"Unsupported sample width: {sample_width}")
            return

        # Calculate expected chunk size (20ms chunks at 8kHz = 160 samples)
        chunk_duration_ms = 20
        chunk_size_samples = int(sample_rate * chunk_duration_ms / 1000)
        expected_chunks = int(n_frames / chunk_size_samples)

        print(f"Expected chunk size: {chunk_size_samples} samples ({chunk_duration_ms}ms)")
        print(f"Expected number of chunks: {expected_chunks}")
        print()

        # Analyze in chunks
        print("Analyzing audio chunks...")
        print("-" * 80)

        silence_threshold = 100  # Amplitude threshold for silence
        chunk_analysis = []
        total_silence_duration = 0
        total_audio_duration = 0
        gap_count = 0
        gaps = []

        for i in range(0, len(audio_array), chunk_size_samples):
            chunk = audio_array[i : i + chunk_size_samples]
            if len(chunk) < chunk_size_samples:
                break

            chunk_time = i / sample_rate
            max_amplitude = np.max(np.abs(chunk))
            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))

            is_silence = max_amplitude < silence_threshold

            chunk_analysis.append(
                {
                    "chunk_num": len(chunk_analysis),
                    "time": chunk_time,
                    "max_amplitude": max_amplitude,
                    "rms": rms,
                    "is_silence": is_silence,
                }
            )

            if is_silence:
                total_silence_duration += chunk_size_samples / sample_rate
            else:
                total_audio_duration += chunk_size_samples / sample_rate

        # Detect gaps (consecutive silence chunks)
        consecutive_silence = 0
        gap_start = None

        for chunk in chunk_analysis:
            if chunk["is_silence"]:
                if gap_start is None:
                    gap_start = chunk["time"]
                consecutive_silence += 1
            else:
                if consecutive_silence > 0:
                    gap_duration = consecutive_silence * (chunk_size_samples / sample_rate)
                    if gap_duration > 0.1:  # Gaps longer than 100ms
                        gap_count += 1
                        gaps.append(
                            {
                                "start": gap_start,
                                "end": chunk["time"],
                                "duration": gap_duration,
                            }
                        )
                    consecutive_silence = 0
                    gap_start = None

        # Check for final gap
        if consecutive_silence > 0:
            gap_duration = consecutive_silence * (chunk_size_samples / sample_rate)
            if gap_duration > 0.1:
                gap_count += 1
                gaps.append(
                    {
                        "start": gap_start,
                        "end": chunk_analysis[-1]["time"] + (chunk_size_samples / sample_rate),
                        "duration": gap_duration,
                    }
                )

        print(f"Total chunks analyzed: {len(chunk_analysis)}")
        print(
            f"Silence duration: {total_silence_duration:.3f}s ({total_silence_duration/duration*100:.1f}%)"
        )
        print(
            f"Audio duration: {total_audio_duration:.3f}s ({total_audio_duration/duration*100:.1f}%)"
        )
        print(f"Gaps detected (>100ms): {gap_count}")
        print()

        # Report gaps
        if gaps:
            print("Gap Analysis:")
            print("-" * 80)
            for i, gap in enumerate(gaps[:20], 1):
                print(
                    f"Gap {i}: {gap['start']:.3f}s - {gap['end']:.3f}s (duration: {gap['duration']:.3f}s)"
                )
            if len(gaps) > 20:
                print(f"... and {len(gaps) - 20} more gaps")
            print()

        # Check for missing chunks (timing gaps)
        print("Timing Analysis:")
        print("-" * 80)
        expected_duration = len(chunk_analysis) * (chunk_size_samples / sample_rate)
        actual_duration = duration
        duration_diff = actual_duration - expected_duration

        print(f"Expected duration (based on chunks): {expected_duration:.3f}s")
        print(f"Actual file duration: {actual_duration:.3f}s")
        print(f"Difference: {duration_diff:.3f}s ({abs(duration_diff)/expected_duration*100:.2f}%)")

        if abs(duration_diff) > 0.1:
            print("⚠️  WARNING: Significant duration mismatch!")
        print()

        # Check for sudden drops (potential missing chunks)
        print("Break Detection:")
        print("-" * 80)
        breaks = []
        for i in range(1, len(chunk_analysis)):
            prev_chunk = chunk_analysis[i - 1]
            curr_chunk = chunk_analysis[i]

            if not prev_chunk["is_silence"] and curr_chunk["is_silence"]:
                if prev_chunk["max_amplitude"] > 1000:
                    breaks.append(
                        {
                            "time": curr_chunk["time"],
                            "type": "audio_to_silence",
                            "prev_amplitude": prev_chunk["max_amplitude"],
                        }
                    )

        if breaks:
            print(f"Detected {len(breaks)} potential breaks:")
            for i, break_info in enumerate(breaks[:10], 1):
                print(f"  Break {i}: at {break_info['time']:.3f}s")
            if len(breaks) > 10:
                print(f"  ... and {len(breaks) - 10} more")
        print()

        # Summary
        print("=" * 80)
        print("SUMMARY:")
        print(f"  Total duration: {duration:.3f}s")
        print(
            f"  Silence: {total_silence_duration:.3f}s ({total_silence_duration/duration*100:.1f}%)"
        )
        print(f"  Audio: {total_audio_duration:.3f}s ({total_audio_duration/duration*100:.1f}%)")
        print(f"  Gaps (>100ms): {gap_count}")
        print(f"  Breaks: {len(breaks)}")
        if duration_diff > 0.1:
            print(f"  ⚠️  Duration mismatch: {duration_diff:.3f}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_recording.py <recording.wav>")
        sys.exit(1)

    wav_path = sys.argv[1]
    if not Path(wav_path).exists():
        print(f"Error: File not found: {wav_path}")
        sys.exit(1)

    try:
        analyze_wav_file(wav_path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
