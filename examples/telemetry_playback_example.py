"""
Example usage of TelemetryPlaybackSource.

This demonstrates how to:
- Load telemetry from CSV or JSON files
- Replay at real-time or accelerated speeds
- Feed frames into StateBuffer for SCYTHE consumption
"""

from product.integrations.state_buffer import StateBuffer
from product.integrations.telemetry_playback import TelemetryPlaybackSource
import time


def example_csv_playback():
    """Example: Replay telemetry from CSV file at real-time speed."""
    buffer = StateBuffer()
    source = TelemetryPlaybackSource(
        buffer=buffer,
        file_path="examples/example_telemetry.csv",
        speed_multiplier=1.0,  # Real-time
        loop=False,
    )

    print(f"Loaded {source.get_frame_count()} frames")
    print(f"Duration: {source.get_duration_seconds():.1f} seconds")

    source.start()
    print("Playback started...")

    # Monitor buffer for a few seconds.
    for i in range(6):
        time.sleep(1.0)
        latest = buffer.get_latest()
        if latest:
            print(
                f"t={i}s: pos={latest.position}, vel={latest.velocity}, "
                f"source={latest.source}"
            )

    source.stop()
    print("Playback stopped.")


def example_json_accelerated():
    """Example: Replay telemetry from JSON file at 2x speed."""
    buffer = StateBuffer()
    source = TelemetryPlaybackSource(
        buffer=buffer,
        file_path="examples/example_telemetry.json",
        speed_multiplier=2.0,  # 2x speed
        loop=True,  # Loop indefinitely
    )

    print(f"Loaded {source.get_frame_count()} frames")
    print(f"Duration: {source.get_duration_seconds():.1f} seconds (log time)")
    print("Playing at 2x speed (will loop)...")

    source.start()

    # Monitor for a few seconds (wall-clock time).
    for i in range(4):
        time.sleep(1.0)
        latest = buffer.get_latest()
        if latest:
            print(
                f"t={i}s (wall): pos={latest.position}, "
                f"log_timestamp={latest.timestamp:.1f}"
            )

    source.stop()
    print("Playback stopped.")


if __name__ == "__main__":
    print("=== CSV Playback Example ===")
    example_csv_playback()
    print("\n=== JSON Accelerated Playback Example ===")
    example_json_accelerated()
