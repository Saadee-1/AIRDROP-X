"""
Telemetry playback source for SCYTHE.

This module provides a `TelemetryPlaybackSource` that reads telemetry frames
from files (CSV or JSON) and replays them at configurable speeds.

Use cases:
- Offline validation of SCYTHE with recorded telemetry
- Reproducible experiments with known telemetry sequences
- Paper-ready results using consistent telemetry datasets

This source is UAV-agnostic and treats telemetry as generic position/velocity/attitude data.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path
from typing import Optional, List

from product.integrations.telemetry_contract import TelemetryFrame, TelemetrySource
from product.integrations.state_buffer import StateBuffer


class TelemetryPlaybackSource(TelemetrySource):
    """
    Replay telemetry frames from a file at configurable speed.

    Supports CSV and JSON formats. Frames are replayed respecting their original
    timestamps, scaled by a speed multiplier (1.0 = real-time, 2.0 = 2x speed, etc.).

    File formats:
        CSV: Header row with columns: timestamp, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, roll, pitch, yaw, source
        JSON: Array of objects with keys: timestamp, position (array), velocity (array), attitude (array), source
    """

    def __init__(
        self,
        buffer: StateBuffer,
        file_path: str | Path,
        speed_multiplier: float = 1.0,
        loop: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        buffer : StateBuffer
            StateBuffer that will receive TelemetryFrame updates.
        file_path : str | Path
            Path to telemetry file (CSV or JSON).
        speed_multiplier : float, optional
            Playback speed multiplier. 1.0 = real-time, 2.0 = 2x speed, etc.
            Must be > 0. Default 1.0.
        loop : bool, optional
            If True, loop playback indefinitely. If False, stop when file ends.
            Default False.
        """
        self._buffer = buffer
        self._file_path = Path(file_path)
        if speed_multiplier <= 0:
            raise ValueError(f"speed_multiplier must be > 0, got {speed_multiplier}")
        self._speed_multiplier = float(speed_multiplier)
        self._loop = bool(loop)

        self._frames: List[TelemetryFrame] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._alive_lock = threading.Lock()
        self._running = False

        # Load frames from file.
        self._load_frames()

    # --- TelemetrySource interface ---

    def start(self) -> None:
        """
        Start replaying telemetry frames.

        Safe to call once. Repeated calls when already running are ignored.
        """
        with self._alive_lock:
            if self._running:
                return
            if not self._frames:
                raise RuntimeError("No frames loaded; cannot start playback")
            self._running = True
            self._stop_event.clear()

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Stop replaying frames and wait for the background loop to exit.
        """
        with self._alive_lock:
            if not self._running:
                return
            self._running = False
            self._stop_event.set()

        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def is_alive(self) -> bool:
        """
        Return True if playback is actively running.
        """
        with self._alive_lock:
            return self._running

    # --- File loading ---

    def _load_frames(self) -> None:
        """
        Load telemetry frames from file (CSV or JSON).

        Raises
        ------
        FileNotFoundError
            If file does not exist.
        ValueError
            If file format is invalid or unsupported.
        """
        if not self._file_path.exists():
            raise FileNotFoundError(f"Telemetry file not found: {self._file_path}")

        suffix = self._file_path.suffix.lower()
        if suffix == ".csv":
            self._load_csv()
        elif suffix == ".json":
            self._load_json()
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. Supported: .csv, .json"
            )

    def _load_csv(self) -> None:
        """
        Load frames from CSV file.

        Expected format:
            timestamp,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,roll,pitch,yaw,source
            0.0,10.0,20.0,100.0,5.0,0.0,0.0,0.0,0.0,1.57,replayed
            ...
        """
        frames = []
        with open(self._file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    frame = TelemetryFrame(
                        timestamp=float(row["timestamp"]),
                        position=(
                            float(row["pos_x"]),
                            float(row["pos_y"]),
                            float(row["pos_z"]),
                        ),
                        velocity=(
                            float(row["vel_x"]),
                            float(row["vel_y"]),
                            float(row["vel_z"]),
                        ),
                        attitude=(
                            float(row["roll"]),
                            float(row["pitch"]),
                            float(row["yaw"]),
                        ),
                        source=row.get("source", "replayed"),
                    )
                    frames.append(frame)
                except (KeyError, ValueError) as e:
                    raise ValueError(
                        f"Invalid CSV row: {row}. Error: {e}"
                    ) from e

        if not frames:
            raise ValueError(f"No valid frames found in CSV file: {self._file_path}")
        self._frames = frames

    def _load_json(self) -> None:
        """
        Load frames from JSON file.

        Expected format:
            [
                {
                    "timestamp": 0.0,
                    "position": [10.0, 20.0, 100.0],
                    "velocity": [5.0, 0.0, 0.0],
                    "attitude": [0.0, 0.0, 1.57],
                    "source": "replayed"
                },
                ...
            ]
        """
        with open(self._file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                f"JSON file must contain an array of frames, got {type(data).__name__}"
            )

        frames = []
        for i, obj in enumerate(data):
            try:
                frame = TelemetryFrame(
                    timestamp=float(obj["timestamp"]),
                    position=tuple(float(x) for x in obj["position"]),
                    velocity=tuple(float(x) for x in obj["velocity"]),
                    attitude=tuple(float(x) for x in obj["attitude"]),
                    source=str(obj.get("source", "replayed")),
                )
                if len(frame.position) != 3:
                    raise ValueError("position must have 3 elements")
                if len(frame.velocity) != 3:
                    raise ValueError("velocity must have 3 elements")
                if len(frame.attitude) != 3:
                    raise ValueError("attitude must have 3 elements")
                frames.append(frame)
            except (KeyError, ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid JSON frame at index {i}: {obj}. Error: {e}"
                ) from e

        if not frames:
            raise ValueError(f"No valid frames found in JSON file: {self._file_path}")
        self._frames = frames

    # --- Playback loop ---

    def _run_loop(self) -> None:
        """
        Background thread that replays frames at configured speed.
        """
        frame_idx = 0
        t0_wall = time.time()
        t0_log = None  # First frame's log timestamp

        while not self._stop_event.is_set():
            if frame_idx >= len(self._frames):
                if self._loop:
                    # Reset to beginning for looping.
                    frame_idx = 0
                    t0_wall = time.time()
                    t0_log = None
                else:
                    # End of file; stop playback.
                    with self._alive_lock:
                        self._running = False
                    break

            frame = self._frames[frame_idx]

            # Initialize time base on first frame.
            if t0_log is None:
                t0_log = frame.timestamp
                t0_wall = time.time()

            # Compute when this frame should be played back (wall-clock time).
            log_delta = frame.timestamp - t0_log
            wall_delta = log_delta / self._speed_multiplier
            target_wall_time = t0_wall + wall_delta

            # Sleep until it's time to play this frame.
            now_wall = time.time()
            sleep_time = target_wall_time - now_wall
            if sleep_time > 0:
                # Sleep in small increments to allow early stop.
                sleep_until = time.time() + sleep_time
                while time.time() < sleep_until and not self._stop_event.is_set():
                    time.sleep(min(0.01, sleep_until - time.time()))
            # If sleep_time <= 0, frame is late; play it immediately.

            if self._stop_event.is_set():
                break

            # Update buffer with this frame.
            # Note: We preserve the original timestamp from the log file.
            self._buffer.update(frame)

            frame_idx += 1

    def get_frame_count(self) -> int:
        """
        Return the number of frames loaded from file.

        Returns
        -------
        int
            Number of frames available for playback.
        """
        return len(self._frames)

    def get_duration_seconds(self) -> Optional[float]:
        """
        Return the duration of the telemetry log in seconds.

        Returns
        -------
        Optional[float]
            Duration in seconds (last timestamp - first timestamp), or None if
            no frames are loaded.
        """
        if not self._frames:
            return None
        return self._frames[-1].timestamp - self._frames[0].timestamp
