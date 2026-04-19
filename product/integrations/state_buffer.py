"""
StateBuffer for SCYTHE telemetry.

This module provides a minimal, thread-safe container for the **latest**
`TelemetryFrame` received from any TelemetrySource.

What it IS:
    - A small, in-memory buffer holding the most recent telemetry snapshot.
    - A place to check when the last update arrived and whether it is stale.

What it is NOT:
    - It does **not** modify, smooth, filter, or predict telemetry.
    - It does **not** perform any physics, estimation, or decision logic.
    - It does **not** know or care about the specific UAV or hardware.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from product.integrations.telemetry_contract import TelemetryFrame


class StateBuffer:
    """
    Thread-safe holder for the most recent TelemetryFrame.

    The buffer is designed for simple producer/consumer patterns where one
    component (e.g. a TelemetrySource) calls `update()` whenever a new frame
    is available, and other components call `get_latest()` to read the most
    recent snapshot.

    Telemetry values are stored **exactly** as provided in the frame; this
    class never modifies or derives new telemetry values.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[TelemetryFrame] = None
        self._last_update_time: Optional[float] = None  # wall-clock seconds
        # Track update rate: store last N update timestamps (simple sliding window).
        self._update_times: list[float] = []  # wall-clock seconds
        self._max_update_history = 10  # keep last 10 updates for rate estimation

    def update(self, frame: TelemetryFrame) -> None:
        """
        Store a new telemetry frame as the latest snapshot.

        Parameters
        ----------
        frame : TelemetryFrame
            The new telemetry snapshot. Its contents are stored **verbatim**
            and are never modified by this class.
        """
        now = time.time()
        with self._lock:
            self._latest = frame
            self._last_update_time = now
            # Track update times for rate estimation.
            self._update_times.append(now)
            if len(self._update_times) > self._max_update_history:
                self._update_times.pop(0)

    def get_latest(self) -> Optional[TelemetryFrame]:
        """
        Return the most recent TelemetryFrame, or None if no frame seen yet.

        The returned object is the same instance that was passed into
        `update()`. Callers must treat it as immutable and must NOT modify
        its fields in place.
        """
        with self._lock:
            return self._latest

    def is_stale(self, max_age_seconds: float) -> bool:
        """
        Check whether the latest frame is older than the given age.

        Parameters
        ----------
        max_age_seconds : float
            Maximum acceptable age (in wall-clock seconds) for the buffer
            to be considered \"fresh\".

        Returns
        -------
        bool
            True if:
              - no telemetry has ever been received, OR
              - the time since the last update is greater than max_age_seconds.
        """
        now = time.time()
        with self._lock:
            if self._last_update_time is None:
                return True
            age = now - self._last_update_time
        return age > max_age_seconds

    def get_age_seconds(self) -> Optional[float]:
        """
        Get the age (in wall-clock seconds) of the latest telemetry frame.

        Returns
        -------
        Optional[float]
            Age in seconds, or None if no telemetry has been received.
        """
        now = time.time()
        with self._lock:
            if self._last_update_time is None:
                return None
            return now - self._last_update_time

    def estimate_update_rate_hz(self) -> Optional[float]:
        """
        Estimate current update rate in Hz from recent update history.

        Returns
        -------
        Optional[float]
            Estimated update rate in Hz, or None if insufficient history
            (< 2 updates) to compute a rate.
        """
        with self._lock:
            if len(self._update_times) < 2:
                return None
            # Compute average interval from recent updates.
            intervals = [
                self._update_times[i] - self._update_times[i - 1]
                for i in range(1, len(self._update_times))
            ]
            avg_interval = sum(intervals) / len(intervals)
            if avg_interval <= 0:
                return None
            return 1.0 / avg_interval

