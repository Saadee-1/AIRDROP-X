"""
MOCK / SIMULATED telemetry source for SCYTHE.

This module is for **testing only**. It provides a simple, deterministic
`TelemetrySource` implementation that:

- generates synthetic `TelemetryFrame` instances at a configurable update rate
- simulates a reasonable UAV motion pattern (simple circular trajectory)
- feeds frames into a provided `StateBuffer`

No real UAV or telemetry hardware is required.

IMPORTANT:
    - This source is clearly labeled MOCK / SIMULATED.
    - It introduces **no randomness** unless a caller explicitly chooses to
      randomize parameters before constructing the source.
    - It performs no physics, filtering, or decision logic.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

from product.integrations.telemetry_contract import TelemetryFrame, TelemetrySource
from product.integrations.state_buffer import StateBuffer


class MockTelemetrySource(TelemetrySource):
    """
    Deterministic, MOCK / SIMULATED telemetry source.

    Motion model:
        - simple horizontal circle in a local Cartesian frame
          (x forward, y right, z up)
        - constant altitude (z = z0)
        - constant angular rate
        - roll = 0, pitch = 0
        - yaw follows direction of travel (tangent to the circle)

    Time base:
        - `timestamp` in TelemetryFrame is **mission-relative**, starting
          from zero at the moment `start()` is called.
    """

    def __init__(
        self,
        buffer: StateBuffer,
        update_rate_hz: float = 10.0,
        radius_m: float = 50.0,
        altitude_m: float = 100.0,
        angular_rate_rad_s: Optional[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        buffer : StateBuffer
            StateBuffer that will receive TelemetryFrame updates.
        update_rate_hz : float, optional
            Rate at which frames are produced, in Hertz (default 10 Hz).
        radius_m : float, optional
            Radius of the circular trajectory in meters.
        altitude_m : float, optional
            Constant altitude (z) in meters.
        angular_rate_rad_s : float, optional
            Angular rate in radians/second. If None, a default angular rate
            is chosen so that one full circle takes ~120 seconds.
        """
        self._buffer = buffer
        self._dt = 1.0 / max(update_rate_hz, 1e-6)
        self._radius = float(radius_m)
        self._altitude = float(altitude_m)
        if angular_rate_rad_s is None:
            # Default: one full circle in ~120 seconds (slow, readable motion).
            self._omega = 2.0 * math.pi / 120.0
        else:
            self._omega = float(angular_rate_rad_s)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._alive_lock = threading.Lock()
        self._running = False

    # --- TelemetrySource interface ---

    def start(self) -> None:
        """
        Start producing synthetic telemetry frames.

        Safe to call once. Repeated calls when already running are ignored.
        """
        with self._alive_lock:
            if self._running:
                return
            self._running = True
            self._stop_event.clear()

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Stop producing frames and wait for the background loop to exit.
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
        Return True if the mock source is actively running its update loop.
        """
        with self._alive_lock:
            return self._running

    # --- Internal loop ---

    def _run_loop(self) -> None:
        """
        Background thread that generates deterministic circular motion frames.
        """
        t0 = time.time()
        while not self._stop_event.is_set():
            now = time.time()
            t_rel = now - t0  # mission-relative time (seconds)

            # Simple circular trajectory in x-y plane.
            theta = self._omega * t_rel
            x = self._radius * math.cos(theta)
            y = self._radius * math.sin(theta)
            z = self._altitude

            vx = -self._radius * self._omega * math.sin(theta)
            vy = self._radius * self._omega * math.cos(theta)
            vz = 0.0

            # Yaw aligned with direction of travel (tangent to circle).
            yaw = math.atan2(vy, vx)
            roll = 0.0
            pitch = 0.0

            frame = TelemetryFrame(
                timestamp=t_rel,
                position=(x, y, z),
                velocity=(vx, vy, vz),
                attitude=(roll, pitch, yaw),
                source="simulated",
            )
            self._buffer.update(frame)

            # Sleep until next update, respecting configured rate.
            time.sleep(self._dt)

