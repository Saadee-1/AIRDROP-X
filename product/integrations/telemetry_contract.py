"""
Standard, drone-agnostic telemetry data contract for SCYTHE.

This module defines:

- `TelemetryFrame`: one immutable snapshot of vehicle state as seen by the
  ground station at a point in time.
- `TelemetrySource`: abstract base class for any source of telemetry frames
  (log replay, network link, simulation, etc.).

No physics, filtering, or decision logic is implemented here. This is a pure
data/interface definition layer and remains UAV- and hardware-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple
from abc import ABC, abstractmethod


Position3D = Tuple[float, float, float]
Velocity3D = Tuple[float, float, float]
AttitudeRPY = Tuple[float, float, float]


@dataclass(frozen=True)
class TelemetryFrame:
    """
    One telemetry snapshot as seen by SCYTHE at a given time.

    All units and frames are explicitly documented to support auditability.

    Attributes
    ----------
    timestamp : float
        Wall-clock time in **seconds**. Interpretation (e.g. UNIX epoch vs
        mission-relative) is defined by the TelemetrySource, but MUST be
        consistent for a given run.

    position : (float, float, float)
        Vehicle position in **meters** in a local Cartesian frame:
        - x: forward
        - y: right
        - z: up

        NOTE: This is intentionally generic and drone-agnostic. Any mapping
        from a specific UAV or navigation frame into this local frame MUST be
        performed outside this module.

    velocity : (float, float, float)
        Vehicle translational velocity in **meters per second (m/s)** in the
        same local Cartesian frame as `position`:
        - vx: forward component (m/s)
        - vy: right component (m/s)
        - vz: up component (m/s)

    attitude : (float, float, float)
        Vehicle attitude expressed as **roll, pitch, yaw** in **radians**:
        - roll: rotation about x-axis (right-hand rule)
        - pitch: rotation about y-axis (right-hand rule)
        - yaw: rotation about z-axis (right-hand rule)

        This is a generic aerospace-style convention and does not assume any
        specific autopilot or UAV implementation.

    source : str
        Label describing the origin of this frame, for example:
        - \"measured\"  : directly reported by sensors / GNSS / IMU
        - \"estimated\" : fused or filtered estimate
        - \"replayed\"  : from a log file
        - \"simulated\" : from an offline simulation

        SCYTHE treats this as metadata only; no behavior is implied.
    """

    timestamp: float
    position: Position3D
    velocity: Velocity3D
    attitude: AttitudeRPY
    source: str


class TelemetrySource(ABC):
    """
    Abstract producer of TelemetryFrame instances.

    Examples of concrete sources (implemented elsewhere) include:
    - Log file replay
    - Network/MAVLink adapter
    - Synthetic/simulated telemetry generator

    This base class intentionally contains **no UAV- or hardware-specific**
    code. Implementations are responsible for managing connections and for
    exposing frames via their own mechanisms (callbacks, queues, etc.).
    """

    @abstractmethod
    def start(self) -> None:
        """
        Start producing telemetry frames.

        Implementations may spawn background threads, start I/O loops, or
        perform any necessary setup. This method must be idempotent or
        clearly document any restrictions on repeated calls.
        """

    @abstractmethod
    def stop(self) -> None:
        """
        Stop producing telemetry frames and release external resources.

        Implementations must ensure that after `stop()` returns, no further
        TelemetryFrame instances are emitted from this source.
        """

    @abstractmethod
    def is_alive(self) -> bool:
        """
        Return True if the source is actively able to produce frames.

        This is a shallow health check for the source itself (e.g. link up,
        file not exhausted). No physics or decision logic is performed here.
        """

