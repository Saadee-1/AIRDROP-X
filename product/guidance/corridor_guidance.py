"""
Corridor guidance from release envelope result.

Converts envelope (optimal offset and release time) into a guidance result:
target release point, guidance vector, heading error, and drop status.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _wrap_angle(rad: float) -> float:
    """Wrap angle to [-pi, pi]."""
    rad = float(rad)
    while rad > np.pi:
        rad -= 2.0 * np.pi
    while rad < -np.pi:
        rad += 2.0 * np.pi
    return rad


@dataclass
class GuidanceResult:
    """Guidance output for corridor-based release."""

    status: str
    target_release_point: np.ndarray  # (3,)
    guidance_vector: np.ndarray       # (2,)
    heading_error: float
    distance_to_corridor: float
    time_to_release: float
    optimal_p_hit: float


def compute_corridor_guidance(
    envelope_result: Any,
    pos_uav: np.ndarray,
    vel_uav: np.ndarray,
    current_time: float,
    *,
    threshold: float = 0.5,
    approach_distance_m: float = 5.0,
    drop_now_time_threshold: float = 0.5,
) -> GuidanceResult:
    """
    Compute guidance from a release envelope result and current UAV state.

    Parameters
    ----------
    envelope_result : ReleaseEnvelopeResult
        Result of compute_release_envelope.
    pos_uav : array_like, shape (3,)
        Current UAV position (x, y, z).
    vel_uav : array_like, shape (3,)
        Current UAV velocity (x, y, z).
    current_time : float
        Current time (s); reserved for future use.
    threshold : float
        Minimum optimal_p_hit for drop eligibility; below this status is NO_DROP.
    approach_distance_m : float
        Unused; retained for API compatibility.
    drop_now_time_threshold : float
        Drop window (s); if abs(time_to_release) < this, status is DROP_NOW.
    Status logic uses lateral_distance = abs(entry.offset) and corridor_half_width
    from envelope feasible_offsets; APPROACH_CORRIDOR when lateral_distance > corridor_half_width.

    Returns
    -------
    GuidanceResult
    """
    pos_uav = np.asarray(pos_uav, dtype=float).reshape(3)
    vel_uav = np.asarray(vel_uav, dtype=float).reshape(3)
    threshold = float(np.clip(threshold, 0.05, 0.95))

    # 1) Best envelope entry by optimal_p_hit
    envelope = getattr(envelope_result, "envelope", [])
    if not envelope:
        # No envelope: return a safe no-drop result
        return GuidanceResult(
            status="NO_DROP",
            target_release_point=pos_uav.copy(),
            guidance_vector=np.zeros(2, dtype=float),
            heading_error=0.0,
            distance_to_corridor=0.0,
            time_to_release=0.0,
            optimal_p_hit=0.0,
        )

    entry = max(envelope, key=lambda e: getattr(e, "optimal_p_hit", 0.0))

    # 2) Forward direction (unit vector in horizontal plane)
    v_xy = vel_uav[:2]
    norm_xy = float(np.linalg.norm(v_xy))
    if norm_xy < 1e-9:
        v_hat = np.array([1.0, 0.0], dtype=float)
    else:
        v_hat = v_xy / norm_xy

    # 3) Lateral direction (perpendicular, right-handed)
    lateral = np.array([-v_hat[1], v_hat[0]], dtype=float)

    # 4) Target release point (2D then extend to 3D)
    speed = float(np.linalg.norm(vel_uav[:2])) if norm_xy >= 1e-9 else 0.0
    p_release_2d = (
        pos_uav[:2]
        + v_hat * speed * float(entry.optimal_release_time)
        + lateral * float(entry.offset)
    )
    p_release = np.zeros(3, dtype=float)
    p_release[:2] = p_release_2d
    p_release[2] = pos_uav[2]

    # 5) Guidance vector and heading error
    g = p_release[:2] - pos_uav[:2]
    distance = float(np.linalg.norm(g))
    if distance < 1e-9:
        heading_target = float(np.arctan2(v_hat[1], v_hat[0]))
    else:
        heading_target = float(np.arctan2(g[1], g[0]))
    heading_current = float(np.arctan2(v_hat[1], v_hat[0]))
    heading_error = _wrap_angle(heading_target - heading_current)

    # 6) Status
    optimal_p_hit = float(getattr(entry, "optimal_p_hit", 0.0))
    time_to_release = float(entry.optimal_release_time)
    distance_to_corridor = abs(float(entry.offset))

    feasible_offsets = getattr(envelope_result, "feasible_offsets", None) or []
    if len(feasible_offsets) == 0:
        return GuidanceResult(
            status="NO_DROP",
            target_release_point=np.zeros(3),
            guidance_vector=np.zeros(2),
            heading_error=0.0,
            distance_to_corridor=0.0,
            time_to_release=0.0,
            optimal_p_hit=0.0,
        )
    corridor_half_width = max(abs(o) for o in feasible_offsets)
    lateral_distance = distance_to_corridor

    if optimal_p_hit < threshold:
        status = "NO_DROP"
    elif lateral_distance > corridor_half_width:
        status = "APPROACH_CORRIDOR"
    elif abs(time_to_release) < drop_now_time_threshold:
        status = "DROP_NOW"
    else:
        status = "IN_DROP_ZONE"

    return GuidanceResult(
        status=status,
        target_release_point=p_release,
        guidance_vector=np.asarray(g, dtype=float).reshape(2),
        heading_error=heading_error,
        distance_to_corridor=distance_to_corridor,
        time_to_release=time_to_release,
        optimal_p_hit=optimal_p_hit,
    )
