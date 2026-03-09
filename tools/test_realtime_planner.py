"""
Realtime-style release planner test.

Simulates 20 s of UAV telemetry and, at each 0.1 s step:
  - Updates VehicleState and MotionPredictor.
  - Runs compute_release_envelope.
  - Records optimal release time, optimal P_hit, and corridor width.

Finally, plots how these values evolve over time.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import concurrent.futures as cf
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on sys.path when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from product.physics.propagation_context import build_propagation_context
from product.explorer import compute_release_envelope
from product.aircraft import VehicleState, MotionPredictor


@contextlib.contextmanager
def _suppress_engine_prints():
    """Suppress engine timing / debug prints for cleaner output."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


@dataclass
class _Config:
    """Configuration for release envelope computation."""

    # Envelope / release-time grid (kept very small for realtime loop speed).
    # Note: max_lateral_offset=0 collapses envelope to a single offset (0 m).
    max_lateral_offset: float = 0.0
    offset_step: float = 4.0
    drop_probability_threshold: float = 0.5
    compute_heatmap: bool = False
    max_release_time: float = 2.0
    release_time_step: float = 2.0
    target_radius: float = 15.0
    release_delay: float = 0.1

    # UT uncertainty parameters.
    wind_sigma0: float = 0.8
    wind_sigma_altitude_coeff: float = 0.001
    wind_sigma_max: float = 4.0
    release_pos_sigma: float = 0.5
    velocity_sigma: float = 0.02

    # Hybrid MC disabled for realtime-style test (speed).
    enable_hybrid_estimation: bool = False
    wind_std: float = 0.8
    random_seed: int = 42
    n_samples: int = 500
    max_mc_verifications: int = 10


def _corridor_width(result) -> float:
    """Return corridor width from ReleaseEnvelopeResult."""
    if not result.feasible_offsets:
        return 0.0
    offsets = np.array(result.feasible_offsets, dtype=float)
    return float(offsets.max() - offsets.min())


def _simulate_trajectory(total_time: float, dt_telemetry: float) -> List[VehicleState]:
    """
    Generate a list of VehicleState samples for 0..total_time with step dt_telemetry.

    0–10 s: straight flight (constant heading).
    10–20 s: coordinated turn at constant speed.
    """
    states: List[VehicleState] = []

    # Initial conditions.
    # Lower altitude to keep planning loop fast.
    z = 80.0
    speed = 25.0  # m/s
    heading = 0.0  # radians
    omega = 0.12  # rad/s yaw rate during turn (~7 deg/s)

    x = 0.0
    y = 0.0

    t = 0.0
    n_steps = int(total_time / dt_telemetry) + 1

    for k in range(n_steps):
        if t < 10.0:
            # Straight flight: heading fixed, no lateral acceleration.
            heading = 0.0
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)
            ax = 0.0
            ay = 0.0
        else:
            # Turning flight: constant speed, nonzero yaw rate.
            heading = heading + omega * dt_telemetry
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)
            # Centripetal acceleration at constant speed v: a = v * omega (perpendicular to velocity).
            ax = -speed * omega * np.sin(heading)
            ay = speed * omega * np.cos(heading)

        # Position update using simple Euler step on previous velocity.
        if k > 0:
            x += vx * dt_telemetry
            y += vy * dt_telemetry

        pos = np.array([x, y, z], dtype=float)
        vel = np.array([vx, vy, 0.0], dtype=float)
        acc = np.array([ax, ay, 0.0], dtype=float)

        states.append(
            VehicleState(
                position=pos,
                velocity=vel,
                acceleration=acc,
                timestamp=t,
            )
        )

        t += dt_telemetry

    return states


def _compute_envelope_metrics_for_state(vs: VehicleState):
    """
    Worker: build MotionPredictor, run compute_release_envelope, and return metrics.
    Defined at module scope for Windows multiprocessing pickling.
    """
    # Scenario physics.
    mass = 1.0
    Cd = 1.0
    area = 0.01
    wind_mean = np.array([2.0, 0.0, 0.0])
    dt_physics = 0.25

    context = build_propagation_context(
        mass=mass,
        Cd=Cd,
        area=area,
        wind_ref=wind_mean.reshape(3),
        shear=None,
        target_z=0.0,
        dt=dt_physics,
    )

    target_pos = np.array([72.0, 0.0, 0.0])
    cfg = _Config()
    cfg.wind_sigma0 = cfg.wind_std

    mp = MotionPredictor(vs)
    t0 = time.perf_counter()
    with _suppress_engine_prints():
        result = compute_release_envelope(
            context,
            cfg,
            vs.position,
            vs.velocity,
            target_pos,
            motion_predictor=mp,
        )
    step_ms = (time.perf_counter() - t0) * 1000.0

    best_entry = max(result.envelope, key=lambda e: e.optimal_p_hit)
    corr_width = _corridor_width(result)

    return (
        float(vs.timestamp),
        float(best_entry.optimal_release_time),
        float(best_entry.optimal_p_hit),
        float(corr_width),
        float(step_ms),
    )


def main():
    total_time = 20.0  # seconds
    dt_telemetry = 0.1

    states = _simulate_trajectory(total_time, dt_telemetry)

    times: List[float] = []
    opt_release_times: List[float] = []
    opt_p_hits: List[float] = []
    corridor_widths: List[float] = []
    step_times_ms: List[float] = []

    print("Realtime planner test (20 s trajectory)\n", flush=True)

    # Parallelize per-timestep planning (each step independent). This keeps the
    # experiment faithful while avoiding very long runtimes.
    max_workers = max(1, min(6, (os.cpu_count() or 2) - 1))
    results = []
    with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_compute_envelope_metrics_for_state, vs) for vs in states]
        completed = 0
        for fut in cf.as_completed(futures):
            results.append(fut.result())
            completed += 1
            if completed % 25 == 0:
                print(f"[progress] {completed}/{len(states)} steps", flush=True)

    results.sort(key=lambda r: r[0])
    for t_s, t_rel, p_hit, w, dt_ms in results:
        times.append(t_s)
        opt_release_times.append(t_rel)
        opt_p_hits.append(p_hit)
        corridor_widths.append(w)
        step_times_ms.append(dt_ms)

    # Print simple summary.
    print("t (s)\topt_t_rel (s)\topt_P_hit\tcorr_width (m)\tstep_ms")
    for t, tr, p, w, dt_ms in zip(
        times[::5], opt_release_times[::5], opt_p_hits[::5], corridor_widths[::5], step_times_ms[::5]
    ):
        print(f"{t:5.1f}\t{tr:10.3f}\t{p:9.3f}\t{w:12.3f}\t{dt_ms:7.1f}")

    # Plot evolution over time.
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(times, opt_release_times, "b-")
    axes[0].set_ylabel("Optimal release\nlead time (s)")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(times, opt_p_hits, "g-")
    axes[1].set_ylabel("Optimal P_hit")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].plot(times, corridor_widths, "r-")
    axes[2].set_ylabel("Corridor width (m)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

