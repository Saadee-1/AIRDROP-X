"""
Stress-test for release envelope solver under extreme scenarios.

Scenarios:
  1) Strong wind
  2) Strong gust turbulence (via hybrid MC with gust model)
  3) UAV acceleration (nonzero forward acceleration)
  4) UAV turning (lateral velocity / acceleration)
  5) Small target radius
  6) Large target radius

For each scenario, prints:
  - optimal release time
  - optimal P_hit
  - corridor width
  - computation time
and verifies that no NaNs or invalid probabilities appear.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    """Configuration object for UT-based release planning and hybrid MC."""

    # Envelope / release-time grid.
    # Reduced grid for faster stress-test runs.
    max_lateral_offset: float = 20.0
    offset_step: float = 4.0
    drop_probability_threshold: float = 0.5
    compute_heatmap: bool = False
    max_release_time: float = 3.0
    release_time_step: float = 0.15
    target_radius: float = 15.0
    release_delay: float = 0.1

    # UT uncertainty parameters.
    wind_sigma0: float = 0.8
    wind_sigma_altitude_coeff: float = 0.001
    wind_sigma_max: float = 4.0
    release_pos_sigma: float = 0.5
    velocity_sigma: float = 0.02

    # Hybrid MC parameters (disabled for stress test to keep runtime bounded).
    enable_hybrid_estimation: bool = False
    wind_std: float = 0.8
    random_seed: int = 42
    n_samples: int = 500
    max_mc_verifications: int = 10

    # Gust turbulence model (used by MC when enabled).
    enable_gust_model: bool = False
    gust_theta: float = 1.5
    gust_sigma: float = 1.0


def _corridor_width(result) -> float:
    """Return corridor width from ReleaseEnvelopeResult."""
    if not result.feasible_offsets:
        return 0.0
    offsets = np.array(result.feasible_offsets, dtype=float)
    return float(offsets.max() - offsets.min())


def _check_probabilities(result) -> bool:
    """Verify there are no NaNs or invalid probabilities."""
    ok = True
    for e in result.envelope:
        p_vals = [
            float(e.optimal_p_hit),
            float(e.smoothed_p_hit),
        ]
        if getattr(e, "optimal_p_hit_ut", None) is not None:
            p_vals.append(float(e.optimal_p_hit_ut))
        if getattr(e, "optimal_p_hit_mc", None) is not None:
            p_vals.append(float(e.optimal_p_hit_mc))

        for p in p_vals:
            if not np.isfinite(p) or p < 0.0 or p > 1.0:
                ok = False
                break
        if not ok:
            break
    return ok


def _run_scenario(
    name: str,
    *,
    wind_mean: np.ndarray,
    wind_std: float,
    cfg_overrides: Optional[dict] = None,
    pos0: Optional[np.ndarray] = None,
    vel0: Optional[np.ndarray] = None,
    target_radius: float = 15.0,
    motion_predictor: Optional[MotionPredictor] = None,
    enable_gust: bool = False,
):
    mass = 1.0
    Cd = 1.0
    area = 0.01
    # Slightly larger dt for speed; this is a stress test, not a precision benchmark.
    dt = 0.02
    target_pos = np.array([72.0, 0.0, 0.0])

    if pos0 is None:
        pos0 = np.array([0.0, 0.0, 300.0])
    if vel0 is None:
        vel0 = np.array([25.0, 0.0, 0.0])

    context = build_propagation_context(
        mass=mass,
        Cd=Cd,
        area=area,
        wind_ref=np.asarray(wind_mean, dtype=float).reshape(3),
        shear=None,
        target_z=0.0,
        dt=dt,
    )

    cfg = _Config()
    cfg.wind_sigma0 = float(wind_std)
    cfg.wind_std = float(wind_std)
    cfg.target_radius = float(target_radius)
    cfg.enable_gust_model = bool(enable_gust)

    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)

    t0 = time.perf_counter()
    with _suppress_engine_prints():
        result = compute_release_envelope(
            context,
            cfg,
            pos0,
            vel0,
            target_pos,
            motion_predictor=motion_predictor,
        )
    elapsed = (time.perf_counter() - t0) * 1000.0  # ms

    corr_width = _corridor_width(result)
    ok_probs = _check_probabilities(result)

    # Best envelope entry by hit probability.
    best_entry = max(result.envelope, key=lambda e: e.optimal_p_hit)

    print(f"Scenario: {name}")
    print(f"  optimal_release_time = {best_entry.optimal_release_time:.3f} s")
    print(f"  optimal_P_hit        = {best_entry.optimal_p_hit:.3f}")
    print(f"  corridor_width       = {corr_width:.3f} m")
    print(f"  computation_time     = {elapsed:.1f} ms")
    print(f"  probabilities_valid  = {ok_probs}")
    print()


def main():
    print("Release planner stress test\n")

    # 1) Strong wind.
    _run_scenario(
        "Strong wind",
        wind_mean=np.array([8.0, 3.0, 0.0]),
        wind_std=2.0,
    )

    # 2) Strong gust turbulence (MC gust model via hybrid).
    _run_scenario(
        "Strong gust turbulence",
        wind_mean=np.array([4.0, 0.0, 0.0]),
        wind_std=1.5,
        cfg_overrides={
            "enable_gust_model": True,
            "gust_theta": 1.5,
            "gust_sigma": 2.0,
        },
        enable_gust=True,
    )

    # 3) UAV acceleration (forward acceleration).
    vs_accel = VehicleState(
        position=np.array([0.0, 0.0, 300.0]),
        velocity=np.array([20.0, 0.0, 0.0]),
        acceleration=np.array([2.0, 0.0, 0.0]),
        timestamp=0.0,
    )
    mp_accel = MotionPredictor(vs_accel)
    _run_scenario(
        "UAV acceleration",
        wind_mean=np.array([2.0, 0.0, 0.0]),
        wind_std=0.8,
        motion_predictor=mp_accel,
    )

    # 4) UAV turning (lateral velocity + lateral acceleration).
    vs_turn = VehicleState(
        position=np.array([0.0, 0.0, 300.0]),
        velocity=np.array([20.0, 5.0, 0.0]),
        acceleration=np.array([0.0, 1.5, 0.0]),
        timestamp=0.0,
    )
    mp_turn = MotionPredictor(vs_turn)
    _run_scenario(
        "UAV turning",
        wind_mean=np.array([2.0, -1.0, 0.0]),
        wind_std=0.8,
        motion_predictor=mp_turn,
    )

    # 5) Small target radius (hard precision requirement).
    _run_scenario(
        "Small target radius",
        wind_mean=np.array([2.0, 0.0, 0.0]),
        wind_std=0.8,
        target_radius=3.0,
    )

    # 6) Large target radius (easy target).
    _run_scenario(
        "Large target radius",
        wind_mean=np.array([2.0, 0.0, 0.0]),
        wind_std=0.8,
        target_radius=50.0,
    )


if __name__ == "__main__":
    main()

