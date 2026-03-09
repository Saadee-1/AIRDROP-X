"""
Deterministic physics engine sanity tests (no Monte Carlo).

Tests:
  1) Vacuum drop: drag and wind disabled; compare fall time to sqrt(2h/g).
  2) Crosswind drift: constant crosswind; compare lateral drift to wind * fall_time.
  3) Terminal velocity: drop with drag; verify convergence to theoretical v_terminal.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from product.physics.propagation_context import build_propagation_context
from src.monte_carlo import _propagate_payload_batch


GRAVITY = 9.81


def _run_single_trajectory(
    context,
    pos0,
    vel0,
    *,
    use_precise_impact: bool = True,
):
    """
    Helper to run a single deterministic trajectory with RK2 integrator.

    Returns (impact_xy, trajectory, fall_time, impact_speed).
    """
    pos0 = np.asarray(pos0, dtype=float).reshape(3)
    vel0 = np.asarray(vel0, dtype=float).reshape(3)

    impact_xy, trajectories_out, impact_speeds_out = _propagate_payload_batch(
        context,
        pos0,
        vel0,
        return_trajectories=True,
        return_impact_speeds=True,
        use_precise_impact=use_precise_impact,
    )

    traj = trajectories_out[0]
    dt = float(context.dt)

    # Reconstruct fall time using last two altitude samples and linear interpolation
    # to the ground plane (target_z).
    z = traj[:, 2]
    ground_z = float(context.target_z)

    # Find first index where z <= ground_z.
    idx = np.where(z <= ground_z)[0]
    if idx.size == 0 or idx[0] == 0:
        fall_time = dt * float(traj.shape[0] - 1)
    else:
        k = int(idx[0])
        z_prev = float(z[k - 1])
        z_curr = float(z[k])
        denom = z_prev - z_curr
        if denom > 0.0:
            alpha = (z_prev - ground_z) / denom
        else:
            alpha = 1.0
        alpha = float(np.clip(alpha, 0.0, 1.0))
        fall_time = dt * ((k - 1) + alpha)

    impact_speed = float(impact_speeds_out[0])
    return impact_xy[0], traj, fall_time, impact_speed


def test_vacuum_drop():
    print("=== Test 1: Vacuum drop ===")
    h = 500.0  # meters
    mass = 1.0
    Cd = 0.0   # disable drag
    area = 0.01
    wind_ref = np.zeros(3)
    dt = 0.01

    context = build_propagation_context(
        mass=mass,
        Cd=Cd,
        area=area,
        wind_ref=wind_ref.reshape(3),
        shear=None,
        target_z=0.0,
        dt=dt,
    )
    pos0 = np.array([0.0, 0.0, h])
    vel0 = np.array([0.0, 0.0, 0.0])

    _, _, fall_time_sim, _ = _run_single_trajectory(context, pos0, vel0)

    fall_time_analytical = math.sqrt(2.0 * h / GRAVITY)
    err_pct = abs(fall_time_sim - fall_time_analytical) / fall_time_analytical * 100.0

    print(f"Simulated fall time:   {fall_time_sim:.4f} s")
    print(f"Analytical fall time:  {fall_time_analytical:.4f} s")
    print(f"Percent error:         {err_pct:.3f} %")
    print()


def test_crosswind_drift():
    print("=== Test 2: Crosswind drift ===")
    h = 300.0
    mass = 1.0
    Cd = 1.0
    area = 0.05
    wind_speed = 5.0
    wind_ref = np.array([wind_speed, 0.0, 0.0])
    dt = 0.01

    context = build_propagation_context(
        mass=mass,
        Cd=Cd,
        area=area,
        wind_ref=wind_ref.reshape(3),
        shear=None,
        target_z=0.0,
        dt=dt,
    )
    pos0 = np.array([0.0, 0.0, h])
    # Start from rest to isolate wind-driven drift.
    vel0 = np.array([0.0, 0.0, 0.0])

    impact_xy, _, fall_time_sim, _ = _run_single_trajectory(context, pos0, vel0)

    drift_sim = float(impact_xy[0])
    drift_theory = wind_speed * fall_time_sim
    err_pct = abs(drift_sim - drift_theory) / drift_theory * 100.0

    print(f"Simulated lateral drift: {drift_sim:.3f} m")
    print(f"Theoretical drift:        {drift_theory:.3f} m")
    print(f"Percent error:            {err_pct:.3f} %")
    print()


def test_terminal_velocity():
    print("=== Test 3: Terminal velocity ===")
    h = 1000.0
    mass = 1.0
    Cd = 1.0
    area = 0.05
    wind_ref = np.zeros(3)
    dt = 0.01

    context = build_propagation_context(
        mass=mass,
        Cd=Cd,
        area=area,
        wind_ref=wind_ref.reshape(3),
        shear=None,
        target_z=0.0,
        dt=dt,
    )
    pos0 = np.array([0.0, 0.0, h])
    vel0 = np.array([0.0, 0.0, 0.0])

    _, traj, _, _ = _run_single_trajectory(context, pos0, vel0)

    dt_sim = float(context.dt)
    # Approximate velocity history from finite differences on trajectory.
    if traj.shape[0] >= 2:
        v_hist = np.diff(traj, axis=0) / dt_sim
        # Focus on the last 100 ms (or as many samples as available).
        n_tail = min(20, v_hist.shape[0])
        v_tail = v_hist[-n_tail:]
        speed_tail = np.linalg.norm(v_tail, axis=1)
        v_terminal_sim = float(np.mean(speed_tail))
    else:
        v_terminal_sim = 0.0

    # Theoretical terminal velocity using ground-level density as an approximation.
    rho0 = 1.225
    v_terminal_theory = math.sqrt(2.0 * mass * GRAVITY / (rho0 * Cd * area))
    err_pct = abs(v_terminal_sim - v_terminal_theory) / v_terminal_theory * 100.0

    print(f"Simulated terminal speed (tail avg): {v_terminal_sim:.3f} m/s")
    print(f"Theoretical terminal speed:          {v_terminal_theory:.3f} m/s")
    print(f"Percent error:                       {err_pct:.3f} %")
    print()


def main():
    test_vacuum_drop()
    test_crosswind_drift()
    test_terminal_velocity()


if __name__ == "__main__":
    main()

