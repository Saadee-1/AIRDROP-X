"""
UT vs Monte Carlo validation across multiple wind/uncertainty scenarios.

For each scenario, computes:
  - Monte Carlo mean impact, covariance, CEP50 (w.r.t. target).
  - Unscented Transform mean, covariance, and UT-derived CEP.

Reports mean difference, covariance difference, and CEP difference in a table.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from product.physics.propagation_context import build_propagation_context
from product.uncertainty.unscented_propagation import propagate_unscented
from src.monte_carlo import run_monte_carlo
from src import metrics


@contextlib.contextmanager
def _suppress_prints():
    """Suppress engine timing / debug prints for cleaner validation output."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


class _UTConfig:
    """Minimal config object carrying UT uncertainty parameters."""

    def __init__(
        self,
        wind_sigma0: float,
        wind_sigma_altitude_coeff: float,
        wind_sigma_max: float,
        release_pos_sigma: float,
        velocity_sigma: float,
    ):
        self.wind_sigma0 = wind_sigma0
        self.wind_sigma_altitude_coeff = wind_sigma_altitude_coeff
        self.wind_sigma_max = wind_sigma_max
        self.release_pos_sigma = release_pos_sigma
        self.velocity_sigma = velocity_sigma


def compute_cep_from_covariance(cov_2x2: np.ndarray) -> float:
    """
    CEP50 from a 2x2 Gaussian covariance matrix.

    Uses the Rayleigh approximation: CEP ≈ 1.1774 * sqrt(mean eigenvalue).
    """
    eigvals = np.linalg.eigvalsh(cov_2x2)
    eigvals = np.maximum(eigvals, 0.0)
    sigma_avg = float(np.sqrt(np.mean(eigvals)))
    return 1.1774 * sigma_avg


def run_ut_vs_mc_scenarios():
    # Common kinematics and target.
    pos0 = np.array([0.0, 0.0, 100.0])
    vel0 = np.array([20.0, 0.0, 0.0])
    mass = 1.0
    Cd = 1.0
    area = 0.01
    dt = 0.01
    target_pos = np.array([72.0, 0.0, 0.0])
    target_radius = 5.0

    # Scenario definitions: varying wind mean/std and UT uncertainty scales.
    scenarios = [
        {
            "name": "Low wind, low uncertainty",
            "wind_mean": np.array([1.0, 0.0, 0.0]),
            "wind_std": 0.3,
            "ut": dict(
                wind_sigma0=0.3,
                wind_sigma_altitude_coeff=0.0005,
                wind_sigma_max=2.0,
                release_pos_sigma=0.3,
                velocity_sigma=0.01,
            ),
        },
        {
            "name": "Moderate wind, nominal uncertainty",
            "wind_mean": np.array([2.0, 0.5, 0.0]),
            "wind_std": 0.8,
            "ut": dict(
                wind_sigma0=0.8,
                wind_sigma_altitude_coeff=0.001,
                wind_sigma_max=4.0,
                release_pos_sigma=0.5,
                velocity_sigma=0.02,
            ),
        },
        {
            "name": "Strong wind, high uncertainty",
            "wind_mean": np.array([4.0, -1.0, 0.0]),
            "wind_std": 1.5,
            "ut": dict(
                wind_sigma0=1.5,
                wind_sigma_altitude_coeff=0.0015,
                wind_sigma_max=6.0,
                release_pos_sigma=1.0,
                velocity_sigma=0.05,
            ),
        },
    ]

    n_mc = 3000
    seed = 42

    rows = []

    for sc in scenarios:
        name = sc["name"]
        wind_mean = sc["wind_mean"]
        wind_std = float(sc["wind_std"])

        context = build_propagation_context(
            mass=mass,
            Cd=Cd,
            area=area,
            wind_ref=wind_mean,
            shear=None,
            target_z=0.0,
            dt=dt,
        )

        # --- Monte Carlo ---
        mc_cfg = {"n_samples": n_mc}
        with _suppress_prints():
            mc_impacts = run_monte_carlo(context, pos0, vel0, wind_std, mc_cfg, seed)
        mc_impacts = np.asarray(mc_impacts, dtype=float).reshape(-1, 2)

        mc_mean = np.mean(mc_impacts, axis=0)
        mc_centered = mc_impacts - mc_mean[None, :]
        mc_cov = (mc_centered.T @ mc_centered) / float(mc_impacts.shape[0] - 1)
        mc_cep = metrics.compute_cep50(mc_impacts, target_pos)

        # --- Unscented Transform ---
        ut_cfg_params = sc["ut"]
        ut_config = _UTConfig(**ut_cfg_params)
        with _suppress_prints():
            ut_mean, ut_cov = propagate_unscented(context, ut_config, pos0, vel0)
        ut_cep = compute_cep_from_covariance(ut_cov)

        # --- Differences ---
        d_mean = float(np.linalg.norm(ut_mean - mc_mean))
        d_cov = float(np.linalg.norm(ut_cov - mc_cov, ord="fro"))
        d_cep = abs(ut_cep - mc_cep)

        rows.append(
            dict(
                name=name,
                mc_mean=mc_mean,
                ut_mean=ut_mean,
                d_mean=d_mean,
                d_cov=d_cov,
                mc_cep=mc_cep,
                ut_cep=ut_cep,
                d_cep=d_cep,
            )
        )

    # --- Summary table ---
    print("UT vs Monte Carlo Validation (multiple scenarios)")
    print("=" * 72)
    header = (
        "Scenario".ljust(28)
        + "|| d_mean (m)".ljust(16)
        + "d_cov_F".ljust(14)
        + "CEP_MC".ljust(12)
        + "CEP_UT".ljust(12)
        + "d_CEP"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['name'][:26].ljust(28)}|| "
            f"{r['d_mean']:>10.4f}    "
            f"{r['d_cov']:>8.4f}   "
            f"{r['mc_cep']:>8.3f}   "
            f"{r['ut_cep']:>8.3f}   "
            f"{r['d_cep']:>8.3f}"
        )

    print("\nDetails (first scenario example):")
    if rows:
        r0 = rows[0]
        print(f"  {r0['name']}:")
        print(f"    MC mean = ({r0['mc_mean'][0]:.3f}, {r0['mc_mean'][1]:.3f})")
        print(f"    UT mean = ({r0['ut_mean'][0]:.3f}, {r0['ut_mean'][1]:.3f})")
        print(f"    d_mean  = {r0['d_mean']:.4f} m")
        print(f"    CEP_MC  = {r0['mc_cep']:.3f} m")
        print(f"    CEP_UT  = {r0['ut_cep']:.3f} m")
        print(f"    d_CEP   = {r0['d_cep']:.3f} m")


def main():
    run_ut_vs_mc_scenarios()


if __name__ == "__main__":
    main()

