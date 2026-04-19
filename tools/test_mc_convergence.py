"""
Monte Carlo convergence test for SCYTHE.

Runs Monte Carlo propagation at multiple sample sizes and compares
impact mean, covariance, and CEP50 against a high-sample baseline.

Also plots CEP vs sample count.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on sys.path when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from product.guidance.advisory_layer import build_propagation_context
from src.monte_carlo import run_monte_carlo
from src import metrics


def _run_mc(n_samples: int, seed: int):
    """
    Run a single Monte Carlo experiment and return impacts, mean, cov, CEP.
    Scenario is fixed so results are comparable across N.
    """
    pos0 = (0.0, 0.0, 300.0)
    vel0 = (25.0, 0.0, 0.0)
    mass, Cd, A = 1.0, 1.0, 0.01
    wind_mean, wind_std = (1.0, 0.0, 0.0), 0.8
    target_pos = (72.0, 0.0, 0.0)

    dt = 0.01
    ctx = build_propagation_context(mass, Cd, A, wind_mean, None, 0.0, dt)
    cfg = {"n_samples": n_samples}

    impacts = run_monte_carlo(ctx, pos0, vel0, wind_std, cfg, seed)
    impacts = np.asarray(impacts, dtype=float).reshape(-1, 2)

    impact_mean = impacts.mean(axis=0)
    diff = impacts - impact_mean[None, :]
    impact_cov = (diff.T @ diff) / max(impacts.shape[0] - 1, 1)
    cep = metrics.compute_cep50(impacts, target_pos)
    return impacts, impact_mean, impact_cov, float(cep)


def main():
    sample_sizes = [100, 300, 1000, 3000]
    seed = 42

    results = {}
    for n in sample_sizes:
        print(f"[CONV] Running Monte Carlo with N={n}...")
        _, mu, cov, cep = _run_mc(n, seed)
        results[n] = {"mean": mu, "cov": cov, "cep": cep}

    # Use largest-N run as baseline.
    n_ref = max(sample_sizes)
    mu_ref = results[n_ref]["mean"]
    cov_ref = results[n_ref]["cov"]
    cep_ref = results[n_ref]["cep"]

    print("\n=== Monte Carlo Convergence vs N ===")
    print(f"Reference N = {n_ref}")
    print(f"  mean_ref = {mu_ref}")
    print(f"  CEP_ref  = {cep_ref:.4f} m\n")

    header = (
        "N".ljust(8)
        + "|| d_mean (m)".ljust(16)
        + "d_cov_F (abs)".ljust(16)
        + "CEP".ljust(12)
        + "d_CEP (abs)"
    )
    print(header)
    print("-" * len(header))

    ceps = []
    for n in sample_sizes:
        mu = results[n]["mean"]
        cov = results[n]["cov"]
        cep = results[n]["cep"]

        d_mean = float(np.linalg.norm(mu - mu_ref))
        d_cov = float(np.linalg.norm(cov - cov_ref, ord="fro"))
        d_cep = abs(cep - cep_ref)
        ceps.append(cep)

        print(
            f"{str(n).ljust(8)}|| "
            f"{d_mean:>10.4f}    "
            f"{d_cov:>10.4f}    "
            f"{cep:>8.4f}   "
            f"{d_cep:>8.4f}"
        )

    # Plot CEP vs sample count.
    plt.figure(figsize=(6, 4))
    plt.plot(sample_sizes, ceps, "o-", label="CEP50")
    plt.xlabel("Sample count N")
    plt.ylabel("CEP50 (m)")
    plt.title("Monte Carlo CEP Convergence vs Sample Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xscale("log")
    plt.xticks(sample_sizes, labels=[str(n) for n in sample_sizes])
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

