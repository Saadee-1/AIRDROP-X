"""
Unscented Transform-based deterministic payload propagation.

This module constructs an uncertainty state over key environment and
release parameters, generates sigma points, and propagates each sigma
state through the existing RK2 payload integrator. The result is a
Gaussian approximation to the impact-point distribution.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from product.uncertainty import build_uncertainty_model, generate_sigma_points
from src.monte_carlo import _propagate_payload_batch


def propagate_unscented(
    context,
    config,
    pos0,
    vel0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate Unscented Transform sigma points through RK2 payload dynamics.

    Parameters
    ----------
    context :
        PropagationContext instance (immutable) defining physics and wind model.
    config :
        Configuration object providing UT uncertainty parameters.
    pos0 : array_like, shape (3,)
        Nominal initial position.
    vel0 : array_like, shape (3,)
        Nominal initial velocity.

    Returns
    -------
    impact_mean : (2,)
        Mean impact point (x, y) across all sigma trajectories.
    impact_cov : (2, 2)
        Impact-point covariance matrix.
    """
    # 1) Build Gaussian uncertainty model for UT state.
    mu, Sigma = build_uncertainty_model(context, config)

    # 2) Generate sigma points and weights.
    sigma_points, Wm, Wc = generate_sigma_points(mu, Sigma)
    # Ensure mean weights are numerically normalized (defensive).
    Wm = Wm / np.sum(Wm)

    pos0 = np.asarray(pos0, dtype=float).reshape(3)
    vel0 = np.asarray(vel0, dtype=float).reshape(3)

    # Base wind reference from context; ensure 3-vector.
    base_wind = np.asarray(context.wind_ref, dtype=float).reshape(-1)[:3]

    num_sigma = sigma_points.shape[0]

    # Batch all sigma points into a single _propagate_payload_batch call.
    # Previously: num_sigma sequential N=1 calls (GIL-bound Python loop).
    wind_bias = sigma_points[:, 0:2]
    release_err = sigma_points[:, 2:4]
    velocity_bias = sigma_points[:, 4]

    max_wind_bias = 6.0  # m/s
    wind_bias = max_wind_bias * np.tanh(wind_bias / max_wind_bias)

    max_release_error = 50.0  # meters
    release_err = np.clip(release_err, -max_release_error, max_release_error)

    wind_refs = np.broadcast_to(base_wind, (num_sigma, 3)).copy()
    wind_refs[:, 0] += wind_bias[:, 0]
    wind_refs[:, 1] += wind_bias[:, 1]

    pos_batch = np.broadcast_to(pos0, (num_sigma, 3)).copy()
    pos_batch[:, 0] += release_err[:, 0]
    pos_batch[:, 1] += release_err[:, 1]

    scale = np.maximum(1.0 + velocity_bias, 0.1)
    vel_batch = vel0[None, :] * scale[:, None]

    ctx_ut = context.with_wind(wind_refs)

    impact_xy, _ = _propagate_payload_batch(
        ctx_ut,
        pos0,
        vel0,
        return_trajectories=False,
        return_impact_speeds=False,
        pos0_batch=pos_batch,
        vel0_batch=vel_batch,
    )
    impacts = np.asarray(impact_xy, dtype=float).reshape(num_sigma, 2)

    # 4) Combine results into mean and covariance.
    # Mean impact point.
    impact_mean = np.sum(Wm[:, None] * impacts, axis=0)

    # Impact covariance: sum over weighted outer products.
    diff = impacts - impact_mean[None, :]
    impact_cov = np.zeros((2, 2), dtype=float)
    for i in range(num_sigma):
        v = diff[i]
        impact_cov += Wc[i] * np.outer(v, v)

    # Small diagonal stabilization to improve numerical robustness in
    # downstream eigen/square-root computations.
    impact_cov += 1e-12 * np.eye(2, dtype=float)

    # Guarantee positive definiteness (defensive): if numerical issues
    # produce a slightly negative eigenvalue, shift the diagonal.
    eigvals = np.linalg.eigvalsh(impact_cov)
    min_eig = float(np.min(eigvals))
    if min_eig < 0.0:
        impact_cov += (abs(min_eig) + 1e-9) * np.eye(2, dtype=float)

    return impact_mean, impact_cov

