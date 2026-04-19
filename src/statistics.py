"""
Statistical utilities for SCYTHE decision layer.
Wilson score interval for binomial proportion confidence.
"""
from __future__ import annotations

import math


def compute_wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score confidence interval for binomial proportion.

    Args:
        k: Number of successes (hits).
        n: Number of trials (samples).
        z: Z-score for confidence level (default 1.96 = 95%).

    Returns:
        (ci_low, ci_high) both in [0, 1].
    """
    if n == 0:
        return (0.0, 1.0)
    p_hat = k / n
    z2 = z * z
    denom = n + z2
    center = (k + 0.5 * z2) / denom
    radicand = (k * (n - k) / n) + (z2 / 4.0)
    margin = (z / denom) * math.sqrt(max(0.0, radicand))
    ci_low = max(0.0, min(1.0, center - margin))
    ci_high = max(0.0, min(1.0, center + margin))
    return (ci_low, ci_high)
