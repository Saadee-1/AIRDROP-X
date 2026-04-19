"""
Telemetry health checks for SCYTHE.

This module provides advisory health checks for telemetry sources. These checks
are **informational only** and do NOT block simulation or disable controls.

The operator is always the final authority; this system advises, it does not enforce.
"""

from __future__ import annotations

from typing import Optional, List

from product.integrations.state_buffer import StateBuffer


def check_telemetry_health(
    buffer: Optional[StateBuffer],
    stale_threshold_seconds: float = 5.0,
    min_update_rate_hz: float = 1.0,
) -> List[str]:
    """
    Perform advisory health checks on telemetry StateBuffer.

    Parameters
    ----------
    buffer : StateBuffer, optional
        Telemetry buffer to check. If None, returns a warning that no
        telemetry source is configured.
    stale_threshold_seconds : float, optional
        Maximum acceptable age (seconds) before telemetry is considered stale.
        Default 5.0 seconds.
    min_update_rate_hz : float, optional
        Minimum acceptable update rate (Hz) before a warning is raised.
        Default 1.0 Hz (one update per second minimum).

    Returns
    -------
    List[str]
        List of warning messages. Empty list means no health issues detected.
        Warnings are textual and explicit; they do NOT block simulation.

    Notes
    -----
    This function is **advisory only**. It does not:
        - Block simulation runs
        - Disable UI controls
        - Modify telemetry values
        - Perform any physics or decision logic
    """
    warnings: List[str] = []

    if buffer is None:
        warnings.append("No telemetry source configured. Simulations will use config defaults.")
        return warnings

    latest = buffer.get_latest()
    if latest is None:
        warnings.append("No telemetry received yet. Simulations will use config defaults for UAV state.")
        return warnings

    # Check for stale telemetry.
    if buffer.is_stale(max_age_seconds=stale_threshold_seconds):
        age_seconds = buffer.get_age_seconds()
        age_str = f"{age_seconds:.1f}" if age_seconds is not None else "unknown"
        warnings.append(
            f"Telemetry is stale (age: {age_str} s, threshold: {stale_threshold_seconds:.1f} s). "
            f"Latest frame source: {latest.source}. Simulations may use outdated UAV state."
        )

    # Check update rate.
    rate_hz = buffer.estimate_update_rate_hz()
    if rate_hz is not None:
        if rate_hz < min_update_rate_hz:
            warnings.append(
                f"Telemetry update rate is low ({rate_hz:.2f} Hz, minimum recommended: {min_update_rate_hz:.1f} Hz). "
                f"Consider checking telemetry source connection."
            )
    else:
        # Insufficient history to compute rate (only one update so far).
        warnings.append(
            "Telemetry update rate cannot be determined (insufficient history). "
            "Monitor for consistent updates."
        )

    return warnings
