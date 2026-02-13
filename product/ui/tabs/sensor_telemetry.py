"""
Sensor & Telemetry tab. Display only: measured and derived data with clear labels.
No simulation, estimation, or filtering. Military-grade clarity.
"""

import matplotlib.patches as mpatches

# Import unified military-grade theme
from product.ui.ui_theme import (
    BG_MAIN,
    BG_PANEL,
    TEXT_PRIMARY,
    TEXT_LABEL,
    ACCENT_GO,
    ACCENT_NO_GO,
    ACCENT_WARN,
    BORDER_SUBTLE,
)

# Legacy aliases for backward compat (mapped to theme)
_BG = BG_MAIN
_PANEL = BG_PANEL
_ACCENT = ACCENT_GO
_RED = ACCENT_NO_GO
_AMBER = ACCENT_WARN
_LABEL = TEXT_LABEL
_TEXT = TEXT_PRIMARY
_BORDER = BORDER_SUBTLE
_MEASURED = TEXT_LABEL  # Light green for measured
_ESTIMATED = ACCENT_WARN  # Amber for estimated


def _default_telemetry():
    """Display-only default data. Replace with passed dict when wired from engine."""
    return {
        "gnss_lat": "Awaiting telemetry input",
        "gnss_lon": "Awaiting telemetry input",
        "gnss_speed_ms": "Awaiting telemetry input",
        "gnss_heading_deg": "Awaiting telemetry input",
        "gnss_altitude_m": "Awaiting telemetry input",
        "gnss_fix": "No Fix",
        "gnss_freshness_s": None,
        "roll_deg": None,
        "pitch_deg": None,
        "yaw_deg": None,
        "accel_mag_ms2": None,
        "imu_health": "—",
        "wind_dir_deg": "Awaiting telemetry input",
        "wind_speed_ms": "Awaiting telemetry input",
        "wind_uncertainty": None,
        "wind_source": "—",
        "wind_confidence": "—",
        "wind_mean_ms": None,
        "wind_std_dev_ms": None,
        "telemetry_live": False,
    }


def _fmt(v, fmt_str="{:.2f}", unit=""):
    if v is None:
        return "—"
    try:
        return fmt_str.format(float(v)) + unit
    except (TypeError, ValueError):
        return "—"


def _status_color(status):
    if status is None:
        return _TEXT
    s = str(status).upper()
    if "OK" in s or "3D" in s or "FIX" in s and "NO" not in s:
        return _ACCENT
    if "DEGRADED" in s or "2D" in s:
        return _AMBER
    if "FAULT" in s or "NO FIX" in s or "NONE" in s:
        return _RED
    return _TEXT


def render(ax, **kwargs):
    """
    Draw Sensor & Telemetry tab. Uses kwargs to override defaults; otherwise
    displays placeholder values. Display only; no computation.
    """
    data = _default_telemetry()
    data.update({k: v for k, v in kwargs.items() if k in data})

    ax.set_axis_off()
    ax.set_facecolor(_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Snapshot Mode Badge
    ax.text(
        0.98,
        0.98,
        "SNAPSHOT MODE – NOT LIVE FEED",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=ACCENT_WARN,
        family="monospace",
        weight="bold",
    )

    # Panel layout: three columns
    # Left: Navigation State (Measured)
    # Center: Attitude & Motion (Measured)
    # Right: Environment (Derived)
    panel_w = 0.30
    gap = 0.02
    left = 0.04
    mid = left + panel_w + gap
    right = mid + panel_w + gap

    _draw_navigation_panel(ax.inset_axes([left, 0.08, panel_w, 0.84]), data)
    _draw_attitude_panel(ax.inset_axes([mid, 0.08, panel_w, 0.84]), data)
    _draw_environment_panel(ax.inset_axes([right, 0.08, panel_w, 0.84]), data)


def _draw_panel_frame(ax, title, subtitle=None):
    ax.set_axis_off()
    ax.set_facecolor(_PANEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for spine in ax.spines.values():
        spine.set_color(_BORDER)
    ax.add_patch(mpatches.Rectangle((0.02, 0.02), 0.96, 0.96, linewidth=1,
                 edgecolor=_BORDER, facecolor="none", transform=ax.transAxes))
    ax.text(0.5, 0.96, title, transform=ax.transAxes, fontsize=10,
            color=_ACCENT, ha="center", va="top", family="monospace")
    if subtitle:
        ax.text(0.5, 0.90, subtitle, transform=ax.transAxes, fontsize=7,
                color=_LABEL, ha="center", va="top", family="monospace")


def _draw_navigation_panel(ax, d):
    _draw_panel_frame(ax, "NAVIGATION STATE", "MEASURED")
    y = 0.82
    dy = 0.078

    def row(lbl, val, color=None):
        ax.text(
            0.06,
            y,
            lbl,
            transform=ax.transAxes,
            fontsize=8,
            color=_LABEL,
            va="center",
            family="monospace",
        )
        ax.text(
            0.94,
            y,
            val,
            transform=ax.transAxes,
            fontsize=8,
            color=color or _TEXT,
            ha="right",
            va="center",
            family="monospace",
        )

    lat = (
        _fmt(d["gnss_lat"], "{:.6f}", "°")
        if d["gnss_lat"] is not None
        else "—"
    )
    lon = (
        _fmt(d["gnss_lon"], "{:.6f}", "°")
        if d["gnss_lon"] is not None
        else "—"
    )
    row("Latitude", lat)
    y -= dy
    row("Longitude", lon)
    y -= dy
    row("Speed", _fmt(d["gnss_speed_ms"], "{:.1f}", " m/s"))
    y -= dy
    row("Heading", _fmt(d["gnss_heading_deg"], "{:.0f}", "°"))
    y -= dy
    row("Altitude", _fmt(d["gnss_altitude_m"], "{:.0f}", " m"))
    y -= dy
    fix = d.get("gnss_fix") or "—"
    row("Fix", fix, _status_color(fix))
    y -= dy
    fresh = d.get("gnss_freshness_s")
    freshness = _fmt(fresh, "{:.2f}", " s ago") if fresh is not None else "—"
    row("Freshness", freshness)


def _draw_attitude_panel(ax, d):
    _draw_panel_frame(ax, "ATTITUDE & MOTION", "MEASURED")
    y = 0.82
    dy = 0.078

    def row(lbl, val, color=None):
        ax.text(
            0.06,
            y,
            lbl,
            transform=ax.transAxes,
            fontsize=8,
            color=_LABEL,
            va="center",
            family="monospace",
        )
        ax.text(
            0.94,
            y,
            val,
            transform=ax.transAxes,
            fontsize=8,
            color=color or _TEXT,
            ha="right",
            va="center",
            family="monospace",
        )

    row("Roll", _fmt(d["roll_deg"], "{:.1f}", "°"))
    y -= dy
    row("Pitch", _fmt(d["pitch_deg"], "{:.1f}", "°"))
    y -= dy
    row("Yaw", _fmt(d["yaw_deg"], "{:.1f}", "°"))
    y -= dy
    row("|a|", _fmt(d["accel_mag_ms2"], "{:.2f}", " m/s²"))
    y -= dy
    imu = d.get("imu_health") or "—"
    row("IMU", imu, _status_color(imu))


def _draw_environment_panel(ax, d):
    _draw_panel_frame(ax, "ENVIRONMENT", "DERIVED · ESTIMATED")
    y = 0.82
    dy = 0.078

    def row(lbl, val, color=None):
        ax.text(
            0.06,
            y,
            lbl,
            transform=ax.transAxes,
            fontsize=8,
            color=_LABEL,
            va="center",
            family="monospace",
        )
        ax.text(
            0.94,
            y,
            val,
            transform=ax.transAxes,
            fontsize=8,
            color=color or _ESTIMATED,
            ha="right",
            va="center",
            family="monospace",
        )

    row("Wind direction", _fmt(d["wind_dir_deg"], "{:.0f}", "°"))
    y -= dy
    row("Wind speed", _fmt(d["wind_speed_ms"], "{:.1f}", " m/s"))
    y -= dy
    row("Wind mean", _fmt(d.get("wind_mean_ms"), "{:.2f}", " m/s"))
    y -= dy
    row("Wind std σ", _fmt(d.get("wind_std_dev_ms"), "{:.2f}", " m/s"))
    y -= dy
    u = d.get("wind_uncertainty")
    row("Uncertainty", _fmt(u, "{:.2f}", "") if u is not None else "—")
    y -= dy
    row("Source", (d.get("wind_source") or "—")[:14])
    y -= dy
    row("Confidence", (d.get("wind_confidence") or "—")[:12])
    y -= dy
    if not d.get("telemetry_live", False):
        ax.text(
            0.06,
            max(y, 0.05),
            "Using assumed Gaussian wind model.",
            transform=ax.transAxes,
            fontsize=7,
            color=_LABEL,
            va="center",
            family="monospace",
        )
