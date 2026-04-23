"""
System Status tab. Display only: identity, reproducibility, limitations, warnings.
No engine or mission logic. For transparency, traceability, and auditability.
"""

import matplotlib.patches as mpatches

# Import unified military-grade theme
from product.ui.ui_theme import (
    BG_MAIN,
    BG_PANEL,
    TEXT_PRIMARY,
    ACCENT_GO,
    ACCENT_WARN,
    BORDER_SUBTLE,
)
from typing import Any, Dict, List


def _defaults() -> Dict[str, Any]:
    """Default display values when not passed from layout."""
    return {
        "system_name": "SCYTHE",
        "version": "1.1",
        "build_id": "--",
        "physics_description": (
            "3-DOF point-mass; gravity -Z; quadratic drag (v_rel); RK2 integrator."
        ),
        "mc_description": (
            "Hybrid UT + MC — Unscented Transform coarse scan (11 sigma points), "
            "Monte Carlo trajectory verification with Wilson CI early stopping."
        ),
        "random_seed": None,
        "n_samples": None,
        "uncertainty_model": (
            "Gaussian wind + release error; altitude-adaptive N sampling."
        ),
        "dt": None,
        "snapshot_created_at": None,
        "limitations": [
            "Point-mass payload; no rotation or attitude dynamics.",
            "2D target (X-Y); impact at Z=0 only.",
            "Uniform atmosphere (constant rho); no wind shear or gusts.",
            "Wind uncertainty: isotropic Gaussian; no spatial/temporal "
            "correlation.",
            "Single release point per run; no multi-drop or sequencing.",
            "Model Validity Envelope (Confidence > 90%):",
            "  - Altitude (AGL): 150 ft to 12,000 ft",
            "  - Release Velocity: 0 to 60 m/s (TAS)",
            "  - Payload Mass: 1 kg to 500 kg",
            "  - Descent Rate: < 50 m/s (subsonic)",
            "Confidence decreases primarily due to environmental uncertainty,",
            "specifically local wind field variance not captured by "
            "single-point sampling.",
        ],
        "warnings": ["No active warnings."],
    }


def _fmt(v: Any) -> str:
    if v is None:
        return "--"
    return str(v)


def render(ax, **kwargs):
    """
    Draw System Status tab. Display only; no interactivity.
    kwargs override defaults (e.g. random_seed, n_samples, dt from layout).
    """
    d = _defaults()
    d.update({k: v for k, v in kwargs.items() if k in d})

    ax.set_axis_off()
    ax.set_facecolor(BG_MAIN)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Five sections including numerical stability diagnostics (aligned spacing)
    panel_left = 0.03
    panel_width = 0.94
    row_h = 0.18
    gap = 0.015
    y_start = 0.98

    def section(
        title, y_curr, content_lines, line_height=0.12, title_color=None
    ):
        box = ax.inset_axes([panel_left, y_curr - row_h, panel_width, row_h])
        box.set_axis_off()
        box.set_facecolor(BG_PANEL)
        box.set_xlim(0, 1)
        box.set_ylim(0, 1)
        box.add_patch(
            mpatches.Rectangle(
                (0.02, 0.02),
                0.96,
                0.96,
                linewidth=1,
                edgecolor=BORDER_SUBTLE,
                facecolor="none",
                transform=box.transAxes,
            )
        )
        box.text(
            0.5,
            0.96,
            title,
            transform=box.transAxes,
            fontsize=9,
            color=title_color or ACCENT_GO,
            ha="center",
            va="top",
            family="monospace",
            weight="bold",
        )
        yy = 0.82
        for line in content_lines:
            if yy < 0.05:
                break
            box.text(
                0.04,
                yy,
                line[:95],
                transform=box.transAxes,
                fontsize=8,
                color=TEXT_PRIMARY,
                va="top",
                ha="left",
                family="monospace",
            )
            yy -= line_height
        return y_curr - row_h - gap

    y = y_start

    # 1. Engine Identity
    identity_lines = [
        f"{d['system_name']}  -  v{d['version']}  -  build {d['build_id']}",
        "Physics: " + str(d["physics_description"] or "--"),
        "Engine: " + str(d["mc_description"] or "--"),
        "Uncertainty: " + str(d["uncertainty_model"] or "--"),
        "Config source: Mission Configuration (operator-committed)",
    ]
    y = section("ENGINE IDENTITY", y, identity_lines, line_height=0.14)

    # 2. Reproducibility
    created = d.get("snapshot_created_at")
    created_str = _fmt(created)
    repro_lines = [
        (
            f"Random seed: {_fmt(d['random_seed'])}   "
            f"Sample count: {_fmt(d['n_samples'])}   "
            f"Time step: {_fmt(d['dt'])} s"
        ),
        "Uncertainty: " + str(d["uncertainty_model"] or "--"),
        f"Snapshot created at: {created_str}",
    ]
    y = section("REPRODUCIBILITY", y, repro_lines, line_height=0.14)

    # 3. Numerical Stability (display only; no engine/Monte Carlo in render path)
    stability_lines: List[str] = [
        "Stability diagnostics available after simulation run.",
    ]
    y = section("NUMERICAL STABILITY", y, stability_lines, line_height=0.12)

    # 4. Limitations & Assumptions
    lim_lines: List[str] = ["Known modeling limitations (no hiding):"] + [
        "  - " + item for item in (d.get("limitations") or [])
    ]
    y = section("LIMITATIONS & ASSUMPTIONS", y, lim_lines, line_height=0.11)

    # 5. Warnings / Status Flags
    warnings = d.get("warnings") or []
    if not warnings:
        warnings = ["No active warnings."]
    warn_lines = [
        ("  WARN: " if i == 0 else "  - ") + w for i, w in enumerate(warnings)
    ]
    y = section(
        "WARNINGS / STATUS",
        y,
        warn_lines,
        line_height=0.12,
        title_color=ACCENT_WARN,
    )
