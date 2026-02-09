"""
System Status tab. Display only: identity, reproducibility, limitations, warnings.
No engine or mission logic. For transparency, traceability, and auditability.
"""

import matplotlib.patches as mpatches

_BG = "#0a0c0a"
_PANEL = "#0f120f"
_LABEL = "#6b8e6b"
_TEXT = "#c0d0c0"
_BORDER = "#2a3a2a"
_ACCENT = "#00ff41"
_WARN = "#e6b800"


def _defaults():
    """Default display values when not passed from layout."""
    return {
        "system_name": "AIRDROP-X",
        "version": "1.0",
        "build_id": "—",
        "physics_description": "3-DOF point-mass; gravity -Z; quadratic drag (v_rel); explicit Euler.",
        "mc_description": "One wind sample per trajectory; Gaussian uncertainty; fixed seed.",
        "random_seed": None,
        "n_samples": None,
        "uncertainty_model": "Gaussian wind (mean + std), one sample per trajectory.",
        "dt": None,
        "snapshot_created_at": None,
        "limitations": [
            "Point-mass payload; no rotation or attitude dynamics.",
            "2D target (X–Y); impact at Z=0 only.",
            "Uniform atmosphere (constant rho); no wind shear or gusts.",
            "Wind uncertainty: isotropic Gaussian; no spatial/temporal correlation.",
            "Single release point per run; no multi-drop or sequencing.",
        ],
        "warnings": ["No active warnings."],
    }


def _fmt(v):
    if v is None:
        return "—"
    return str(v)


def render(ax, **kwargs):
    """
    Draw System Status tab. Display only; no interactivity.
    kwargs override defaults (e.g. random_seed, n_samples, dt from layout).
    """
    d = _defaults()
    d.update({k: v for k, v in kwargs.items() if k in d})

    ax.set_axis_off()
    ax.set_facecolor(_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Four sections: enough line height so text never overlaps
    panel_left = 0.03
    panel_width = 0.94
    row_h = 0.23
    gap = 0.018
    y_start = 0.98

    def section(title, y_curr, content_lines, line_height=0.12, title_color=None):
        box = ax.inset_axes([panel_left, y_curr - row_h, panel_width, row_h])
        box.set_axis_off()
        box.set_facecolor(_PANEL)
        box.set_xlim(0, 1)
        box.set_ylim(0, 1)
        box.add_patch(mpatches.Rectangle((0.02, 0.02), 0.96, 0.96, linewidth=1,
                    edgecolor=_BORDER, facecolor="none", transform=box.transAxes))
        box.text(0.5, 0.96, title, transform=box.transAxes, fontsize=9,
                 color=title_color or _ACCENT, ha="center", va="top", family="monospace")
        yy = 0.80
        for line in content_lines:
            if yy < 0.05:
                break
            box.text(0.04, yy, line[:95], transform=box.transAxes, fontsize=8,
                     color=_TEXT, va="top", family="monospace")
            yy -= line_height
        return y_curr - row_h - gap

    y = y_start

    # 1. Engine Identity
    identity_lines = [
        f"{d['system_name']}  ·  v{d['version']}  ·  build {d['build_id']}",
        "Physics: " + (d["physics_description"] or "—"),
        "Monte Carlo: " + (d["mc_description"] or "—"),
        "Config source: configs.mission_configs (ASSUMED)",
    ]
    y = section("ENGINE IDENTITY", y, identity_lines, line_height=0.14)

    # 2. Reproducibility
    created = d.get("snapshot_created_at")
    created_str = _fmt(created)
    repro_lines = [
        f"Random seed: {_fmt(d['random_seed'])}   Sample count: {_fmt(d['n_samples'])}   Time step: {_fmt(d['dt'])} s",
        "Uncertainty: " + (d["uncertainty_model"] or "—"),
        f"Snapshot created at: {created_str}",
    ]
    y = section("REPRODUCIBILITY", y, repro_lines, line_height=0.14)

    # 3. Limitations & Assumptions
    lim_lines = ["Known modeling limitations (no hiding):"] + [
        "  · " + item for item in (d.get("limitations") or [])
    ]
    y = section("LIMITATIONS & ASSUMPTIONS", y, lim_lines, line_height=0.11)

    # 4. Warnings / Status Flags
    warnings = d.get("warnings") or []
    if not warnings:
        warnings = ["No active warnings."]
    warn_lines = [("  ⚠ " if i == 0 else "  · ") + w for i, w in enumerate(warnings)]
    y = section("WARNINGS / STATUS", y, warn_lines, line_height=0.12, title_color=_WARN)
