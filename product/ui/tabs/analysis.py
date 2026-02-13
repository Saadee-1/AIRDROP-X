"""
Analysis tab. Read-only display of analysis and sensitivity plots.
No sliders, no inputs, no recomputation. For understanding only.
"""

import numpy as np
import matplotlib.patches as mpatches

from product.ui import plots

# Import unified military-grade theme
from product.ui.ui_theme import (
    BG_MAIN,
    BG_PANEL,
    TEXT_PRIMARY,
    TEXT_LABEL,
    BORDER_SUBTLE,
)


def _placeholder_panel(
    ax,
    title,
    message="Sensitivity sweep not performed.\nUse Opportunity Analysis to generate sweep.",
):
    """Subdued placeholder when precomputed curve is not provided."""
    ax.set_axis_off()
    ax.set_facecolor(BG_PANEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(
        mpatches.Rectangle(
            (0.02, 0.02),
            0.96,
            0.96,
            linewidth=1,
            edgecolor=BORDER_SUBTLE,
            facecolor="none",
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.5,
        0.6,
        title,
        transform=ax.transAxes,
        fontsize=10,
        color=TEXT_LABEL,
        ha="center",
        va="center",
        family="monospace",
    )
    ax.text(
        0.5,
        0.4,
        message,
        transform=ax.transAxes,
        fontsize=8,
        color=TEXT_LABEL,
        ha="center",
        va="center",
        family="monospace",
    )


def _impact_dynamics_panel(
    ax, impact_velocity_stats, max_safe_impact_speed=None
):
    """IMPACT DYNAMICS panel: mean, std, p95 impact speed + engineering note."""
    ax.set_axis_off()
    ax.set_facecolor(BG_PANEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(
        mpatches.Rectangle(
            (0.02, 0.02),
            0.96,
            0.96,
            linewidth=1,
            edgecolor=BORDER_SUBTLE,
            facecolor="none",
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.5,
        0.92,
        "IMPACT DYNAMICS",
        transform=ax.transAxes,
        fontsize=10,
        color=TEXT_LABEL,
        ha="center",
        va="top",
        family="monospace",
    )
    y = 0.70
    dy = 0.22
    if impact_velocity_stats:
        ax.text(
            0.1,
            y,
            "Mean impact speed",
            transform=ax.transAxes,
            fontsize=9,
            color=TEXT_LABEL,
            va="center",
            family="monospace",
        )
        ax.text(
            0.9,
            y,
            f"{impact_velocity_stats.get('mean_impact_speed', 0):.2f} m/s",
            transform=ax.transAxes,
            fontsize=9,
            color=TEXT_PRIMARY,
            ha="right",
            va="center",
            family="monospace",
        )
        y -= dy
        ax.text(
            0.1,
            y,
            "Std dev",
            transform=ax.transAxes,
            fontsize=9,
            color=TEXT_LABEL,
            va="center",
            family="monospace",
        )
        ax.text(
            0.9,
            y,
            f"{impact_velocity_stats.get('std_impact_speed', 0):.2f} m/s",
            transform=ax.transAxes,
            fontsize=9,
            color=TEXT_PRIMARY,
            ha="right",
            va="center",
            family="monospace",
        )
        y -= dy
        ax.text(
            0.1,
            y,
            "95% percentile",
            transform=ax.transAxes,
            fontsize=9,
            color=TEXT_LABEL,
            va="center",
            family="monospace",
        )
        ax.text(
            0.9,
            y,
            f"{impact_velocity_stats.get('p95_impact_speed', 0):.2f} m/s",
            transform=ax.transAxes,
            fontsize=9,
            color=TEXT_PRIMARY,
            ha="right",
            va="center",
            family="monospace",
        )
        y -= dy
    if max_safe_impact_speed is not None and impact_velocity_stats:
        p95 = float(impact_velocity_stats.get("p95_impact_speed", 0.0))
        safe = float(max_safe_impact_speed)
        ratio = p95 / max(safe, 1e-9)
        if ratio <= 0.80:
            risk = "LOW"
        elif ratio <= 1.00:
            risk = "MODERATE"
        else:
            risk = "HIGH"
        ax.text(
            0.1,
            y,
            "Survivability Risk",
            transform=ax.transAxes,
            fontsize=9,
            color=TEXT_LABEL,
            va="center",
            family="monospace",
        )
        ax.text(
            0.9,
            y,
            risk,
            transform=ax.transAxes,
            fontsize=9,
            color=TEXT_PRIMARY,
            ha="right",
            va="center",
            family="monospace",
        )
        y -= dy
    else:
        ax.text(
            0.5,
            max(y, 0.24),
            "No structural limit defined.",
            transform=ax.transAxes,
            fontsize=8,
            color=TEXT_LABEL,
            ha="center",
            va="center",
            family="monospace",
        )
    ax.text(
        0.5,
        0.12,
        "Impact survivability depends on payload structural limits.",
        transform=ax.transAxes,
        fontsize=7,
        color=TEXT_LABEL,
        ha="center",
        va="center",
        family="monospace",
        wrap=True,
    )


def _cep_summary_panel(ax, cep50, target_hit_percentage):
    """Read-only CEP and hit probability summary."""
    ax.set_axis_off()
    ax.set_facecolor(BG_PANEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(
        mpatches.Rectangle(
            (0.02, 0.02),
            0.96,
            0.96,
            linewidth=1,
            edgecolor=BORDER_SUBTLE,
            facecolor="none",
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.5,
        0.88,
        "CEP SUMMARY",
        transform=ax.transAxes,
        fontsize=10,
        color=TEXT_LABEL,
        ha="center",
        va="top",
        family="monospace",
    )
    y = 0.68
    dy = 0.14
    if cep50 is not None:
        ax.text(
            0.1,
            y,
            "CEP50",
            transform=ax.transAxes,
            fontsize=9,
            color=TEXT_LABEL,
            va="center",
            family="monospace",
        )
        ax.text(
            0.9,
            y,
            f"{float(cep50):.2f} m",
            transform=ax.transAxes,
            fontsize=9,
            color=TEXT_PRIMARY,
            ha="right",
            va="center",
            family="monospace",
        )
        y -= dy
    if target_hit_percentage is not None:
        ax.text(
            0.1,
            y,
            "Hit %",
            transform=ax.transAxes,
            fontsize=9,
            color=TEXT_LABEL,
            va="center",
            family="monospace",
        )
        ax.text(
            0.9,
            y,
            f"{float(target_hit_percentage):.1f}%",
            transform=ax.transAxes,
            fontsize=9,
            color=TEXT_PRIMARY,
            ha="right",
            va="center",
            family="monospace",
        )
        y -= dy
    ax.text(
        0.5,
        0.22,
        "Read-only · No recomputation",
        transform=ax.transAxes,
        fontsize=7,
        color=TEXT_LABEL,
        ha="center",
        va="center",
        family="monospace",
    )


def render(
    ax,
    impact_points=None,
    target_position=None,
    target_radius=None,
    uav_position=None,
    wind_mean=None,
    cep50=None,
    target_hit_percentage=None,
    prob_vs_distance=None,
    prob_vs_wind_uncertainty=None,
    impact_velocity_stats=None,
    max_safe_impact_speed=None,
    **_,
):
    """
    Draw Analysis tab. All arguments optional. Read-only; no recomputation.
    - prob_vs_distance: (x_values, y_values) for P_hit vs target distance
    - prob_vs_wind_uncertainty: (x_values, y_values) for P_hit vs wind uncertainty
    """
    ax.set_axis_off()
    ax.set_facecolor(BG_MAIN)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 2x2 + IMPACT DYNAMICS: top row (prob/distance, impact dispersion), bottom row (wind, CEP, IMPACT DYNAMICS)
    margin = 0.02
    w_top = 0.46
    w_bot = 0.30
    h_bottom = 0.44
    h_top = 0.38
    tl = [margin, 1.0 - margin - h_top, w_top, h_top]
    tr = [margin + w_top + 0.02, 1.0 - margin - h_top, w_top, h_top]
    bl = [margin, margin, w_bot, h_bottom]
    bm = [margin + w_bot + 0.02, margin, w_bot, h_bottom]
    br = [margin + 2 * w_bot + 0.04, margin, w_bot, h_bottom]

    # Top-left: Probability vs target distance
    ax_tl = ax.inset_axes(tl)
    ax_tl.set_clip_on(True)
    if prob_vs_distance is not None and len(prob_vs_distance) == 2:
        x_vals, y_vals = prob_vs_distance[0], prob_vs_distance[1]
        if len(x_vals) and len(y_vals):
            plots.plot_sensitivity(
                ax_tl,
                x_vals,
                y_vals,
                "Target distance (m)",
                "Hit probability",
                title="P hit vs target distance",
            )
        else:
            _placeholder_panel(ax_tl, "P hit vs target distance")
    else:
        _placeholder_panel(ax_tl, "P hit vs target distance")

    # Top-right: Impact dispersion
    ax_tr = ax.inset_axes(tr)
    ax_tr.set_clip_on(True)
    if (
        impact_points is not None
        and target_position is not None
        and np.size(impact_points) >= 2
    ):
        impact_points = np.asarray(impact_points, dtype=float)
        if impact_points.ndim == 1:
            impact_points = impact_points.reshape(-1, 2)
        plots.plot_impact_dispersion(
            ax_tr,
            impact_points,
            target_position,
            target_radius or 0,
            cep50,
            release_point=(
                uav_position[:2] if uav_position is not None else None
            ),
            wind_vector=(wind_mean[:2] if wind_mean is not None else None),
        )
        ax_tr.set_title(
            "Impact dispersion",
            color=TEXT_LABEL,
            fontsize=9,
            family="monospace",
        )
    else:
        _placeholder_panel(ax_tr, "Impact dispersion", "No impact data")

    # Bottom-left: Probability vs wind uncertainty
    ax_bl = ax.inset_axes(bl)
    ax_bl.set_clip_on(True)
    if (
        prob_vs_wind_uncertainty is not None
        and len(prob_vs_wind_uncertainty) == 2
    ):
        x_vals, y_vals = (
            prob_vs_wind_uncertainty[0],
            prob_vs_wind_uncertainty[1],
        )
        if len(x_vals) and len(y_vals):
            plots.plot_sensitivity(
                ax_bl,
                x_vals,
                y_vals,
                "Wind uncertainty (m/s)",
                "Hit probability",
                title="P hit vs wind uncertainty",
            )
        else:
            _placeholder_panel(ax_bl, "P hit vs wind uncertainty")
    else:
        _placeholder_panel(ax_bl, "P hit vs wind uncertainty")

    # Bottom-center: CEP summary
    ax_bm = ax.inset_axes(bm)
    ax_bm.set_clip_on(True)
    _cep_summary_panel(ax_bm, cep50, target_hit_percentage)

    # Bottom-right: IMPACT DYNAMICS
    ax_br = ax.inset_axes(br)
    ax_br.set_clip_on(True)
    _impact_dynamics_panel(
        ax_br,
        impact_velocity_stats,
        max_safe_impact_speed=max_safe_impact_speed,
    )
