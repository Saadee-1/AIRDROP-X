"""
Mission Overview tab. Command-and-control: decision banner, key metrics, target view.
Receives precomputed data only; no computation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

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
    SCATTER_PRIMARY,
    CEP_CIRCLE,
    MEAN_MARKER,
)


def render(
    ax,
    decision,
    target_hit_percentage,
    cep50,
    threshold,
    mode,
    impact_points,
    target_position,
    target_radius,
    advisory_result=None,
    **kwargs
):
    ax.set_axis_off()
    ax.set_facecolor(BG_MAIN)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Left: decision banner + metrics + advisory
    left_ax = ax.inset_axes([0.02, 0.04, 0.36, 0.92])
    _draw_banner_metrics_advisory(
        left_ax,
        decision,
        target_hit_percentage,
        cep50,
        threshold,
        mode,
        advisory_result,
        len(impact_points) if impact_points is not None else 0,
        kwargs.get("confidence_index"),
    )

    # Right: target / crosshair view
    right_ax = ax.inset_axes([0.42, 0.06, 0.56, 0.90])
    _draw_target_view(
        right_ax,
        impact_points,
        target_position,
        target_radius,
        cep50,
        kwargs.get("release_point"),
        kwargs.get("wind_vector"),
    )


def _draw_banner_metrics_advisory(
    ax,
    decision,
    target_hit_percentage,
    cep50,
    threshold,
    mode,
    advisory,
    n_samples,
    confidence_index=None,
):
    ax.set_axis_off()
    ax.set_facecolor(BG_PANEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Panel border
    ax.add_patch(
        mpatches.Rectangle(
            (0, 0),
            1,
            1,
            linewidth=1,
            edgecolor=BORDER_SUBTLE,
            facecolor=BG_PANEL,
            transform=ax.transAxes,
        )
    )

    is_drop = str(decision).upper() == "DROP"
    color = ACCENT_GO if is_drop else ACCENT_NO_GO
    label = "DROP" if is_drop else "NO DROP"

    # 1. Decision Banner (Neutral bg, colored border)
    # y=0.78 to 0.96
    banner_box = mpatches.Rectangle(
        (0.05, 0.78),
        0.90,
        0.18,
        linewidth=2,
        edgecolor=color,
        facecolor=BG_PANEL,
        transform=ax.transAxes,
    )
    ax.add_patch(banner_box)

    ax.text(
        0.5,
        0.89,
        label,
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        color=color,
        ha="center",
        va="center",
        family="monospace",
    )

    # Confidence index
    phit = float(target_hit_percentage) / 100.0
    ci = float(confidence_index) if confidence_index is not None else 0.5
    if ci >= 0.75:
        ci_label = "High"
    elif ci >= 0.50:
        ci_label = "Moderate"
    else:
        ci_label = "Low"

    ax.text(
        0.5,
        0.82,
        f"Confidence Index: {ci:.2f} ({ci_label})",
        transform=ax.transAxes,
        fontsize=10,
        color=TEXT_PRIMARY,
        ha="center",
        va="center",
        family="monospace",
        weight="bold",
    )

    # Stats line inside banner
    stats = (
        f"HIT {target_hit_percentage:.1f}% | THR {threshold:.1f}% | "
        f"CEP50 {cep50:.2f}m"
    )
    ax.text(
        0.5,
        0.74,
        stats,
        transform=ax.transAxes,
        fontsize=8,
        color=TEXT_LABEL,
        ha="center",
        va="top",
        family="monospace",
    )

    # 2. Key Metrics
    y = 0.64
    ax.text(
        0.06,
        y,
        "KEY METRICS",
        transform=ax.transAxes,
        fontsize=10,
        color=ACCENT_GO,
        family="monospace",
        va="center",
    )

    y -= 0.06

    def row(lbl, val, y_pos):
        ax.text(
            0.06,
            y_pos,
            lbl,
            transform=ax.transAxes,
            fontsize=9,
            color=TEXT_LABEL,
            va="center",
            family="monospace",
        )
        ax.text(
            0.94,
            y_pos,
            val,
            transform=ax.transAxes,
            fontsize=9,
            color=TEXT_PRIMARY,
            ha="right",
            va="center",
            family="monospace",
        )

    row("Mode", str(mode), y)
    y -= 0.05
    row("Samples", str(n_samples), y)
    y -= 0.05
    # Seed is not passed here currently, skip or add if crucial.
    # It's in system status.
    row("Hits", f"{int(n_samples * phit)}/{n_samples}", y)

    # 3. Advisory Panel
    if advisory:
        y -= 0.12
        ax.text(
            0.06,
            y,
            "ADVISORY",
            transform=ax.transAxes,
            fontsize=10,
            color=ACCENT_GO,
            family="monospace",
            va="center",
        )
        y -= 0.06

        # Feasibility
        ax.text(
            0.06,
            y,
            "Feasibility",
            transform=ax.transAxes,
            fontsize=9,
            color=TEXT_LABEL,
            va="center",
            family="monospace",
        )
        ax.text(
            0.94,
            y,
            advisory.current_feasibility,
            transform=ax.transAxes,
            fontsize=9,
            color=ACCENT_GO,
            ha="right",
            va="center",
            family="monospace",
        )
        y -= 0.05

        # Trend
        ax.text(
            0.06,
            y,
            "Trend",
            transform=ax.transAxes,
            fontsize=9,
            color=TEXT_LABEL,
            va="center",
            family="monospace",
        )
        ax.text(
            0.94,
            y,
            advisory.trend_summary[:25],
            transform=ax.transAxes,
            fontsize=8,
            color=TEXT_PRIMARY,
            ha="right",
            va="center",
            family="monospace",
        )
        y -= 0.05

        # Analytical Text
        def _to_analytic(direction):
            if not direction or direction == "Hold Position":
                return "Position optimal."
            mapping = {
                "Move Forward": "Gradient suggests +X region.",
                "Move Backward": "Feasibility improves in -X.",
                "Move Left": "Feasibility improves in -Y.",
                "Move Right": "Gradient suggests +Y region.",
                "Unsafe": "Env. uncertainty > limits.",
            }
            for k, v in mapping.items():
                if k in direction:
                    return v
            return f"Gradient shift: {direction}"

        analytic_msg = _to_analytic(advisory.suggested_direction)

        ax.text(
            0.06,
            y,
            "Analysis",
            transform=ax.transAxes,
            fontsize=9,
            color=TEXT_LABEL,
            va="top",
            family="monospace",
        )
        ax.text(
            0.94,
            y,
            analytic_msg,
            transform=ax.transAxes,
            fontsize=8,
            color=TEXT_PRIMARY,
            ha="right",
            va="top",
            family="monospace",
            wrap=True,
        )


def _draw_target_view(
    ax,
    impact_points,
    target_position,
    target_radius,
    cep50,
    release_point=None,
    wind_vector=None,
):
    impact_points = np.asarray(impact_points, dtype=float)
    target_position = np.asarray(target_position, dtype=float).reshape(2)

    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TEXT_LABEL, labelsize=8)
    ax.xaxis.label.set_color(TEXT_LABEL)
    ax.yaxis.label.set_color(TEXT_LABEL)
    for spine in ax.spines.values():
        spine.set_color(BORDER_SUBTLE)
    ax.grid(True, color=BORDER_SUBTLE, alpha=0.4)

    # Plot impacts
    if impact_points.size >= 2:
        ax.scatter(
            impact_points[:, 0],
            impact_points[:, 1],
            color=SCATTER_PRIMARY,
            alpha=0.35,
            s=8,
            edgecolors="none",
        )
        mean_impact = np.mean(impact_points, axis=0)
        ax.scatter(
            mean_impact[0],
            mean_impact[1],
            color=MEAN_MARKER,
            s=50,
            zorder=5,
            marker="x",
            linewidths=1.5,
        )
    if release_point is not None:
        rp = np.asarray(release_point, dtype=float).reshape(2)
        ax.scatter(rp[0], rp[1], color=ACCENT_WARN, s=55, marker="D", zorder=5)

    # Target
    ax.add_patch(
        plt.Circle(
            target_position,
            target_radius,
            color=ACCENT_GO,
            fill=False,
            linewidth=1.5,
        )
    )
    ax.scatter(
        target_position[0],
        target_position[1],
        color=ACCENT_GO,
        s=30,
        zorder=5,
        edgecolors=BG_PANEL,
        linewidths=0.5,
    )

    if cep50 is not None and cep50 > 0:
        ax.add_patch(
            plt.Circle(
                target_position,
                cep50,
                color=CEP_CIRCLE,
                fill=False,
                linestyle="--",
                linewidth=1,
            )
        )
    if wind_vector is not None:
        wind = np.asarray(wind_vector, dtype=float).reshape(2)
        if float(np.linalg.norm(wind)) > 0:
            start = np.asarray(target_position, dtype=float).reshape(2)
            vec = wind * 6.0
            ax.arrow(
                start[0],
                start[1],
                vec[0],
                vec[1],
                width=0.18,
                head_width=0.9,
                head_length=1.3,
                color=ACCENT_WARN,
                length_includes_head=True,
            )

    # Crosshair
    cx, cy = float(target_position[0]), float(target_position[1])
    ax.plot(
        [cx - 2, cx + 2], [cy, cy], color=ACCENT_GO, linewidth=1, alpha=0.8
    )
    ax.plot(
        [cx, cx], [cy - 2, cy + 2], color=ACCENT_GO, linewidth=1, alpha=0.8
    )

    # Annotations
    ax.text(
        0.02,
        0.98,
        "IMPACT DISPERSION",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=TEXT_LABEL,
        fontsize=10,
        family="monospace",
    )
    ax.text(
        0.98,
        0.02,
        "Model: Low-subsonic, drag-dominated free fall",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=TEXT_LABEL,
        fontsize=7,
        family="monospace",
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SCATTER_PRIMARY,
            markeredgecolor="none",
            markersize=6,
            label="Impacts",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color=MEAN_MARKER,
            markersize=8,
            label="Mean",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=ACCENT_GO,
            markeredgecolor=ACCENT_GO,
            markersize=6,
            label="Target",
        ),
        Line2D([0], [0], linestyle="--", color=CEP_CIRCLE, label="CEP50"),
        Line2D([0], [0], color=ACCENT_WARN, linewidth=2, label="Wind"),
    ]
    if release_point is not None:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor=ACCENT_WARN,
                markeredgecolor=ACCENT_WARN,
                markersize=6,
                label="Release",
            )
        )
    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        fontsize=7,
        facecolor=BG_PANEL,
        edgecolor=BORDER_SUBTLE,
        labelcolor=TEXT_LABEL,
    )

    # Symmetric limit logic (replicated from plots.py for consistency if
    # this standalone view is used)
    # Actually, let's reuse logic:
    if impact_points.size > 0:
        max_d = np.max(
            np.linalg.norm(impact_points - target_position, axis=1)
        )
        rad = max(max_d * 1.1, target_radius * 2.5, 20.0)
        ax.set_xlim(cx - rad, cx + rad)
        ax.set_ylim(cy - rad, cy + rad)
    else:
        ax.set_xlim(cx - 50, cx + 50)
        ax.set_ylim(cy - 50, cy + 50)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
