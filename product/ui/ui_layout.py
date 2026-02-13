"""
Unified single-window UI. Tab container: one figure, tab bar, content area.
Handles tab switching only. No business logic.
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from . import tabs
from .tabs import mission_overview
from .tabs import payload_library
from .tabs import analysis
from .tabs import system_status

# Import unified military-grade theme
from product.ui.ui_theme import (
    BG_PANEL,
    TEXT_PRIMARY,
    BUTTON_HOVER,
)


def _make_tab_buttons(fig, n_tabs, on_tab_click):
    """Create a row of tab button axes; return list of Button widgets."""
    left, bottom, width, height = 0.04, 0.88, 0.14, 0.06
    gap = 0.02
    buttons = []
    for i in range(n_tabs):
        ax = fig.add_axes([left + i * (width + gap), bottom, width, height])
        ax.set_facecolor(BG_PANEL)
        ax.set_axis_off()
        btn = Button(
            ax, tabs.TABS[i][0], color=BG_PANEL, hovercolor=BUTTON_HOVER
        )
        btn.label.set_color(TEXT_PRIMARY)
        btn.label.set_fontsize(9)
        btn.on_clicked(lambda event, idx=i: on_tab_click(idx))
        buttons.append(btn)
    return buttons


def launch_unified_ui(
    impact_points,
    P_hit,
    cep50,
    target_position,
    target_radius,
    mission_state,
    advisory_result,
    initial_threshold_percent,
    initial_mode,
    slider_min,
    slider_max,
    slider_step,
    mode_thresholds,
    on_threshold_change,
    random_seed=None,
    n_samples=None,
    dt=None,
    run_simulation_callback=None,
    impact_velocity_stats=None,
    confidence_index=None,
    release_point=None,
    wind_vector=None,
    wind_mean=None,
    wind_std=None,
    max_safe_impact_speed=None,
):
    """
    Create one figure with tab bar and content area. Only one tab's content
    is shown at a time. Same signature for app compatibility; no business
    logic in this module.
    """
    fig = plt.figure(figsize=(10, 6), facecolor="#0c0e0c")
    content_ax = fig.add_axes([0.06, 0.06, 0.88, 0.78])

    mission_data = {
        "decision": advisory_result.current_feasibility,
        "target_hit_percentage": P_hit * 100.0,
        "cep50": cep50,
        "threshold": initial_threshold_percent,
        "mode": initial_mode,
        "impact_points": impact_points,
        "target_position": target_position,
        "target_radius": target_radius,
        "advisory_result": advisory_result,
        "impact_velocity_stats": impact_velocity_stats,
        "confidence_index": confidence_index,
        "release_point": release_point,
        "wind_vector": wind_vector,
        "wind_mean": wind_mean,
        "wind_std": wind_std,
        "max_safe_impact_speed": max_safe_impact_speed,
    }

    def show_tab(index):
        # Remove axes from other tabs (e.g. Payload Library widgets)
        keep = {content_ax} | set(tab_button_axes)
        for ax in list(fig.axes):
            if ax not in keep:
                fig.delaxes(ax)
        content_ax.clear()
        content_ax.set_axis_off()
        if index == 0:
            mission_overview.render(content_ax, **mission_data)
        elif index == 1:
            # Pass callback if provided
            payload_library.render(
                content_ax,
                fig,
                run_simulation_callback=run_simulation_callback,
            )
        elif index == 2:
            tabs.TABS[2][1](
                content_ax,
                wind_mean_ms=(
                    float(mission_data["wind_mean"][0])
                    if mission_data.get("wind_mean") is not None
                    else None
                ),
                wind_std_dev_ms=mission_data.get("wind_std"),
                telemetry_live=False,
            )
        elif index == 3:
            analysis.render(
                content_ax,
                impact_points=mission_data["impact_points"],
                target_position=mission_data["target_position"],
                target_radius=mission_data["target_radius"],
                cep50=mission_data["cep50"],
                target_hit_percentage=mission_data["target_hit_percentage"],
                impact_velocity_stats=mission_data.get("impact_velocity_stats"),
                max_safe_impact_speed=mission_data.get("max_safe_impact_speed"),
            )
        elif index == 4:
            system_status.render(
                content_ax,
                random_seed=random_seed,
                n_samples=n_samples,
                dt=dt,
            )
        else:
            tabs.TABS[index][1](content_ax)
        fig.canvas.draw_idle()

    buttons = _make_tab_buttons(fig, len(tabs.TABS), lambda idx: show_tab(idx))
    tab_button_axes = [b.ax for b in buttons]
    show_tab(0)

    plt.show()
