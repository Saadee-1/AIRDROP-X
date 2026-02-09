"""
Qt-based desktop application entrypoint for AIRDROP-X.

This wraps the existing engine and Matplotlib-based tab renderers in a
PyQt6 window with a tab widget. The engine and decision logic remain
unchanged; this module is UI-only.
"""

from __future__ import annotations

import sys
import random
from datetime import datetime
from typing import Dict, Any

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
)

from configs import mission_configs as cfg
from src import decision_logic
from product.payloads.payload_base import Payload
from product.missions.target_manager import Target
from product.missions.environment import Environment
from product.missions.mission_state import MissionState
from product.guidance.advisory_layer import (
    get_impact_points_and_metrics,
    evaluate_advisory,
)
from product.ui import qt_bridge
from product.ui.tabs import mission_overview, payload_library, sensor_telemetry, analysis, system_status


def run_simulation_from_config(random_seed: int) -> Dict[str, Any]:
    """
    Run one engine evaluation and return a simulation snapshot.

    The engine and decision logic are treated as a black box. This helper
    only wires config into the engine and packages the results.
    """
    payload = Payload(
        mass=cfg.mass,
        drag_coefficient=cfg.Cd,
        reference_area=cfg.A,
    )
    target = Target(position=cfg.target_pos, radius=cfg.target_radius)
    environment = Environment(
        wind_mean=cfg.wind_mean,
        wind_std=cfg.wind_std,
    )
    mission_state = MissionState(
        payload=payload,
        target=target,
        environment=environment,
        uav_position=cfg.uav_pos,
        uav_velocity=cfg.uav_vel,
    )
    # Engine call: Monte Carlo and metrics.
    impact_points, P_hit, cep50 = get_impact_points_and_metrics(
        mission_state,
        random_seed,
    )
    advisory_result = evaluate_advisory(
        mission_state,
        cfg.THRESHOLD_SLIDER_INIT / 100.0,
        random_seed=random_seed,
    )

    # Immutable snapshot: config + results + timestamp.
    config: Dict[str, Any] = {
        "random_seed": random_seed,
        "n_samples": cfg.n_samples,
        "dt": cfg.dt,
        "mass": cfg.mass,
        "Cd": cfg.Cd,
        "A": cfg.A,
        "uav_pos": cfg.uav_pos,
        "uav_vel": cfg.uav_vel,
        "target_pos": cfg.target_pos,
        "target_radius": cfg.target_radius,
        "wind_mean": cfg.wind_mean,
        "wind_std": cfg.wind_std,
        "mode_thresholds": cfg.MODE_THRESHOLDS,
    }

    results: Dict[str, Any] = {
        "impact_points": impact_points,
        "P_hit": P_hit,
        "cep50": cep50,
        "target_position": mission_state.target.position,
        "target_radius": mission_state.target.radius,
        "mission_state": mission_state,
        "advisory_result": advisory_result,
        "initial_threshold_percent": cfg.THRESHOLD_SLIDER_INIT,
        "initial_mode": "Balanced",
        "slider_min": cfg.THRESHOLD_SLIDER_MIN,
        "slider_max": cfg.THRESHOLD_SLIDER_MAX,
        "slider_step": cfg.THRESHOLD_SLIDER_STEP,
        "mode_thresholds": cfg.MODE_THRESHOLDS,
    }

    snapshot: Dict[str, Any] = {
        "config": config,
        "results": results,
        "created_at": datetime.now(),
    }
    return snapshot


class AirdropMainWindow(QMainWindow):
    """Qt main window hosting the AIRDROP-X tabs."""

    def __init__(self, snapshot: Dict[str, Any]) -> None:
        super().__init__()
        self.setWindowTitle("AIRDROP-X")

        # Current immutable simulation snapshot driving all tabs.
        self._snapshot: Dict[str, Any] = snapshot

        # Seed regeneration mode (explicitly toggled by operator).
        self._regen_seed_mode: bool = False

        # UI references
        self._tabs: QTabWidget | None = None
        self._snapshot_label: QLabel | None = None
        self._status_seed_label: QLabel | None = None
        self._regen_seed_checkbox: QCheckBox | None = None

        self._init_ui()

    # Convenience accessors
    @property
    def _cfg(self) -> Dict[str, Any]:
        return self._snapshot["config"]

    @property
    def _results(self) -> Dict[str, Any]:
        return self._snapshot["results"]

    def _init_ui(self) -> None:
        """Build static window chrome plus the tab widget."""
        central = QWidget(self)
        vlayout = QVBoxLayout(central)
        vlayout.setContentsMargins(4, 4, 4, 4)

        # Snapshot banner + explicit re-run control.
        header = QWidget(central)
        hlayout = QHBoxLayout(header)
        hlayout.setContentsMargins(0, 0, 0, 0)

        self._snapshot_label = QLabel(header)
        rerun_btn = QPushButton("Re-Run Simulation", header)
        rerun_btn.clicked.connect(self._on_rerun_clicked)

        hlayout.addWidget(self._snapshot_label)
        hlayout.addStretch(1)
        hlayout.addWidget(rerun_btn)

        self._tabs = QTabWidget(central)
        self._build_tabs()
        # Keep snapshot banner and operator context in sync with tab changes.
        self._tabs.currentChanged.connect(self._on_tab_changed)

        vlayout.addWidget(header)
        vlayout.addWidget(self._tabs)

        self.setCentralWidget(central)
        self._update_snapshot_banner()

    # ---- Snapshot lifecycle ----

    def _build_tabs(self) -> None:
        """(Re)build all tabs from the current snapshot."""
        assert self._tabs is not None
        self._tabs.clear()
        self._tabs.addTab(self._make_mission_overview_tab(), "Mission Overview")
        self._tabs.addTab(self._make_payload_tab(), "Payload Library")
        self._tabs.addTab(self._make_sensor_tab(), "Sensor & Telemetry")
        self._tabs.addTab(self._make_analysis_tab(), "Analysis")
        self._tabs.addTab(self._make_system_status_tab(), "System Status")

    def _update_snapshot_banner(self) -> None:
        """Update header label describing the current snapshot."""
        if self._snapshot_label is None:
            return
        created = self._snapshot["created_at"].strftime("%Y-%m-%d %H:%M:%S")
        seed = self._cfg["random_seed"]
        mode = "non-reproducible" if self._regen_seed_mode else "reproducible"
        # Show which tab is active so the operator always knows context.
        active_tab = ""
        if self._tabs is not None and self._tabs.count() > 0:
            idx = self._tabs.currentIndex()
            if idx >= 0:
                active_tab = self._tabs.tabText(idx)
        banner = f"Simulation snapshot @ {created}  ·  seed={seed} ({mode})"
        if active_tab:
            banner += f"  ·  Active tab: {active_tab}"
        self._snapshot_label.setText(banner)
        if self._status_seed_label is not None:
            self._status_seed_label.setText(
                f"Seed used for this snapshot: {seed}"
            )

    def _on_tab_changed(self, index: int) -> None:
        """Refresh header when operator switches tabs (context clarity)."""
        # index is not used directly; we recompute from current tab state.
        self._update_snapshot_banner()

    def _on_rerun_clicked(self) -> None:
        """Explicitly re-run the engine and replace the snapshot."""
        current_seed = int(self._cfg["random_seed"])
        seed = current_seed
        if self._regen_seed_mode:
            # Non-reproducible mode: draw a fresh seed for this snapshot.
            seed = random.randint(0, 2**31 - 1)
            print(f"[AIRDROP-X] New non-reproducible seed generated: {seed}")

        # New immutable snapshot from the engine.
        self._snapshot = run_simulation_from_config(seed)
        # Rebuild all tabs from the new snapshot.
        self._build_tabs()
        self._update_snapshot_banner()

    def _on_regen_seed_toggled(self, checked: bool) -> None:
        """Operator explicitly toggles non-reproducible seed mode."""
        self._regen_seed_mode = checked
        self._update_snapshot_banner()

    # ---- Tab factories ----

    def _wrap_canvas(self, fig) -> QWidget:
        canvas = qt_bridge.create_canvas(fig)
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas)
        return widget

    def _make_mission_overview_tab(self) -> QWidget:
        fig = qt_bridge.create_figure()
        mission_data = {
            "decision": self._results["advisory_result"].current_feasibility,
            "target_hit_percentage": self._results["P_hit"] * 100.0,
            "cep50": self._results["cep50"],
            "threshold": self._results["initial_threshold_percent"],
            "mode": self._results["initial_mode"],
            "impact_points": self._results["impact_points"],
            "target_position": self._results["target_position"],
            "target_radius": self._results["target_radius"],
        }
        qt_bridge.render_into_single_axes(fig, mission_overview.render, **mission_data)
        return self._wrap_canvas(fig)

    def _make_payload_tab(self) -> QWidget:
        """
        Payload Library is view-only in the Qt app.

        Controls in this tab do NOT trigger a new engine run, so we render in
        non-interactive mode and mark this as locked for the current snapshot.
        """
        fig = qt_bridge.create_figure()
        ax = fig.add_subplot(1, 1, 1)
        payload_library.render(ax, fig, interactive=False)
        return self._wrap_canvas(fig)

    def _make_sensor_tab(self) -> QWidget:
        fig = qt_bridge.create_figure()
        qt_bridge.render_into_single_axes(fig, sensor_telemetry.render)
        return self._wrap_canvas(fig)

    def _make_analysis_tab(self) -> QWidget:
        fig = qt_bridge.create_figure()
        analysis_kwargs = {
            "impact_points": self._results["impact_points"],
            "target_position": self._results["target_position"],
            "target_radius": self._results["target_radius"],
            "cep50": self._results["cep50"],
            "target_hit_percentage": self._results["P_hit"] * 100.0,
        }
        qt_bridge.render_into_single_axes(fig, analysis.render, **analysis_kwargs)
        return self._wrap_canvas(fig)

    def _make_system_status_tab(self) -> QWidget:
        """
        System Status tab: audit-grade identity and reproducibility info.

        Includes a visible seed banner and a control to switch between
        reproducible and non-reproducible seed behaviour.
        """
        fig = qt_bridge.create_figure()
        status_kwargs = {
            "random_seed": self._cfg.get("random_seed"),
            "n_samples": self._cfg.get("n_samples"),
            "dt": self._cfg.get("dt"),
        }
        # Pass snapshot timestamp down so System Status can show creation time.
        status_kwargs["snapshot_created_at"] = self._snapshot["created_at"]
        qt_bridge.render_into_single_axes(fig, system_status.render, **status_kwargs)

        canvas = qt_bridge.create_canvas(fig)

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Seed banner + regenerate control (operator-visible).
        seed_banner = QLabel(container)
        self._status_seed_label = seed_banner
        self._update_snapshot_banner()

        regen_box = QCheckBox("Regenerate seed (non-reproducible)", container)
        regen_box.setChecked(False)
        regen_box.toggled.connect(self._on_regen_seed_toggled)
        self._regen_seed_checkbox = regen_box

        layout.addWidget(seed_banner)
        layout.addWidget(regen_box)

        # Brief seed mode explanation for auditors / operators.
        mode_explain = QLabel(
            "Seed policy: unchecked = reproducible (same seed each re-run); "
            "checked = new random seed for each re-run (non-reproducible).",
            container,
        )
        layout.addWidget(mode_explain)
        layout.addWidget(canvas)

        return container


def main() -> None:
    """Entry point for the Qt desktop application."""
    # Default is fully reproducible: use config seed unless operator chooses otherwise.
    initial_seed = cfg.RANDOM_SEED
    snapshot = run_simulation_from_config(initial_seed)

    app = QApplication(sys.argv)
    window = AirdropMainWindow(snapshot)
    window.resize(1200, 720)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

