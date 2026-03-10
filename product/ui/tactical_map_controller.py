from __future__ import annotations

from typing import Any, Optional, Tuple
import time
import math

from PySide6.QtCore import QObject, QTimer

from product.runtime.system_state import SystemState
from product.system.tactical_map_state import TacticalMapState
from product.ui.widgets.tactical_map_widget import TacticalMapWidget


class TacticalMapController(QObject):
    """Bridge SystemState to TacticalMapWidget at ~30 Hz."""

    def __init__(self, system_state: SystemState, widget: TacticalMapWidget, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._state = system_state
        self._widget = widget
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_tick)
        self._last_tick = None
        self._last_impact_version = None
        self._frame_count = 0

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def _on_tick(self) -> None:
        now = time.monotonic()
        if self._last_tick is not None:
            interval = now - self._last_tick
            if interval > 0.05:
                print("[TacticalMap] UI lag detected")
            if interval > 0:
                self._widget.update_fps(1.0 / interval)
        self._last_tick = now
        self._frame_count += 1
        if self._frame_count % 300 == 0:
            self._widget.normalize_transform()

        with self._state.lock:
            assert getattr(self._state, "monte_carlo_running", False) is False
            vehicle_state = getattr(self._state, "vehicle_state", None)
            target_position = getattr(self._state, "target_position", None)
            tactical_state = getattr(self._state, "tactical_map_state", None)
            envelope_result = getattr(self._state, "envelope_result", None)
            wind_vector = getattr(self._state, "wind_vector", None)
            impact_points = getattr(self._state, "impact_points", None)
            guidance_result = getattr(self._state, "guidance_result", None)
            wind_variance = getattr(self._state, "wind_variance", None)
            wind_variance_threshold = getattr(self._state, "wind_variance_threshold", 1.0)
            impact_version = getattr(self._state, "impact_data_version", None)
            hits = getattr(self._state, "hits", None)
            total_samples = getattr(self._state, "n_samples", None)
            p_hit = getattr(self._state, "P_hit", None)
            release_corridor = getattr(self._state, "release_corridor", None)
            if release_corridor is None and isinstance(tactical_state, TacticalMapState):
                release_corridor = getattr(tactical_state, "release_corridor", None)
            if release_corridor is None:
                release_corridor = self._get(envelope_result, "release_corridor", None)

        vehicle_pos, vehicle_heading, vehicle_velocity = self._extract_vehicle(vehicle_state)
        if vehicle_pos is not None and vehicle_heading is not None:
            self._widget.update_vehicle_position(vehicle_pos[0], vehicle_pos[1], vehicle_heading)

        if target_position is not None:
            self._widget.update_target(target_position[0], target_position[1])

        if isinstance(tactical_state, TacticalMapState):
            self._apply_tactical_state(tactical_state, vehicle_pos)
        else:
            self._apply_envelope_state(envelope_result, vehicle_pos)

        if wind_vector is not None:
            self._widget.update_wind(wind_vector[0], wind_vector[1])

        if impact_points is not None and impact_version is not None:
            if impact_version != self._last_impact_version:
                self._last_impact_version = impact_version
                self._widget.update_scatter(impact_points)
                self._widget.update_heatmap(impact_points)

        self._widget.update_guidance_arrow()

        status = self._get(guidance_result, "status", None)
        if status is not None:
            self._widget.update_status(status)

        if wind_variance is not None and wind_variance_threshold is not None:
            self._widget.update_wind_warning(float(wind_variance) > float(wind_variance_threshold))

        ci = None
        if hits is not None and total_samples:
            try:
                p_hit_val = float(hits) / float(total_samples)
                ci = math.sqrt(p_hit_val * (1.0 - p_hit_val) / float(total_samples))
            except Exception:
                p_hit_val = None
        else:
            p_hit_val = p_hit
        self._widget.update_p_hit(p_hit_val, ci)

        t_drop = self._compute_drop_time(vehicle_state, release_corridor)
        self._widget.update_release_timer(t_drop)

        release_point = self._extract_release_point(guidance_result, envelope_result, tactical_state)
        impact_mean = self._extract_impact_mean(envelope_result, tactical_state)
        if release_point and impact_mean:
            self._widget.update_drift(release_point[0], release_point[1], impact_mean[0], impact_mean[1])
        else:
            self._widget.drift_arrow.set_visible(False)

    def _apply_tactical_state(self, state: TacticalMapState, vehicle_pos: Optional[Tuple[float, float]]) -> None:
        if state.impact_mean and state.ellipse_axes and state.ellipse_angle is not None:
            self._widget.update_impact_ellipse(
                state.impact_mean[0],
                state.impact_mean[1],
                state.ellipse_axes[0],
                state.ellipse_axes[1],
                state.ellipse_angle,
            )

        if state.release_corridor:
            self._widget.update_corridor(state.release_corridor)

        # Guidance arrow uses corridor centerline in widget.

    def _apply_envelope_state(self, envelope_result: Any, vehicle_pos: Optional[Tuple[float, float]]) -> None:
        if envelope_result is None:
            return
        impact_mean = self._get(envelope_result, "impact_mean", None)
        ellipse_axes = self._get(envelope_result, "ellipse_axes", None)
        ellipse_angle = self._get(envelope_result, "ellipse_angle", None)
        corridor = self._get(envelope_result, "release_corridor", None)
        guidance_vector = self._get(envelope_result, "guidance_vector", None)

        if impact_mean and ellipse_axes and ellipse_angle is not None:
            self._widget.update_impact_ellipse(
                impact_mean[0], impact_mean[1], ellipse_axes[0], ellipse_axes[1], ellipse_angle
            )

        if corridor:
            self._widget.update_corridor(corridor)

        # Guidance arrow uses corridor centerline in widget.

    @staticmethod
    def _extract_vehicle(vehicle_state: Any) -> Tuple[Optional[Tuple[float, float]], Optional[float], Optional[Tuple[float, float]]]:
        if vehicle_state is None:
            return None, None, None
        pos = TacticalMapController._get(vehicle_state, "position", None)
        heading = TacticalMapController._get(vehicle_state, "heading", None)
        if heading is None:
            heading = TacticalMapController._get(vehicle_state, "heading_deg", None)
        velocity = TacticalMapController._get(vehicle_state, "velocity", None)
        if pos is None:
            return None, heading, velocity
        return (float(pos[0]), float(pos[1])), heading, velocity

    @staticmethod
    def _compute_drop_time(vehicle_state: Any, corridor: Any) -> Optional[float]:
        pos = TacticalMapController._get(vehicle_state, "position", None)
        vel = TacticalMapController._get(vehicle_state, "velocity", None)
        if pos is None or vel is None or corridor is None:
            return None
        try:
            vx, vy = float(vel[0]), float(vel[1])
            speed = math.hypot(vx, vy)
            if speed <= 0:
                return None
            pts = list(corridor)
            if len(pts) < 2:
                return None
            entry_x = (pts[0][0] + pts[1][0]) * 0.5
            entry_y = (pts[0][1] + pts[1][1]) * 0.5
            dist = math.hypot(entry_x - pos[0], entry_y - pos[1])
            return dist / speed
        except Exception:
            return None

    @staticmethod
    def _extract_release_point(guidance_result: Any, envelope_result: Any, tactical_state: Any) -> Optional[Tuple[float, float]]:
        rp = TacticalMapController._get(guidance_result, "target_release_point", None)
        if rp is None:
            rp = TacticalMapController._get(envelope_result, "release_point", None)
        if rp is None and isinstance(tactical_state, TacticalMapState):
            rp = TacticalMapController._get(tactical_state, "release_point", None)
        if rp is None:
            return None
        return (float(rp[0]), float(rp[1]))

    @staticmethod
    def _extract_impact_mean(envelope_result: Any, tactical_state: Any) -> Optional[Tuple[float, float]]:
        im = TacticalMapController._get(envelope_result, "impact_mean", None)
        if im is None and isinstance(tactical_state, TacticalMapState):
            im = TacticalMapController._get(tactical_state, "impact_mean", None)
        if im is None:
            return None
        return (float(im[0]), float(im[1]))

    @staticmethod
    def _get(obj: Any, name: str, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)
