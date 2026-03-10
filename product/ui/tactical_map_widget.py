from PySide6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPolygonItem,
    QGraphicsTextItem,
)
from PySide6.QtGui import QPen, QBrush, QPolygonF, QFont
from PySide6.QtCore import Qt, QPointF
import numpy as np
import math


class TacticalMapWidget(QGraphicsView):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(self.RenderHint.Antialiasing, True)
        self.setBackgroundBrush(QBrush(Qt.black))
        self.scene.setSceneRect(-1000.0, -1000.0, 2000.0, 2000.0)

        self.grid_layer = self.scene.createItemGroup([])
        self.zone_layer = self.scene.createItemGroup([])
        self.corridor_layer = self.scene.createItemGroup([])

        self.target_item = self.scene.createItemGroup([])
        self._target_circle = self.scene.addEllipse(
            -6.0, -6.0, 12.0, 12.0, QPen(Qt.green, 2), QBrush(Qt.transparent)
        )
        self._target_cross_h = self.scene.addLine(
            -10.0, 0.0, 10.0, 0.0, QPen(Qt.green, 1)
        )
        self._target_cross_v = self.scene.addLine(
            0.0, -10.0, 0.0, 10.0, QPen(Qt.green, 1)
        )
        self._target_radius_item = self.scene.addEllipse(
            -1.0, -1.0, 2.0, 2.0, QPen(Qt.green, 1), QBrush(Qt.transparent)
        )
        self.target_item.addToGroup(self._target_circle)
        self.target_item.addToGroup(self._target_cross_h)
        self.target_item.addToGroup(self._target_cross_v)
        self.target_item.addToGroup(self._target_radius_item)

        uav_poly = QPolygonF(
            [
                QPointF(10.0, 0.0),
                QPointF(-8.0, -6.0),
                QPointF(-8.0, 6.0),
            ]
        )
        self.uav_item = QGraphicsPolygonItem(uav_poly)
        self.uav_item.setPen(QPen(Qt.cyan, 1.5))
        self.uav_item.setBrush(QBrush(Qt.cyan))
        self.uav_item.setTransformOriginPoint(0.0, 0.0)
        self.scene.addItem(self.uav_item)

        self.guidance_arrow = self.scene.addLine(
            0.0, 0.0, 0.0, 0.0, QPen(Qt.cyan, 2)
        )
        self.impact_ellipse = self.scene.addEllipse(
            -1.0, -1.0, 2.0, 2.0, QPen(Qt.red, 1), QBrush(Qt.transparent)
        )

        self._zone_polygon = QGraphicsPolygonItem()
        self._zone_polygon.setPen(QPen(Qt.darkRed, 1))
        self._zone_polygon.setBrush(QBrush(Qt.transparent))
        self.zone_layer.addToGroup(self._zone_polygon)

        self._corridor_polygon = QGraphicsPolygonItem()
        self._corridor_polygon.setPen(QPen(Qt.white, 1))
        self._corridor_polygon.setBrush(QBrush(Qt.white))
        self._corridor_polygon.setOpacity(0.25)
        self.corridor_layer.addToGroup(self._corridor_polygon)

        self._corridor_length = 200.0
        self._last_envelope_key = None

        self._banner_item = QGraphicsTextItem()
        banner_font = QFont()
        banner_font.setBold(True)
        banner_font.setPointSize(20)
        self._banner_item.setFont(banner_font)
        self._banner_item.setDefaultTextColor(Qt.cyan)
        self._banner_item.setZValue(20)
        self.scene.addItem(self._banner_item)

        self._uncertainty_text = QGraphicsTextItem()
        unc_font = QFont("Consolas", 9)
        unc_font.setBold(True)
        self._uncertainty_text.setFont(unc_font)
        self._uncertainty_text.setDefaultTextColor(Qt.white)
        self._uncertainty_text.setOpacity(0.65)
        self._uncertainty_text.setZValue(15)
        self.scene.addItem(self._uncertainty_text)

        self._build_grid()

    def _build_grid(self) -> None:
        self.grid_layer.setZValue(-10)
        pen = QPen(Qt.lightGray, 0)
        step = 50.0
        extent = 1000.0
        for x in np.arange(-extent, extent + step, step):
            line = self.scene.addLine(x, -extent, x, extent, pen)
            self.grid_layer.addToGroup(line)
        for y in np.arange(-extent, extent + step, step):
            line = self.scene.addLine(-extent, y, extent, y, pen)
            self.grid_layer.addToGroup(line)

    def update_state(self, system_state) -> None:
        lock = getattr(system_state, "lock", None)
        if lock is not None:
            lock.acquire()
        try:
            vehicle_state = self._get(system_state, "vehicle_state", {})
            guidance_result = self._get(system_state, "guidance_result", {})
            envelope_result = self._get(system_state, "envelope_result", {})
            target_position = self._get(system_state, "target_position", None)
        finally:
            if lock is not None:
                lock.release()

        if target_position is not None:
            tp = np.asarray(target_position, dtype=float).flatten()[:2]
            if tp.size == 2:
                self.target_item.setPos(float(tp[0]), float(tp[1]))
                self.target_item.setVisible(True)
        else:
            self.target_item.setVisible(False)

        pos = self._get(vehicle_state, "position", None)
        vel = self._get(vehicle_state, "velocity", None)
        if pos is not None:
            vp = np.asarray(pos, dtype=float).flatten()[:2]
            if vp.size == 2:
                self.uav_item.setPos(float(vp[0]), float(vp[1]))
                self.uav_item.setVisible(True)
                if vel is not None:
                    vv = np.asarray(vel, dtype=float).flatten()[:2]
                    if vv.size == 2:
                        heading = math.degrees(math.atan2(float(vv[1]), float(vv[0])))
                        self.uav_item.setRotation(heading)
        else:
            self.uav_item.setVisible(False)

        arrow = self._get(guidance_result, "heading_vector", None)
        if arrow is None:
            arrow = self._get(guidance_result, "guidance_vector", None)
        if pos is not None and arrow is not None:
            vp = np.asarray(pos, dtype=float).flatten()[:2]
            av = np.asarray(arrow, dtype=float).flatten()[:2]
            if vp.size == 2 and av.size == 2:
                scale = float(self._get(guidance_result, "arrow_scale", 50.0))
                x1, y1 = float(vp[0]), float(vp[1])
                x2, y2 = x1 + float(av[0]) * scale, y1 + float(av[1]) * scale
                self.guidance_arrow.setLine(x1, y1, x2, y2)
                self.guidance_arrow.setVisible(True)
        else:
            self.guidance_arrow.setVisible(False)

        envelope_key = None
        if envelope_result is not None:
            feasible_offsets = self._get(envelope_result, "feasible_offsets", None)
            impact_mean = self._get(envelope_result, "impact_mean", None)
            impact_cov = self._get(envelope_result, "impact_cov", None)
            envelope_key = (
                self._key(feasible_offsets),
                self._key(impact_mean),
                self._key(impact_cov),
            )

            if envelope_key != self._last_envelope_key:
                self._last_envelope_key = envelope_key

                if feasible_offsets is not None and pos is not None and vel is not None:
                    offsets = np.asarray(feasible_offsets, dtype=float).flatten()
                    if offsets.size:
                        corridor_half_width = float(np.max(np.abs(offsets)))
                        vp = np.asarray(pos, dtype=float).flatten()[:2]
                        vv = np.asarray(vel, dtype=float).flatten()[:2]
                        if vp.size == 2 and vv.size == 2:
                            heading = math.atan2(float(vv[1]), float(vv[0]))
                            dx = math.cos(heading)
                            dy = math.sin(heading)
                            px = -dy
                            py = dx
                            half_len = self._corridor_length * 0.5
                            cx, cy = float(vp[0]), float(vp[1])
                            corners = [
                                (cx - dx * half_len + px * corridor_half_width,
                                 cy - dy * half_len + py * corridor_half_width),
                                (cx + dx * half_len + px * corridor_half_width,
                                 cy + dy * half_len + py * corridor_half_width),
                                (cx + dx * half_len - px * corridor_half_width,
                                 cy + dy * half_len - py * corridor_half_width),
                                (cx - dx * half_len - px * corridor_half_width,
                                 cy - dy * half_len - py * corridor_half_width),
                            ]
                            poly = QPolygonF(
                                [QPointF(float(x), float(y)) for x, y in corners]
                            )
                            self._corridor_polygon.setPolygon(poly)
                            self._corridor_polygon.setVisible(True)
                        else:
                            self._corridor_polygon.setVisible(False)
                else:
                    self._corridor_polygon.setVisible(False)

                zone = self._get(envelope_result, "zone_polygon", None)
                if zone is not None:
                    pts = np.asarray(zone, dtype=float).reshape(-1, 2)
                    if pts.size >= 6:
                        poly = QPolygonF(
                            [QPointF(float(x), float(y)) for x, y in pts]
                        )
                        self._zone_polygon.setPolygon(poly)
                        self._zone_polygon.setVisible(True)
                    else:
                        self._zone_polygon.setVisible(False)
                else:
                    self._zone_polygon.setVisible(False)

                if impact_mean is not None and impact_cov is not None:
                    mean = np.asarray(impact_mean, dtype=float).flatten()[:2]
                    cov = np.asarray(impact_cov, dtype=float).reshape(2, 2)
                    if mean.size == 2:
                        vals, vecs = np.linalg.eig(cov)
                        order = np.argsort(vals)[::-1]
                        vals = vals[order]
                        vecs = vecs[:, order]
                        major = math.sqrt(max(float(vals[0]), 0.0))
                        minor = math.sqrt(max(float(vals[1]), 0.0))
                        angle = math.degrees(math.atan2(float(vecs[1, 0]), float(vecs[0, 0])))
                        cx, cy = float(mean[0]), float(mean[1])
                        self.impact_ellipse.setRect(
                            cx - major, cy - minor, major * 2.0, minor * 2.0
                        )
                        self.impact_ellipse.setTransformOriginPoint(cx, cy)
                        self.impact_ellipse.setRotation(angle)
                        self.impact_ellipse.setPen(QPen(Qt.blue, 1))
                        self.impact_ellipse.setBrush(QBrush(Qt.blue))
                        self.impact_ellipse.setOpacity(0.25)
                        self.impact_ellipse.setVisible(True)
                    else:
                        self.impact_ellipse.setVisible(False)
                else:
                    self.impact_ellipse.setVisible(False)
        else:
            self._corridor_polygon.setVisible(False)
            self._zone_polygon.setVisible(False)
            self.impact_ellipse.setVisible(False)

        if pos is not None:
            rp = self._get(guidance_result, "target_release_point", None)
            if rp is not None:
                rp = np.asarray(rp, dtype=float).flatten()[:2]
                vp = np.asarray(pos, dtype=float).flatten()[:2]
                if rp.size == 2 and vp.size == 2:
                    self.guidance_arrow.setLine(
                        float(vp[0]),
                        float(vp[1]),
                        float(rp[0]),
                        float(rp[1]),
                    )
                    self.guidance_arrow.setVisible(True)
                else:
                    self.guidance_arrow.setVisible(False)
            else:
                self.guidance_arrow.setVisible(False)
        else:
            self.guidance_arrow.setVisible(False)

        self._update_banner(guidance_result)
        self._update_uncertainty_bars(guidance_result, system_state)

        target_radius = None
        if envelope_result is not None:
            target_radius = self._get(envelope_result, "target_radius", None)
        if target_radius is None:
            target_radius = self._get(system_state, "target_radius", None)
        if target_radius is not None:
            r = float(target_radius)
            self._target_radius_item.setRect(-r, -r, r * 2.0, r * 2.0)
            self._target_radius_item.setVisible(True)
        else:
            self._target_radius_item.setVisible(False)

    @staticmethod
    def _get(obj, name, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    @staticmethod
    def _key(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=float).flatten()
            return (arr.size, tuple(np.round(arr, 5)))
        if isinstance(value, dict):
            return tuple(sorted(value.items()))
        return value

    def _update_banner(self, guidance_result) -> None:
        status = str(self._get(guidance_result, "status", "") or "").strip().upper()
        if not status:
            self._banner_item.setPlainText("")
            return
        color = Qt.cyan
        if status == "DROP NOW":
            color = Qt.green
        elif status == "IN DROP ZONE":
            color = Qt.cyan
        elif status == "APPROACH CORRIDOR":
            color = Qt.yellow
        elif status == "NO DROP":
            color = Qt.red
        self._banner_item.setDefaultTextColor(color)
        self._banner_item.setPlainText(status)
        rect = self.scene.sceneRect()
        br = self._banner_item.boundingRect()
        x = rect.center().x() - br.width() * 0.5
        y = rect.top() + 10.0
        self._banner_item.setPos(x, y)

    def _update_uncertainty_bars(self, guidance_result, system_state) -> None:
        p_hit = self._get(guidance_result, "P_hit", None)
        threshold = self._get(guidance_result, "threshold", None)
        if threshold is None:
            threshold = self._get(system_state, "threshold", None)
        try:
            p_hit_val = float(p_hit) if p_hit is not None else None
            threshold_val = float(threshold) if threshold is not None else None
        except Exception:
            p_hit_val = None
            threshold_val = None

        if p_hit_val is None or threshold_val is None or p_hit_val >= threshold_val:
            self._uncertainty_text.setPlainText("")
            return

        contrib = self._get(guidance_result, "uncertainty_contribution", None)
        if contrib is None:
            contrib = self._get(system_state, "uncertainty_contribution", {})
        wind = self._get(contrib, "wind", None)
        drag = self._get(contrib, "drag", None)
        release = self._get(contrib, "release", None)
        vehicle = self._get(contrib, "vehicle", None)
        values = {
            "Wind": wind,
            "Drag": drag,
            "Release": release,
            "Vehicle": vehicle,
        }

        def _norm(val):
            if val is None:
                return 0.0
            v = float(val)
            return v / 100.0 if v > 1.0 else v

        lines = []
        bar_len = 10
        for label, val in values.items():
            pct = _norm(val)
            filled = int(round(pct * bar_len))
            bar = "█" * filled + " " * (bar_len - filled)
            pct_txt = f"{int(round(pct * 100.0)):>3d}%"
            lines.append(f"{label:<8} {bar} {pct_txt}")

        text = "\n".join(lines)
        self._uncertainty_text.setPlainText(text)
        rect = self.scene.sceneRect()
        x = rect.left() + 10.0
        y = rect.bottom() - (self._uncertainty_text.boundingRect().height() + 10.0)
        self._uncertainty_text.setPos(x, y)
