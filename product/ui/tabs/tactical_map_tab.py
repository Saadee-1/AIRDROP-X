from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QFrame, QHBoxLayout, QVBoxLayout, QWidget

from product.ui.widgets.tactical_map_widget import TacticalMapWidget


class HUDOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 10, 0)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignTop | Qt.AlignRight)

        self._north_label = QLabel("N ↑")
        self._north_label.setStyleSheet(
            "color: #00ff41; font-family: Consolas; "
            "font-size: 14px; font-weight: bold;"
        )
        self._north_label.setAlignment(Qt.AlignRight)
        layout.addWidget(self._north_label)

        layout.addSpacing(8)

        self._bars = {}
        bar_defs = [
            ("Wind",    "#ff6464"),
            ("Drag",    "#64b4ff"),
            ("Release", "#ffc850"),
            ("Vehicle", "#64ff96"),
        ]
        for name, color in bar_defs:
            row = QWidget()
            row.setStyleSheet("background: transparent;")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)

            label = QLabel(name)
            label.setFixedWidth(55)
            label.setStyleSheet(
                "color: #aaaaaa; font-family: Consolas; font-size: 9px;"
            )

            bar_bg = QFrame()
            bar_bg.setFixedSize(100, 10)
            bar_bg.setStyleSheet(
                "background: #1a1a1a; border: 1px solid #333333;"
            )

            bar_fill = QFrame(bar_bg)
            bar_fill.setFixedSize(0, 10)
            bar_fill.setStyleSheet(f"background: {color}; border: none;")
            bar_fill.move(0, 0)

            value_label = QLabel("0%")
            value_label.setFixedWidth(35)
            value_label.setStyleSheet(
                "color: #ffffff; font-family: Consolas; font-size: 9px;"
            )
            value_label.setAlignment(Qt.AlignRight)

            row_layout.addWidget(label)
            row_layout.addWidget(bar_bg)
            row_layout.addWidget(value_label)
            layout.addWidget(row)

            self._bars[name] = {
                "fill": bar_fill,
                "value": value_label,
                "bg": bar_bg,
            }

    def update_bars(self, wind_pct, drag_pct, release_pct, vehicle_pct, wind_val, drag_val, release_val, vehicle_val):
        data = [
            ("Wind",    wind_pct,    wind_val),
            ("Drag",    drag_pct,    drag_val),
            ("Release", release_pct, release_val),
            ("Vehicle", vehicle_pct, vehicle_val),
        ]
        for name, pct, val in data:
            bar = self._bars.get(name)
            if bar is None:
                continue
            pct = max(0.0, min(100.0, float(pct)))
            fill_w = int(pct / 100.0 * 100)
            bar["fill"].setFixedWidth(fill_w)
            if isinstance(val, float):
                bar["value"].setText(f"{val:.1f}")
            else:
                bar["value"].setText(str(val))


class TacticalMapTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # Military-grade tactical styling
        self.setStyleSheet("""
            QWidget {
                background-color: #0a0e1a;
                color: #00ff41;
            }
        """)
        
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.map_widget = TacticalMapWidget(self)
        root.addWidget(self.map_widget, 1)

        self.hud_overlay = HUDOverlay(self)
        self.hud_overlay.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.hud_overlay.setGeometry(self.rect())

    def update_uncertainty_bars(
        self, wind_pct, drag_pct, release_pct, vehicle_pct, wind_val, drag_val, release_val, vehicle_val
    ):
        self.hud_overlay.update_bars(
            wind_pct,
            drag_pct,
            release_pct,
            vehicle_pct,
            wind_val,
            drag_val,
            release_val,
            vehicle_val,
        )

    def set_status(self, text: str) -> None:
        self.map_widget.update_status(text or "")

    def set_wind_indicator(self, text: str) -> None:
        # No footer wind label; retained for compatibility.
        return
