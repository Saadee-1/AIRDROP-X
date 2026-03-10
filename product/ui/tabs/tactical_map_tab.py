from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QHBoxLayout, QVBoxLayout, QWidget

from product.ui.widgets.tactical_map_widget import TacticalMapWidget
from product.ui.widgets.uncertainty_hud import UncertaintyHUD


class TacticalMapTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.map_widget = TacticalMapWidget(self)

        hud_container = QWidget(self)
        hud_layout = QHBoxLayout(hud_container)
        hud_layout.setContentsMargins(8, 4, 8, 4)
        hud_layout.setSpacing(12)

        self.status_banner = QLabel("")
        self.status_banner.setAlignment(Qt.AlignCenter)
        self.status_banner.setStyleSheet("font-size: 16px; font-weight: bold; color: #00ff41;")

        self.uncertainty_hud = UncertaintyHUD(hud_container)

        self.wind_indicator = QLabel("Wind: --")
        self.wind_indicator.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.wind_indicator.setStyleSheet("font-size: 12px; color: #00ff41;")

        hud_layout.addWidget(self.status_banner, 2)
        hud_layout.addWidget(self.uncertainty_hud, 3)
        hud_layout.addWidget(self.wind_indicator, 1)

        root.addWidget(self.map_widget, 9)
        root.addWidget(hud_container, 1)

    def set_status(self, text: str) -> None:
        self.status_banner.setText(text or "")

    def set_wind_indicator(self, text: str) -> None:
        self.wind_indicator.setText(text or "")
