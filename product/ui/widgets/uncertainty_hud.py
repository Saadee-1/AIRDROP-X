from __future__ import annotations

from typing import Dict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget


class UncertaintyHUD(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._bars: Dict[str, QProgressBar] = {}
        for label in ("Wind", "Drag", "Release", "Vehicle"):
            title = QLabel(label)
            title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(True)
            bar.setFormat("%p%")
            bar.setMaximumHeight(12)
            bar.setStyleSheet(
                "QProgressBar { border: 1px solid #2a3a2a; background: #0b120b; }"
                "QProgressBar::chunk { background-color: #00ff41; }"
            )
            layout.addWidget(title)
            layout.addWidget(bar)
            self._bars[label] = bar

    def set_values(self, wind: float, drag: float, release: float, vehicle: float) -> None:
        values = {
            "Wind": wind,
            "Drag": drag,
            "Release": release,
            "Vehicle": vehicle,
        }
        for key, val in values.items():
            bar = self._bars.get(key)
            if bar is None:
                continue
            bar.setValue(int(round(float(val))))
