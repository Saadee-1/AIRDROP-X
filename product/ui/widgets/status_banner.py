from enum import IntEnum
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QBrush, QColor, QFont
from PySide6.QtWidgets import QWidget, QLabel


class DropStatus(IntEnum):
    NO_DROP = 0
    APPROACH_CORRIDOR = 1
    IN_DROP_ZONE = 2
    DROP_NOW = 3


class StatusBannerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setFixedSize(320, 44)

        self._current_status = DropStatus.NO_DROP

        # Create label for text
        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("color: white; font-weight: bold; font-size: 13pt; background-color: transparent;")
        self._label.setGeometry(0, 0, 320, 44)

        self._update_display()

    def set_status(self, status: DropStatus):
        if self._current_status != status:
            self._current_status = status
            self._update_display()
            self.update()

    @property
    def current_status(self) -> DropStatus:
        return self._current_status

    def _update_display(self):
        if self._current_status == DropStatus.NO_DROP:
            bg_color = QColor("#CC2200")
            text = "NO DROP"
            text_color = "white"
        elif self._current_status == DropStatus.APPROACH_CORRIDOR:
            bg_color = QColor("#FF8C00")
            text = "APPROACH CORRIDOR"
            text_color = "white"
        elif self._current_status == DropStatus.IN_DROP_ZONE:
            bg_color = QColor("#00AA44")
            text = "IN DROP ZONE"
            text_color = "white"
        elif self._current_status == DropStatus.DROP_NOW:
            bg_color = QColor("#00FF66")
            text = "DROP NOW"
            text_color = "black"
        else:
            bg_color = QColor("#CC2200")
            text = "UNKNOWN"
            text_color = "white"

        bg_color.setAlpha(200)
        self._bg_color = bg_color
        self._label.setText(text)
        self._label.setStyleSheet(f"color: {text_color}; font-weight: bold; font-size: 13pt; background-color: transparent;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(self._bg_color))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 6, 6)