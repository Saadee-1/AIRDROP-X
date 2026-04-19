from enum import IntEnum
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPainter, QBrush, QColor
from PySide6.QtWidgets import QWidget, QLabel


class DropStatus(IntEnum):
    NO_DROP = 0
    APPROACH_CORRIDOR = 1
    IN_DROP_ZONE = 2
    DROP_NOW = 3


class DropReason(IntEnum):
    NONE = 0
    MISSION_PARAMS_NOT_SET = 1
    UAV_TOO_FAR = 2
    WIND_EXCEEDED = 3


class StatusBannerWidget(QWidget):
    navigate_to_tab = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(320, 64)

        self._current_status = DropStatus.NO_DROP
        self._current_reason = DropReason.NONE
        self._blink_visible = True

        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet(
            "color: white; font-weight: bold; font-size: 13pt; background-color: transparent;"
        )
        self._label.setGeometry(0, 0, 320, 32)

        self._advisory_label = QLabel(self)
        self._advisory_label.setAlignment(Qt.AlignCenter)
        self._advisory_label.setStyleSheet(
            "color: white; font-size: 9pt; background-color: transparent;"
        )
        self._advisory_label.setGeometry(0, 30, 320, 28)

        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(1000)
        self._blink_timer.timeout.connect(self._toggle_advisory_blink)

        self._update_display()

    def set_status(self, status: DropStatus, reason: DropReason = DropReason.NONE):
        if self._current_status != status or self._current_reason != reason:
            self._current_status = status
            self._current_reason = reason
            if status == DropStatus.NO_DROP and reason == DropReason.MISSION_PARAMS_NOT_SET:
                if not self._blink_timer.isActive():
                    self._blink_timer.start()
            else:
                if self._blink_timer.isActive():
                    self._blink_timer.stop()
                self._blink_visible = True
                self._advisory_label.setVisible(True)
            self._update_display()
            self.update()

    @property
    def current_status(self) -> DropStatus:
        return self._current_status

    @property
    def current_reason(self) -> DropReason:
        return self._current_reason

    def _update_display(self):
        if self._current_status == DropStatus.NO_DROP:
            bg_color = QColor("#CC2200")
            text = "NO DROP"
            text_color = "white"
            if self._current_reason == DropReason.MISSION_PARAMS_NOT_SET:
                advisory = "Mission parameters not set — click to configure"
                advisory_color = "#FFDD00"
            elif self._current_reason == DropReason.UAV_TOO_FAR:
                advisory = "Outside drop corridor — adjust heading"
                advisory_color = "white"
            elif self._current_reason == DropReason.WIND_EXCEEDED:
                advisory = "Wind envelope exceeded — hold position"
                advisory_color = "white"
            else:
                advisory = ""
                advisory_color = "white"
        elif self._current_status == DropStatus.APPROACH_CORRIDOR:
            bg_color = QColor("#FF8C00")
            text = "APPROACH CORRIDOR"
            text_color = "white"
            advisory = "Intercept heading — maintain altitude"
            advisory_color = "white"
        elif self._current_status == DropStatus.IN_DROP_ZONE:
            bg_color = QColor("#00AA44")
            text = "IN DROP ZONE"
            text_color = "white"
            advisory = "Confirm release conditions"
            advisory_color = "white"
        elif self._current_status == DropStatus.DROP_NOW:
            bg_color = QColor("#00FF66")
            text = "DROP NOW"
            text_color = "black"
            advisory = "Release payload immediately"
            advisory_color = "black"
        else:
            bg_color = QColor("#CC2200")
            text = "UNKNOWN"
            text_color = "white"
            advisory = ""
            advisory_color = "white"

        bg_color.setAlpha(200)
        self._bg_color = bg_color
        self._label.setText(text)
        self._label.setStyleSheet(
            f"color: {text_color}; font-weight: bold; font-size: 13pt; background-color: transparent;"
        )
        self._advisory_label.setText(advisory)
        self._advisory_label.setStyleSheet(
            f"color: {advisory_color}; font-size: 9pt; background-color: transparent;"
        )

    def _toggle_advisory_blink(self):
        self._blink_visible = not self._blink_visible
        self._advisory_label.setVisible(self._blink_visible)

    def mousePressEvent(self, event):
        if (
            self._current_status == DropStatus.NO_DROP
            and self._current_reason == DropReason.MISSION_PARAMS_NOT_SET
        ):
            self.navigate_to_tab.emit(2)

    def wheelEvent(self, event):
        super().wheelEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(self._bg_color))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 6, 6)
