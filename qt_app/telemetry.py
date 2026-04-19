"""Telemetry ingestion worker for SCYTHE Qt app."""

from __future__ import annotations

import random
import time
from pathlib import Path

from PySide6.QtCore import QThread, Signal


def _dict_from_mock():
    age_s = random.uniform(0.1, 1.2)
    if age_s < 0.5:
        status = "Fresh"
    elif age_s < 1.0:
        status = "Delay"
    else:
        status = "Lost"
    return {
        "x": random.uniform(-50.0, 150.0),
        "y": random.uniform(-50.0, 50.0),
        "z": random.uniform(120.0, 260.0),
        "vx": random.uniform(10.0, 30.0),
        "vy": random.uniform(-4.0, 4.0),
        "wind_x": random.uniform(-6.0, 8.0),
        "wind_y": random.uniform(-4.0, 4.0),
        "wind_std": random.uniform(0.5, 3.0),
        "packet_rate_hz": random.uniform(8.0, 12.0),
        "age_s": age_s,
        "status": status,
    }


def _dict_from_snapshot(snapshot, wind_x: float = 2.0, wind_std: float = 0.8):
    """Convert UAVStateSnapshot to the dict shape expected by MainWindow."""
    pos = snapshot.position
    vel = snapshot.velocity
    return {
        "x": pos[0],
        "y": pos[1],
        "z": pos[2],
        "vx": vel[0],
        "vy": vel[1],
        "wind_x": wind_x,
        "wind_y": 0.0,
        "wind_std": wind_std,
        "packet_rate_hz": 10.0,
        "age_s": 0.2,
        "status": "Fresh",
    }


class TelemetryWorker(QThread):
    """Background telemetry producer. source='mock' | 'file'; file_path used when source='file'."""

    telemetry_updated = Signal(dict)

    def __init__(self, parent=None, source: str = "mock", file_path: str | Path | None = None) -> None:
        super().__init__(parent)
        self.running = True
        self._source = str(source).strip().lower() if source else "mock"
        self._file_path = Path(file_path) if file_path else None

    def run(self) -> None:
        if self._source == "file" and self._file_path and self._file_path.is_file():
            self._run_file_replay()
        else:
            self._run_mock()

    def _run_mock(self) -> None:
        while self.running:
            payload = _dict_from_mock()
            self.telemetry_updated.emit(payload)
            self.msleep(500)

    def _run_file_replay(self) -> None:
        try:
            from product.integrations.log_replay import load_replay_csv
            from configs import mission_configs as cfg
            wind_x = float(cfg.wind_mean[0]) if cfg.wind_mean else 2.0
            wind_std = float(cfg.wind_std) if getattr(cfg, "wind_std", None) is not None else 0.8
            snapshots = list(load_replay_csv(str(self._file_path)))
        except Exception:
            snapshots = []
        if not snapshots:
            while self.running:
                self.telemetry_updated.emit(_dict_from_mock())
                self.msleep(500)
            return
        interval_s = 0.1
        t0 = time.monotonic()
        index = 0
        while self.running:
            t = time.monotonic() - t0
            if index >= len(snapshots):
                index = 0
                t0 = time.monotonic()
            snap = snapshots[index]
            payload = _dict_from_snapshot(snap, wind_x=wind_x, wind_std=wind_std)
            self.telemetry_updated.emit(payload)
            index += 1
            self.msleep(int(interval_s * 1000))

    def stop(self) -> None:
        self.running = False
