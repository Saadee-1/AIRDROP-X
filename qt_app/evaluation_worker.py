"""
Continuous real-time evaluation worker for live telemetry.
Runs Monte Carlo at ~5–7 Hz. No overlapping runs. No UI blocking.
"""
from __future__ import annotations

import copy
import threading
import time
from typing import Any

from PySide6.QtCore import QThread, Signal


class TelemetryState:
    """Thread-safe telemetry container. Main thread writes; worker reads."""
    __slots__ = ("lock", "data")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.data: dict[str, Any] = {}


class ConfigState:
    """Thread-safe config container. Main thread writes; worker reads."""
    __slots__ = ("lock", "data")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.data: dict[str, Any] = {}


class EvaluationWorker(QThread):
    """
    Continuous evaluation loop. Reads telemetry + config, runs Monte Carlo,
    emits atomic result packet. Target ~5–7 Hz.
    """

    result_ready = Signal(dict)

    def __init__(
        self,
        telemetry_state: TelemetryState,
        config_state: ConfigState,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.telemetry_state = telemetry_state
        self.config_state = config_state
        self.running = True
        self.target_period = 0.15  # ~6.6 Hz

    def run(self) -> None:
        while self.running:
            cycle_start = time.perf_counter()

            # 1) Freeze telemetry
            with self.telemetry_state.lock:
                telem_snapshot = copy.deepcopy(self.telemetry_state.data)

            # 2) Freeze configuration
            with self.config_state.lock:
                local_config = copy.deepcopy(self.config_state.data)

            # 3) Build config override (merge telemetry into config for LIVE)
            override = dict(local_config)
            if telem_snapshot:
                override["uav_x"] = float(telem_snapshot.get("x", override.get("uav_x", 0.0)))
                override["uav_y"] = float(telem_snapshot.get("y", override.get("uav_y", 0.0)))
                override["uav_altitude"] = float(telem_snapshot.get("z", override.get("uav_altitude", 100.0)))
                override["uav_vx"] = float(telem_snapshot.get("vx", override.get("uav_vx", 20.0)))
                override["uav_vy"] = float(telem_snapshot.get("vy", override.get("uav_vy", 0.0)))
                override["wind_x"] = float(telem_snapshot.get("wind_x", override.get("wind_x", 2.0)))
                override["wind_y"] = float(telem_snapshot.get("wind_y", override.get("wind_y", 0.0)))
                override["wind_std"] = float(telem_snapshot.get("wind_std", override.get("wind_std", 0.8)))

            # 4) Run Monte Carlo (via adapter; no advisory for speed)
            try:
                from adapter import run_simulation_snapshot

                prev_gradient = local_config.get("prev_wind_gradient")
                snapshot = run_simulation_snapshot(
                    config_override=override,
                    include_advisory=False,
                    previous_wind_gradient=prev_gradient,
                )
            except Exception:
                # Skip emit on error; next cycle will retry
                elapsed = time.perf_counter() - cycle_start
                sleep_time = max(0.0, self.target_period - elapsed)
                time.sleep(sleep_time)
                continue

            impact_points = snapshot.get("impact_points", [])
            hits = snapshot.get("hits")
            n_samples = snapshot.get("n_samples")
            if hits is None or n_samples is None:
                raise RuntimeError(
                    "Invariant violation: adapter must provide 'hits' and 'n_samples'."
                )
            hits = int(hits)
            n_samples = int(n_samples)
            P_hit = float(hits) / float(n_samples) if n_samples > 0 else 0.0
            p_hat = P_hit
            cep50 = float(snapshot.get("cep50", 0.0) or 0.0)
            # Threshold frozen from config snapshot only (no override drift)
            threshold_pct = float(local_config.get("threshold_pct", 75.0))
            doctrine = str(local_config.get("doctrine_mode", "BALANCED")).strip().upper()

            # 5) Wilson CI and doctrine-based decision (uses true integer hit count)
            from src.statistics import compute_wilson_ci
            from src.decision_doctrine import evaluate_doctrine, DOCTRINE_DESCRIPTIONS
            ci_low, ci_high = compute_wilson_ci(hits, n_samples)
            threshold_frac = threshold_pct / 100.0
            doctrine_result = evaluate_doctrine(
                p_hat=p_hat,
                ci_low=ci_low,
                ci_high=ci_high,
                threshold=threshold_frac,
                doctrine=doctrine,
                n_samples=n_samples,
            )
            decision = doctrine_result["decision"]
            decision_reason = doctrine_result["reason"]
            doctrine_description = doctrine_result.get("doctrine_description") or DOCTRINE_DESCRIPTIONS.get(doctrine, doctrine)

            # 6) Emit single atomic result packet (includes full telemetry snapshot for unified Tactical Map rendering)
            mission_mode = str(local_config.get("mission_mode", "TACTICAL")).strip().upper()
            if mission_mode not in ("TACTICAL", "HUMANITARIAN"):
                mission_mode = "TACTICAL"
            emit_data = {
                "snapshot_type": "EVALUATION",
                "timestamp": time.time(),
                "telemetry_snapshot": telem_snapshot,
                "impact_points": impact_points,
                "hits": hits,
                "P_hit": P_hit,
                "cep50": cep50,
                "decision": decision,
                "n_samples": n_samples,
                "target_position": snapshot.get("target_position"),
                "target_radius": snapshot.get("target_radius"),
                "confidence_index": snapshot.get("confidence_index"),
                "wind_vector": snapshot.get("wind_vector"),
                "random_seed": local_config.get("random_seed"),
                "threshold_pct": threshold_pct,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_hat": p_hat,
                "decision_reason": decision_reason,
                "doctrine_mode": doctrine,
                "doctrine_description": doctrine_description,
                "mission_mode": mission_mode,
            }
            if "sensitivity_live" in snapshot:
                emit_data["sensitivity_live"] = snapshot["sensitivity_live"]
            if "topology_live" in snapshot:
                emit_data["topology_live"] = snapshot["topology_live"]
            if "release_corridor_live" in snapshot:
                emit_data["release_corridor_live"] = snapshot["release_corridor_live"]
            if "fragility_state" in snapshot:
                emit_data["fragility_state"] = snapshot["fragility_state"]
            updated_grad = snapshot.get("updated_wind_gradient")
            if updated_grad is not None:
                emit_data["updated_wind_gradient"] = updated_grad
            self.result_ready.emit(emit_data)

            # 7) Maintain loop rate (AX-SENSITIVITY-STABILITY-AUDIT-10: log cycle time)
            elapsed = time.perf_counter() - cycle_start
            print("eval_cycle_ms:", round(elapsed * 1000.0, 2))
            sleep_time = max(0.0, self.target_period - elapsed)
            time.sleep(sleep_time)
