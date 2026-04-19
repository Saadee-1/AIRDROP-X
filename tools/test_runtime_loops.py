"""
Run SCYTHE loops for ~12 seconds and capture output to verify they run correctly.
"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import time
import numpy as np

from product.runtime.runtime_loops import (
    BackgroundPlannerLoop,
    GuidanceLoop,
    TelemetryLoop,
    UIRenderLoop,
)
from product.runtime.system_state import SystemState


def main():
    state = SystemState()
    state.target_position = np.array([72.0, 0.0, 0.0], dtype=float)
    state.settings["drop_probability_threshold"] = 0.5

    telemetry = TelemetryLoop(state, update_rate_hz=50.0)
    guidance = GuidanceLoop(state, update_rate_hz=12.0)
    ui = UIRenderLoop(state, update_rate_hz=30.0)
    planner = BackgroundPlannerLoop(state, update_rate_hz=1.0)

    loops = [telemetry, guidance, ui, planner]
    print("[Test] Starting loops for 12 seconds...")
    for loop in loops:
        loop.start()

    start = time.perf_counter()
    while state.running and (time.perf_counter() - start) < 12:
        time.sleep(0.5)

    state.running = False
    for loop in loops:
        loop.join(timeout=2.0)

    print("[Test] Done.")


if __name__ == "__main__":
    main()
