import sys
import os
import subprocess

from configs import mission_configs as cfg
from src import decision_logic
from product.payloads.payload_base import Payload
from product.missions.target_manager import Target
from product.missions.environment import Environment
from product.missions.mission_state import MissionState
from product.guidance.advisory_layer import (
    get_impact_points_and_metrics,
    evaluate_advisory,
)
from product.ui.ui_layout import launch_unified_ui


def _build_mission_state(payload_config=None):
    """
    Build a MissionState from config defaults, optionally overriding payload
    with a custom dict from the Dynamic Payload Builder.
    """
    if payload_config:
        payload = Payload(
            mass=payload_config.get("mass", cfg.mass),
            drag_coefficient=payload_config.get("drag_coefficient", cfg.Cd),
            reference_area=payload_config.get("reference_area", cfg.A),
        )
    else:
        payload = Payload(
            mass=cfg.mass,
            drag_coefficient=cfg.Cd,
            reference_area=cfg.A,
        )

    target = Target(position=cfg.target_pos, radius=cfg.target_radius)
    environment = Environment(wind_mean=cfg.wind_mean, wind_std=cfg.wind_std)

    return MissionState(
        payload=payload,
        target=target,
        environment=environment,
        uav_position=cfg.uav_pos,
        uav_velocity=cfg.uav_vel,
    )


def run_simulation(payload_config=None):
    """
    Run the full simulation loop.
    If payload_config is provided (dict from Payload Builder), use it.
    Otherwise, use defaults from configs.mission_configs.
    Returns (impact_points, advisory_result).
    """
    print(f"\n--- Starting Simulation ---")
    if payload_config:
        print(f"Using Custom Payload: {payload_config.get('name', 'Unknown')}")

    mission_state = _build_mission_state(payload_config)

    # Run Monte Carlo + metrics
    impact_points, P_hit, cep50 = get_impact_points_and_metrics(
        mission_state, cfg.RANDOM_SEED
    )

    # Evaluate advisory (directional analysis)
    advisory_result = evaluate_advisory(
        mission_state, "Balanced", random_seed=cfg.RANDOM_SEED
    )

    print(f"  -> CEP50: {cep50:.2f} m")
    print(f"  -> P(Hit): {P_hit*100:.1f} %")
    print(f"  -> Advisory: {advisory_result.current_feasibility}")

    return impact_points, advisory_result, P_hit, cep50


def _wait_for_streamlit(port=8501, timeout=30):
    """Poll until Streamlit server responds."""
    import urllib.request
    import time
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=2)
            return True
        except Exception:
            time.sleep(0.3)
    return False


def _launch_desktop_ui():
    """Launch Streamlit UI in a desktop window — same look as browser, native window."""
    import webview

    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    if not os.path.isfile(app_path):
        print("app.py not found.")
        return False

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.headless", "true",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        if not _wait_for_streamlit(8501):
            proc.terminate()
            print("Streamlit server did not start in time.")
            return False

        webview.create_window("AIRDROP-X", "http://127.0.0.1:8501", width=1280, height=800)
        webview.start()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    return True


def _launch_browser_ui():
    """Launch Streamlit in default browser (for debugging)."""
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    if not os.path.isfile(app_path):
        print("app.py not found.")
        return False
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    return True


def main():
    """Entry point: run simulation and launch UI.
    Default: desktop window with Streamlit UI (same as browser).
    Use --browser to open in default browser. Use --matplotlib for legacy Matplotlib UI.
    """
    use_matplotlib = "--matplotlib" in sys.argv
    use_browser = "--browser" in sys.argv

    if use_matplotlib:
        impacts, adv, p_hit, cep50 = run_simulation()
        launch_unified_ui(
            impact_points=impacts,
            P_hit=p_hit,
            cep50=cep50,
            target_position=cfg.target_pos,
            target_radius=cfg.target_radius,
            mission_state=None,
            advisory_result=adv,
            initial_threshold_percent=cfg.THRESHOLD_SLIDER_INIT,
            initial_mode="standard",
            slider_min=cfg.THRESHOLD_SLIDER_MIN,
            slider_max=cfg.THRESHOLD_SLIDER_MAX,
            slider_step=cfg.THRESHOLD_SLIDER_STEP,
            mode_thresholds=cfg.MODE_THRESHOLDS,
            on_threshold_change=lambda x: None,
            random_seed=cfg.RANDOM_SEED,
            n_samples=cfg.n_samples,
            dt=cfg.dt,
            run_simulation_callback=run_simulation,
        )
    elif use_browser:
        print("Launching AIRDROP-X in browser.")
        _launch_browser_ui()
    else:
        print("Launching AIRDROP-X in desktop window (same UI as browser). Use --browser for browser, --matplotlib for legacy UI.")
        _launch_desktop_ui()


if __name__ == "__main__":
    main()
