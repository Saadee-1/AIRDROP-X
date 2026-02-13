"""
Advisory guidance only. No waypoints, no control commands, no optimization.
Uses the engine as a black box to evaluate drop feasibility.
"""

import numpy as np

# Position step for nearby feasibility checks (m). SI.
ADVISORY_POSITION_DELTA_M = 5.0

# Keys we temporarily set on config when running the engine with MissionState inputs.
_ENGINE_INPUT_KEYS = (
    "uav_pos",
    "uav_vel",
    "target_pos",
    "target_radius",
    "wind_mean",
    "wind_std",
    "mass",
    "A",
    "Cd",
)


def _run_engine_for_inputs(engine_inputs, probability_threshold, random_seed):
    """
    Run engine once with the given scenario. Temporarily applies engine_inputs to config.
    Returns (P_hit, decision, cep50). Restores config in a finally block.
    """
    from configs import mission_configs as cfg
    from src import monte_carlo
    from src import metrics
    from src import decision_logic

    saved = {}
    try:
        for key in _ENGINE_INPUT_KEYS:
            saved[key] = getattr(cfg, key)
        for key, value in engine_inputs.items():
            setattr(cfg, key, value)
        impact_points = monte_carlo.run_monte_carlo(
            cfg.uav_pos,
            cfg.uav_vel,
            cfg.mass,
            cfg.Cd,
            cfg.A,
            cfg.rho,
            cfg.wind_mean,
            cfg.wind_std,
            cfg.n_samples,
            random_seed,
            dt=cfg.dt,
        )
        target_pos = engine_inputs["target_pos"]
        target_radius = engine_inputs["target_radius"]
        P_hit = metrics.compute_hit_probability(
            impact_points, target_pos, target_radius
        )
        cep50 = metrics.compute_cep50(impact_points, target_pos)
        decision = decision_logic.evaluate_drop_decision(
            P_hit, probability_threshold
        )
        return (P_hit, decision, cep50)
    finally:
        for key, value in saved.items():
            setattr(cfg, key, value)


def get_impact_points_and_metrics(mission_state, random_seed):
    """
    Run engine once for the given mission state; return impact points and metrics.
    For use by product entry point (e.g. UI).
    Returns (impact_points, P_hit, cep50, impact_velocity_stats).
    impact_velocity_stats: dict with mean_impact_speed, std_impact_speed, p95_impact_speed (m/s).
    """
    from configs import mission_configs as cfg
    from src import monte_carlo
    from src import metrics

    mission_state.validate()
    engine_inputs = mission_state.export_engine_inputs()
    saved = {}
    try:
        for key in _ENGINE_INPUT_KEYS:
            saved[key] = getattr(cfg, key)
        for key, value in engine_inputs.items():
            setattr(cfg, key, value)
        impact_points, impact_speeds = monte_carlo.run_monte_carlo(
            cfg.uav_pos,
            cfg.uav_vel,
            cfg.mass,
            cfg.Cd,
            cfg.A,
            cfg.rho,
            cfg.wind_mean,
            cfg.wind_std,
            cfg.n_samples,
            random_seed,
            dt=cfg.dt,
            return_impact_speeds=True,
        )
        target_pos = engine_inputs["target_pos"]
        target_radius = engine_inputs["target_radius"]
        P_hit = metrics.compute_hit_probability(
            impact_points, target_pos, target_radius
        )
        cep50 = metrics.compute_cep50(impact_points, target_pos)
        impact_velocity_stats = metrics.compute_impact_velocity_stats(impact_speeds)
        return (impact_points, P_hit, cep50, impact_velocity_stats)
    finally:
        for key, value in saved.items():
            setattr(cfg, key, value)


def _resolve_threshold(decision_policy):
    """decision_policy: float (threshold in [0,1]) or str ('Conservative'|'Balanced'|'Aggressive')."""
    if isinstance(decision_policy, (int, float)):
        t = float(decision_policy)
        if not (0 <= t <= 1):
            raise ValueError("probability_threshold must be in [0, 1]")
        return t
    if isinstance(decision_policy, str):
        from configs import mission_configs as cfg
        return cfg.MODE_THRESHOLDS[decision_policy.strip()]
    raise TypeError("decision_policy must be a float (threshold) or a mode name string")


class AdvisoryResult:
    """
    Advisory output for human interpretation. No control actions.
    """

    def __init__(
        self,
        current_feasibility,
        current_P_hit,
        current_cep50_m,
        trend_summary,
        suggested_direction,
        improvement_directions,
        degradation_directions,
    ):
        self.current_feasibility = current_feasibility
        self.current_P_hit = current_P_hit
        self.current_cep50_m = current_cep50_m
        self.trend_summary = trend_summary
        self.suggested_direction = suggested_direction
        self.improvement_directions = tuple(improvement_directions)
        self.degradation_directions = tuple(degradation_directions)


def evaluate_advisory(
    mission_state,
    decision_policy,
    random_seed=42,
    position_delta_m=None,
):
    """
    Evaluate drop feasibility at current UAV state and at small position variations.
    Returns an AdvisoryResult: feasibility at current position, relative trend nearby,
    and a qualitative suggested direction. No waypoints or commands.

    mission_state: MissionState instance (payload, target, environment, UAV state).
    decision_policy: float in [0, 1] (probability threshold) or str ('Conservative'|'Balanced'|'Aggressive').
    random_seed: int, for deterministic engine runs.
    position_delta_m: float, step for position variations (m). Default ADVISORY_POSITION_DELTA_M.
    """
    probability_threshold = _resolve_threshold(decision_policy)
    if position_delta_m is None:
        position_delta_m = ADVISORY_POSITION_DELTA_M
    delta = float(position_delta_m)
    if delta <= 0:
        raise ValueError("position_delta_m must be positive")

    mission_state.validate()
    base_inputs = mission_state.export_engine_inputs()
    uav_pos = list(base_inputs["uav_pos"])

    P_hit_current, decision_current, cep50_current = _run_engine_for_inputs(
        base_inputs, probability_threshold, random_seed
    )

    forward_inputs = dict(base_inputs)
    forward_inputs["uav_pos"] = (uav_pos[0] + delta, uav_pos[1], uav_pos[2])
    P_forward, _, _ = _run_engine_for_inputs(
        forward_inputs, probability_threshold, random_seed
    )

    backward_inputs = dict(base_inputs)
    backward_inputs["uav_pos"] = (uav_pos[0] - delta, uav_pos[1], uav_pos[2])
    P_backward, _, _ = _run_engine_for_inputs(
        backward_inputs, probability_threshold, random_seed
    )

    right_inputs = dict(base_inputs)
    right_inputs["uav_pos"] = (uav_pos[0], uav_pos[1] + delta, uav_pos[2])
    P_right, _, _ = _run_engine_for_inputs(
        right_inputs, probability_threshold, random_seed
    )

    left_inputs = dict(base_inputs)
    left_inputs["uav_pos"] = (uav_pos[0], uav_pos[1] - delta, uav_pos[2])
    P_left, _, _ = _run_engine_for_inputs(
        left_inputs, probability_threshold, random_seed
    )

    best_P = max(P_forward, P_backward, P_right, P_left)
    worst_P = min(P_forward, P_backward, P_right, P_left)

    if best_P > P_hit_current and worst_P < P_hit_current:
        trend_summary = (
            "Feasibility varies with direction. Some nearby positions are "
            "better, some worse."
        )
    elif best_P > P_hit_current:
        trend_summary = (
            "Feasibility improves at least in one direction from the current "
            "position."
        )
    elif worst_P < P_hit_current:
        trend_summary = (
            "Feasibility degrades in all sampled directions from the current "
            "position."
        )
    else:
        trend_summary = (
            "Feasibility is similar in all sampled directions; no strong "
            "trend."
        )

    improvement_directions = []
    degradation_directions = []
    if P_forward > P_hit_current:
        improvement_directions.append("forward (positive X)")
    elif P_forward < P_hit_current:
        degradation_directions.append("forward (positive X)")
    if P_backward > P_hit_current:
        improvement_directions.append("backward (negative X)")
    elif P_backward < P_hit_current:
        degradation_directions.append("backward (negative X)")
    if P_right > P_hit_current:
        improvement_directions.append("right (positive Y)")
    elif P_right < P_hit_current:
        degradation_directions.append("right (positive Y)")
    if P_left > P_hit_current:
        improvement_directions.append("left (negative Y)")
    elif P_left < P_hit_current:
        degradation_directions.append("left (negative Y)")

    if improvement_directions:
        suggested_direction = (
            "Consider moving "
            + " or ".join(improvement_directions)
            + " for higher hit probability."
        )
    else:
        suggested_direction = (
            "No strong directional recommendation from nearby samples; current "
            "position is among the best."
        )

    return AdvisoryResult(
        current_feasibility=decision_current,
        current_P_hit=P_hit_current,
        current_cep50_m=cep50_current,
        trend_summary=trend_summary,
        suggested_direction=suggested_direction,
        improvement_directions=improvement_directions,
        degradation_directions=degradation_directions,
    )
