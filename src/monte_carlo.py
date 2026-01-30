# ============================================================
# Probabilistic Unpowered Payload Drop Simulation (MVP)
# AIRDROP-X
# Author: Saadee
# Description:
#   Physics-based Monte Carlo decision-support module for
#   unpowered payload airdrop under uncertainty.
#   Includes UI controls for user-defined threshold, decision modes,
#   CEP metric, and sensitivity visualizations.
# ============================================================

# ----------------------------
# 1. Import libraries
# ----------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# ----------------------------
# 2. Define constants
# ----------------------------
g = 9.81
dt = 0.01
mass = 1.0
Cd = 1.0
rho = 1.225
A = 0.01

# ----------------------------
# 3. Initial conditions
# ----------------------------
uav_pos = np.array([0.0, 0.0, 100.0])
uav_vel = np.array([20.0, 0.0, 0.0])

payload_pos0 = uav_pos.copy()
payload_vel0 = uav_vel.copy()

target_pos = np.array([72.0, 0.0])
target_radius = 5.0

# ----------------------------
# 4. Trajectory propagation
# ----------------------------
def propagate_trajectory(pos0, vel0, wind):
    pos = pos0.copy()
    vel = vel0.copy()
    trajectory = []

    while pos[2] > 0:
        v_rel = vel - wind
        v_rel_mag = np.linalg.norm(v_rel)

        if v_rel_mag > 0:
            drag_force = -0.5 * rho * Cd * A * v_rel_mag * v_rel
        else:
            drag_force = np.zeros(3)

        acc = np.array([0.0, 0.0, -g]) + drag_force / mass
        vel += acc * dt
        pos += vel * dt
        trajectory.append(pos.copy())

    return np.array(trajectory)

# ----------------------------
# 5. Monte Carlo simulation
# ----------------------------
def monte_carlo_simulation(n_samples, wind_mean, wind_std):
    impact_points = []
    for _ in range(n_samples):
        wind_sample = wind_mean + np.random.normal(0, wind_std, size=3)
        traj = propagate_trajectory(payload_pos0, payload_vel0, wind_sample)
        impact_points.append(traj[-1][:2])
    return np.array(impact_points)

# ----------------------------
# 6. Metrics & decision logic
# ----------------------------
def compute_hit_probability(impact_points, target_pos, target_radius):
    distances = np.linalg.norm(impact_points - target_pos, axis=1)
    hits = np.sum(distances <= target_radius)
    return hits / len(impact_points)


def compute_cep(impact_points, target_pos, percentile=50):
    distances = np.linalg.norm(impact_points - target_pos, axis=1)
    return np.percentile(distances, percentile)


def evaluate_drop_decision(P_hit, P_threshold):
    return "DROP" if P_hit >= P_threshold else "NO DROP"

# ----------------------------
# 7. Main execution + UI
# ----------------------------
if __name__ == "__main__":

    np.random.seed(42)

    n_samples = 300
    wind_mean = np.array([2.0, 0.0, 0.0])
    wind_std = 0.8

    impact_points = monte_carlo_simulation(n_samples, wind_mean, wind_std)
    P_hit = compute_hit_probability(impact_points, target_pos, target_radius)
    cep50 = compute_cep(impact_points, target_pos, 50)

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(left=0.25, bottom=0.3)

    ax.scatter(impact_points[:, 0], impact_points[:, 1], alpha=0.4)
    ax.add_patch(plt.Circle(target_pos, target_radius, color='r', fill=False))
    ax.scatter(target_pos[0], target_pos[1], color='r')

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.axis("equal")
    ax.grid(True)

    title = ax.set_title("")

    ax_slider = plt.axes([0.25, 0.15, 0.6, 0.03])
    threshold_slider = Slider(
        ax=ax_slider,
        label="Probability Threshold (%)",
        valmin=50,
        valmax=95,
        valinit=75,
        valstep=0.5
    )

    ax_radio = plt.axes([0.025, 0.4, 0.18, 0.2])
    mode_radio = RadioButtons(
        ax_radio,
        ('Conservative', 'Balanced', 'Aggressive'),
        active=1
    )

    mode_thresholds = {
        'Conservative': 0.90,
        'Balanced': 0.75,
        'Aggressive': 0.60
    }

    last_control = {'source': 'mode'}

    def update(val=None):
        selected_mode = mode_radio.value_selected

        if val is not None:
            last_control['source'] = 'slider'

        if last_control['source'] == 'mode':
            threshold_slider.eventson = False
            forced_threshold = mode_thresholds[selected_mode] * 100.0
            threshold_slider.set_val(forced_threshold)
            threshold_slider.eventson = True
            P_threshold = mode_thresholds[selected_mode]
        else:
            P_threshold = threshold_slider.val / 100.0

        decision = evaluate_drop_decision(P_hit, P_threshold)

        title.set_text(
            f"Mode: {selected_mode} | Decision: {decision} | "
            f"Target Hit % = {P_hit*100:.1f}% | Threshold = {P_threshold*100:.1f}% | CEP50 = {cep50:.2f} m"
        )
        fig.canvas.draw_idle()

    threshold_slider.on_changed(update)

    def on_mode_change(label):
        last_control['source'] = 'mode'
        update()

    mode_radio.on_clicked(on_mode_change)

    update()

    distances = np.linspace(50, 100, 20)
    probs_dist = []
    for d in distances:
        impacts = monte_carlo_simulation(n_samples, wind_mean, wind_std)
        probs_dist.append(compute_hit_probability(impacts, np.array([d, 0.0]), target_radius) * 100)

    plt.figure()
    plt.plot(distances, probs_dist, marker='o')
    plt.xlabel("Target Distance (m)")
    plt.ylabel("Probability of Success (%)")
    plt.title("Probability vs Target Distance")
    plt.grid(True)

    wind_stds = np.linspace(0.1, 2.0, 15)
    probs_wind = []
    for ws in wind_stds:
        impacts = monte_carlo_simulation(n_samples, wind_mean, ws)
        probs_wind.append(compute_hit_probability(impacts, target_pos, target_radius) * 100)

    plt.figure()
    plt.plot(wind_stds, probs_wind, marker='s')
    plt.xlabel("Wind Uncertainty (m/s)")
    plt.ylabel("Probability of Success (%)")
    plt.title("Probability vs Wind Uncertainty")
    plt.grid(True)

    plt.show()
