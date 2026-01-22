# ============================================================
# Probabilistic Unpowered Payload Drop Simulation (MVP)
# Author: Saadee
# Description:
#   Physics-based Monte Carlo decision-support module for
#   unpowered payload airdrop under uncertainty.
# ============================================================

# ----------------------------
# 1. Import libraries
# ----------------------------
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 2. Define constants
# ----------------------------
g = 9.81            # gravity (m/s^2)
dt = 0.01           # time step (s)
mass = 1.0          # payload mass (kg)
Cd = 1.0            # drag coefficient (dimensionless)
rho = 1.225         # air density (kg/m^3)
A = 0.01            # reference area (m^2)

# ----------------------------
# 3. Initial conditions
# ----------------------------
# UAV state at release
uav_pos = np.array([0.0, 0.0, 100.0])     # x, y, z (m)
uav_vel = np.array([20.0, 0.0, 0.0])      # vx, vy, vz (m/s)

# Payload state at release
payload_pos0 = uav_pos.copy()
payload_vel0 = uav_vel.copy()

# Target definition
target_pos = np.array([72.0, 0.0])        # x, y (m)
target_radius = 5.0                        # acceptable radius (m)

# ----------------------------
# 4. Trajectory propagation
# ----------------------------
def propagate_trajectory(pos0, vel0, wind):
    """Propagate payload trajectory after release."""
    pos = pos0.copy()
    vel = vel0.copy()
    trajectory = []

    while pos[2] > 0:
        # Relative velocity
        v_rel = vel - wind
        v_rel_mag = np.linalg.norm(v_rel)

        # Quadratic drag
        if v_rel_mag > 0:
            drag_force = -0.5 * rho * Cd * A * v_rel_mag * v_rel
        else:
            drag_force = np.zeros(3)

        # Acceleration
        acc = np.array([0.0, 0.0, -g]) + drag_force / mass

        # Integrate (Euler)
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
# 6. Decision logic
# ----------------------------
def decision_logic(impact_points, target_pos, target_radius, threshold):
    distances = np.linalg.norm(impact_points - target_pos, axis=1)
    hits = np.sum(distances <= target_radius)
    probability = hits / len(impact_points)

    decision = "DROP" if probability >= threshold else "NO DROP"

    return decision, probability, probability * 100

# ----------------------------
# 7. Plotting utilities
# ----------------------------
def plot_results(impact_points, target_pos, target_radius, decision, probability_percent):
    plt.figure(figsize=(8, 8))

    plt.scatter(impact_points[:, 0], impact_points[:, 1], alpha=0.5, label="Impact Points")

    circle = plt.Circle(target_pos, target_radius, color='r', fill=False, linewidth=2, label="Target Area")
    plt.gca().add_patch(circle)

    plt.scatter(target_pos[0], target_pos[1], color='r')

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title(f"Decision: {decision} | Probability: {probability_percent:.1f}%")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show(block=False)

# ----------------------------
# 8. Sensitivity studies
# ----------------------------
def probability_vs_distance(distances, n_samples, wind_mean, wind_std, threshold):
    probs = []
    for d in distances:
        impacts = monte_carlo_simulation(n_samples, wind_mean, wind_std)
        _, _, p = decision_logic(impacts, np.array([d, 0.0]), target_radius, threshold)
        probs.append(p)
    return probs


def probability_vs_wind(wind_stds, n_samples, wind_mean, target_pos, threshold):
    probs = []
    for ws in wind_stds:
        impacts = monte_carlo_simulation(n_samples, wind_mean, ws)
        _, _, p = decision_logic(impacts, target_pos, target_radius, threshold)
        probs.append(p)
    return probs

# ----------------------------
# 9. Main execution
# ----------------------------
if __name__ == "__main__":

    n_samples = 300
    wind_mean = np.array([2.0, 0.0, 0.0])
    wind_std = 0.8
    threshold = 0.75

    # Core simulation
    impacts = monte_carlo_simulation(n_samples, wind_mean, wind_std)
    decision, prob, prob_percent = decision_logic(
        impacts, target_pos, target_radius, threshold
    )

    print(f"Decision: {decision}")
    print(f"Probability of Success: {prob:.2f} ({prob_percent:.1f}%)")

    plot_results(impacts, target_pos, target_radius, decision, prob_percent)

    # Sensitivity: distance
    distances = np.linspace(50, 100, 20)
    probs_dist = probability_vs_distance(distances, n_samples, wind_mean, wind_std, threshold)

    plt.figure()
    plt.plot(distances, probs_dist, marker='o')
    plt.xlabel("Target Distance (m)")
    plt.ylabel("Probability of Success (%)")
    plt.title("Probability vs Target Distance")
    plt.grid(True)
    plt.show(block=False)

    # Sensitivity: wind uncertainty
    wind_stds = np.linspace(0.1, 2.0, 15)
    probs_wind = probability_vs_wind(wind_stds, n_samples, wind_mean, target_pos, threshold)

    plt.figure()
    plt.plot(wind_stds, probs_wind, marker='s')
    plt.xlabel("Wind Uncertainty (m/s)")
    plt.ylabel("Probability of Success (%)")
    plt.title("Probability vs Wind Uncertainty")
    plt.grid(True)
    plt.show(block=False)
# Keep all figures open
plt.show()


