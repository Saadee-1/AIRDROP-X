"""
AIRDROP-X Web Application
Streamlit-based web interface for the AIRDROP-X simulation framework.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

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
from product.ui import plots
from product.ui.ui_theme import *

# Page configuration
st.set_page_config(
    page_title="AIRDROP-X Simulation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for military-grade dark theme
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {BG_MAIN};
        color: {TEXT_PRIMARY};
    }}
    .stSidebar {{
        background-color: {BG_PANEL};
    }}
    h1, h2, h3 {{
        color: {ACCENT_GO};
        font-family: {FONT_FAMILY};
    }}
    .stButton>button {{
        background-color: {BG_INPUT};
        color: {ACCENT_GO};
        border: 1px solid {BORDER_SUBTLE};
        font-family: {FONT_FAMILY};
    }}
    .stButton>button:hover {{
        background-color: {BUTTON_HOVER};
        border-color: {ACCENT_GO};
    }}
    </style>
    """, unsafe_allow_html=True)


def build_mission_state(payload_config=None):
    """Build MissionState from config or custom payload."""
    if payload_config:
        payload = Payload(
            mass=payload_config.get("mass", cfg.mass),
            drag_coefficient=payload_config.get("drag_coefficient", cfg.Cd),
            reference_area=payload_config.get("reference_area", cfg.A),
        )
    else:
        payload = Payload(mass=cfg.mass, drag_coefficient=cfg.Cd, reference_area=cfg.A)

    target = Target(position=cfg.target_pos, radius=cfg.target_radius)
    environment = Environment(wind_mean=cfg.wind_mean, wind_std=cfg.wind_std)

    return MissionState(
        payload=payload,
        target=target,
        environment=environment,
        uav_position=cfg.uav_pos,
        uav_velocity=cfg.uav_vel,
    )


def run_simulation(random_seed=None):
    """Run Monte Carlo simulation and return results."""
    if random_seed is None:
        random_seed = cfg.RANDOM_SEED

    mission_state = build_mission_state()
    impact_points, P_hit, cep50 = get_impact_points_and_metrics(
        mission_state, random_seed
    )
    advisory_result = evaluate_advisory(
        mission_state, "Balanced", random_seed=random_seed
    )

    return impact_points, P_hit, cep50, advisory_result, mission_state


def plot_impact_dispersion_fig(impact_points, target_position, target_radius, cep50):
    """Create impact dispersion figure."""
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG_PANEL)
    plots.plot_impact_dispersion(
        ax, impact_points, target_position, target_radius, cep50
    )
    ax.set_title("Impact Dispersion", color=TEXT_LABEL, fontsize=12)
    plt.tight_layout()
    return fig


def plot_decision_banner(decision, P_hit, cep50, threshold):
    """Create decision banner visualization."""
    fig, ax = plt.subplots(figsize=(10, 3), facecolor=BG_PANEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    is_drop = decision == "DROP"
    color = ACCENT_GO if is_drop else ACCENT_NO_GO

    # Decision banner
    ax.axhspan(0.3, 0.7, xmin=0.05, xmax=0.95,
               facecolor=color, alpha=0.2)
    ax.text(0.5, 0.5, decision,
            ha='center', va='center',
            fontsize=32, fontweight='bold',
            color=color, family='monospace')

    # Stats
    ax.text(0.5, 0.15,
            f"HIT {P_hit*100:.1f}%  |  THRESH {threshold:.1f}%  |  CEP50 {cep50:.2f}m",
            ha='center', va='center',
            fontsize=11, color=TEXT_PRIMARY, family='monospace')

    plt.tight_layout()
    return fig


def main():
    # Header
    st.title("🎯 AIRDROP-X Simulation")
    st.markdown("**Probabilistic guidance and decision-support simulation for precision unpowered payload delivery**")
    st.markdown("---")

    # Sidebar - Mission Parameters
    with st.sidebar:
        st.header("⚙️ Mission Parameters")

        st.subheader("Payload")
        mass = st.number_input("Mass (kg)", value=float(cfg.mass), min_value=0.1, step=0.1)
        Cd = st.number_input("Drag Coefficient", value=float(cfg.Cd), min_value=0.1, step=0.01)
        A = st.number_input("Reference Area (m²)", value=float(cfg.A), min_value=0.001, step=0.001, format="%.4f")

        st.subheader("UAV State")
        uav_x = st.number_input("UAV X (m)", value=float(cfg.uav_pos[0]), step=1.0)
        uav_y = st.number_input("UAV Y (m)", value=float(cfg.uav_pos[1]), step=1.0)
        uav_z = st.number_input("UAV Altitude (m)", value=float(cfg.uav_pos[2]), min_value=10.0, step=10.0)
        uav_vx = st.number_input("UAV Velocity X (m/s)", value=float(cfg.uav_vel[0]), step=1.0)

        st.subheader("Target")
        target_x = st.number_input("Target X (m)", value=float(cfg.target_pos[0]), step=1.0)
        target_y = st.number_input("Target Y (m)", value=float(cfg.target_pos[1]), step=1.0)
        target_radius = st.number_input("Target Radius (m)", value=float(cfg.target_radius), min_value=0.1, step=0.5)

        st.subheader("Environment")
        wind_x = st.number_input("Wind X (m/s)", value=float(cfg.wind_mean[0]), step=0.5)
        wind_std = st.number_input("Wind Std Dev (m/s)", value=float(cfg.wind_std), min_value=0.0, step=0.1)

        st.subheader("Simulation")
        n_samples = st.number_input("Monte Carlo Samples", value=cfg.n_samples, min_value=50, max_value=1000, step=50)
        random_seed = st.number_input("Random Seed", value=cfg.RANDOM_SEED, min_value=0, step=1)

        st.markdown("---")
        run_button = st.button("🚀 RUN SIMULATION", use_container_width=True)

        st.subheader("Decision Threshold")
        threshold_pct = st.slider(
            "Probability Threshold (%)",
            min_value=int(cfg.THRESHOLD_SLIDER_MIN),
            max_value=int(cfg.THRESHOLD_SLIDER_MAX),
            value=int(cfg.THRESHOLD_SLIDER_INIT),
            step=1
        )

    # Update config with sidebar values
    cfg.mass = mass
    cfg.Cd = Cd
    cfg.A = A
    cfg.uav_pos = (uav_x, uav_y, uav_z)
    cfg.uav_vel = (uav_vx, 0.0, 0.0)
    cfg.target_pos = (target_x, target_y)
    cfg.target_radius = target_radius
    cfg.wind_mean = (wind_x, 0.0, 0.0)
    cfg.wind_std = wind_std
    cfg.n_samples = n_samples
    cfg.RANDOM_SEED = random_seed

    # Run simulation on button press or initial load
    if run_button or 'results' not in st.session_state:
        with st.spinner('Running Monte Carlo simulation...'):
            impact_points, P_hit, cep50, advisory_result, mission_state = run_simulation(random_seed)
            st.session_state.results = {
                'impact_points': impact_points,
                'P_hit': P_hit,
                'cep50': cep50,
                'advisory_result': advisory_result,
                'mission_state': mission_state,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        st.success(f'✓ Simulation complete - {n_samples} samples')

    # Display results
    if 'results' in st.session_state:
        results = st.session_state.results
        P_hit = results['P_hit']
        cep50 = results['cep50']
        impact_points = results['impact_points']
        advisory_result = results['advisory_result']

        # Evaluate decision with current threshold
        threshold = threshold_pct / 100.0
        decision = decision_logic.evaluate_drop_decision(P_hit, threshold)

        # Snapshot info
        st.info(f"📸 Snapshot: {results['timestamp']} | Seed: {random_seed} | Samples: {n_samples}")

        # Decision Banner
        st.subheader("Mission Advisory")
        fig_banner = plot_decision_banner(decision, P_hit, cep50, threshold_pct)
        st.pyplot(fig_banner)
        plt.close(fig_banner)

        # Metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Decision", decision)
        with col2:
            st.metric("Hit Probability", f"{P_hit*100:.1f}%")
        with col3:
            st.metric("CEP50", f"{cep50:.2f} m")
        with col4:
            st.metric("Threshold", f"{threshold_pct}%")

        st.markdown("---")

        # Visualization tabs
        tab1, tab2, tab3 = st.tabs(["📊 Impact Dispersion", "📈 Advisory Analysis", "ℹ️ System Info"])

        with tab1:
            st.subheader("Impact Point Distribution")
            fig_impact = plot_impact_dispersion_fig(
                impact_points, cfg.target_pos, cfg.target_radius, cep50
            )
            st.pyplot(fig_impact)
            plt.close(fig_impact)

            st.markdown(f"""
            **Analysis:**
            - Total samples: {len(impact_points)}
            - Hits within target: {int(P_hit * len(impact_points))}
            - CEP50 (median miss distance): {cep50:.2f} m
            - Target radius: {cfg.target_radius:.2f} m
            """)

        with tab2:
            st.subheader("Advisory Guidance")
            st.write(f"**Current Feasibility:** {advisory_result.current_feasibility}")
            st.write(f"**Hit Probability:** {advisory_result.current_P_hit*100:.2f}%")
            st.write(f"**CEP50:** {advisory_result.current_cep50_m:.2f} m")

            st.markdown("**Trend Analysis:**")
            st.write(advisory_result.trend_summary)

            st.markdown("**Directional Guidance:**")
            st.write(advisory_result.suggested_direction)

            if advisory_result.improvement_directions:
                st.success(f"Improvement directions: {', '.join(advisory_result.improvement_directions)}")

            if advisory_result.degradation_directions:
                st.warning(f"Degradation directions: {', '.join(advisory_result.degradation_directions)}")

        with tab3:
            st.subheader("System Information")
            st.markdown(f"""
            **Physics Model:**
            - 3-DOF point-mass dynamics
            - Gravity: {cfg.g} m/s²
            - Air density: {cfg.rho} kg/m³
            - Time step: {cfg.dt} s

            **Current Configuration:**
            - Payload mass: {cfg.mass} kg
            - Drag coefficient: {cfg.Cd}
            - Reference area: {cfg.A} m²
            - UAV position: ({cfg.uav_pos[0]:.1f}, {cfg.uav_pos[1]:.1f}, {cfg.uav_pos[2]:.1f}) m
            - UAV velocity: ({cfg.uav_vel[0]:.1f}, {cfg.uav_vel[1]:.1f}, {cfg.uav_vel[2]:.1f}) m/s
            - Target position: ({cfg.target_pos[0]:.1f}, {cfg.target_pos[1]:.1f}) m
            - Target radius: {cfg.target_radius} m
            - Wind mean: ({cfg.wind_mean[0]:.1f}, {cfg.wind_mean[1]:.1f}, {cfg.wind_mean[2]:.1f}) m/s
            - Wind std: {cfg.wind_std} m/s

            **Reproducibility:**
            - Random seed: {random_seed}
            - Monte Carlo samples: {n_samples}
            - Snapshot created: {results['timestamp']}

            **Limitations:**
            - Point-mass payload (no rotation/attitude dynamics)
            - 2D target (X-Y plane, impact at Z=0)
            - Uniform atmosphere (constant density, no wind shear)
            - Gaussian wind uncertainty (isotropic, no correlation)
            - Single release point per run
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6b8e6b; font-size: 0.9em;'>
    AIRDROP-X v1.0 | Probabilistic guidance simulation for unpowered payload delivery<br>
    For research and simulation purposes only
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
