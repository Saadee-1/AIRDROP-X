"""
AIRDROP-X Web Application
Streamlit-based web interface — full parity with desktop matplotlib UI.
5 tabs: Mission Overview, Payload Library, Sensor & Telemetry, Analysis, System Status.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime

from configs import mission_configs as cfg
from src import decision_logic
from src import metrics
from product.payloads.payload_base import Payload
from product.missions.target_manager import Target
from product.missions.environment import Environment
from product.missions.mission_state import MissionState
from product.guidance.advisory_layer import (
    get_impact_points_and_metrics,
    evaluate_advisory,
)
from product.guidance.numerical_diagnostics import quick_stability_check
from product.ui import plots
from product.ui.ui_theme import (
    BG_MAIN,
    BG_PANEL,
    BG_INPUT,
    TEXT_PRIMARY,
    TEXT_LABEL,
    ACCENT_GO,
    ACCENT_WARN,
    ACCENT_NO_GO,
    BORDER_SUBTLE,
    FONT_FAMILY,
    BUTTON_HOVER,
)
from product.ui.tabs.payload_library import PAYLOAD_LIBRARY, CATEGORIES

# ─── Snapshot ID ──────────────────────────────────────────────────────────────
SNAPSHOT_ID = f"AX-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AIRDROP-X Simulation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS — Military-grade dark theme ───────────────────────────────────
st.markdown(f"""
<style>
    /* ── Global ── */
    .stApp {{
        background-color: {BG_MAIN};
        color: {TEXT_PRIMARY};
        font-family: {FONT_FAMILY};
    }}
    [data-testid="stSidebar"] {{
        background-color: {BG_PANEL};
        border-right: 1px solid {BORDER_SUBTLE};
    }}

    /* ── Headers ── */
    h1, h2, h3, h4, h5, h6 {{
        color: {ACCENT_GO} !important;
        font-family: {FONT_FAMILY} !important;
    }}

    /* ── Text ── */
    p, li, span, label, div {{
        font-family: {FONT_FAMILY};
    }}
    .stMarkdown {{
        font-family: {FONT_FAMILY};
    }}

    /* ── Tabs ── */
    [data-testid="stTabs"] button {{
        background-color: {BG_PANEL} !important;
        color: {TEXT_LABEL} !important;
        border: 1px solid {BORDER_SUBTLE} !important;
        font-family: {FONT_FAMILY} !important;
        font-size: 0.85rem;
        padding: 8px 16px;
    }}
    [data-testid="stTabs"] button[aria-selected="true"] {{
        background-color: {BG_INPUT} !important;
        color: {ACCENT_GO} !important;
        border-bottom: 2px solid {ACCENT_GO} !important;
    }}

    /* ── Buttons ── */
    .stButton > button {{
        background-color: {BG_INPUT};
        color: {ACCENT_GO};
        border: 1px solid {BORDER_SUBTLE};
        font-family: {FONT_FAMILY};
    }}
    .stButton > button:hover {{
        background-color: {BUTTON_HOVER};
        border-color: {ACCENT_GO};
    }}

    /* ── Inputs ── */
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] > div {{
        background-color: {BG_INPUT} !important;
        color: {TEXT_PRIMARY} !important;
        border-color: {BORDER_SUBTLE} !important;
        font-family: {FONT_FAMILY} !important;
    }}

    /* ── Metrics ── */
    [data-testid="stMetric"] {{
        background-color: {BG_PANEL};
        border: 1px solid {BORDER_SUBTLE};
        border-radius: 4px;
        padding: 12px;
    }}
    [data-testid="stMetricLabel"] {{
        color: {TEXT_LABEL} !important;
        font-family: {FONT_FAMILY} !important;
        font-size: 0.75rem !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {ACCENT_GO} !important;
        font-family: {FONT_FAMILY} !important;
    }}

    /* ── Slider ── */
    .stSlider [data-baseweb="slider"] {{
        font-family: {FONT_FAMILY};
    }}

    /* ── Expander ── */
    [data-testid="stExpander"] {{
        background-color: {BG_PANEL};
        border: 1px solid {BORDER_SUBTLE};
    }}
    [data-testid="stExpander"] summary {{
        color: {ACCENT_GO} !important;
        font-family: {FONT_FAMILY} !important;
    }}

    /* ── Info/Success/Warning boxes ── */
    .stAlert {{
        background-color: {BG_PANEL};
        border-color: {BORDER_SUBTLE};
        font-family: {FONT_FAMILY};
    }}

    /* ── Footer ── */
    .footer-text {{
        text-align: center;
        color: {TEXT_LABEL};
        font-size: 0.85em;
        font-family: {FONT_FAMILY};
        padding: 20px 0;
    }}

    /* ── Panel card ── */
    .panel-card {{
        background-color: {BG_PANEL};
        border: 1px solid {BORDER_SUBTLE};
        border-radius: 4px;
        padding: 16px;
        margin: 4px 0;
    }}
    .panel-title {{
        color: {ACCENT_GO};
        font-family: {FONT_FAMILY};
        font-size: 0.85rem;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .panel-row {{
        display: flex;
        justify-content: space-between;
        padding: 3px 0;
        font-family: {FONT_FAMILY};
        font-size: 0.82rem;
    }}
    .panel-label {{
        color: {TEXT_LABEL};
    }}
    .panel-value {{
        color: {TEXT_PRIMARY};
    }}
    .panel-value-accent {{
        color: {ACCENT_GO};
    }}
    .panel-value-warn {{
        color: {ACCENT_WARN};
    }}

    /* ── Decision banner ── */
    .decision-banner {{
        text-align: center;
        padding: 20px;
        border-radius: 4px;
        margin: 8px 0;
        font-family: {FONT_FAMILY};
        background-color: {BG_PANEL}; /* Neutral dark panel */
    }}
    .decision-drop {{
        border: 2px solid {ACCENT_GO};
        color: {ACCENT_GO};
    }}
    .decision-nodrop {{
        border: 2px solid {ACCENT_NO_GO};
        color: {ACCENT_NO_GO};
    }}
    .decision-text {{
        font-size: 2.2rem; /* Slightly reduced */
        font-weight: bold;
        font-family: {FONT_FAMILY};
        letter-spacing: 4px;
    }}
    .decision-stats {{
        font-size: 0.85rem;
        color: {TEXT_LABEL}; /* Monospace smaller */
        font-family: {FONT_FAMILY};
        margin-top: 8px;
    }}
    .confidence-text {{
        font-size: 0.9rem;
        color: {TEXT_PRIMARY};
        font-family: {FONT_FAMILY};
        margin-top: 4px;
        font-weight: bold;
    }}

    /* ── Badges ── */
    .regime-badge {{
        position: fixed;
        top: 60px;
        right: 20px;
        background-color: rgba(10, 12, 10, 0.8);
        border: 1px solid {BORDER_SUBTLE};
        color: {TEXT_LABEL};
        padding: 4px 8px;
        font-size: 0.7rem;
        font-family: {FONT_FAMILY};
        z-index: 999;
        border-radius: 2px;
        letter-spacing: 1px;
    }}
    .snapshot-badge {{
        background-color: {ACCENT_WARN};
        color: #000;
        padding: 2px 6px;
        border-radius: 2px;
        font-size: 0.7rem;
        font-weight: bold;
        margin-bottom: 8px;
        display: inline-block;
    }}
    .user-def-badge {{
        background-color: {BORDER_SUBTLE};
        color: {TEXT_LABEL};
        padding: 2px 4px;
        font-size: 0.65rem;
        border-radius: 2px;
        margin-left: 6px;
    }}
</style>
""", unsafe_allow_html=True)

# ── Global Regime Badge ──
st.markdown('<div class="regime-badge">LOW-SUBSONIC | DRAG-DOMINATED</div>', unsafe_allow_html=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────

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
    impact_points, P_hit, cep50, impact_velocity_stats = get_impact_points_and_metrics(
        mission_state, random_seed
    )
    advisory_result = evaluate_advisory(
        mission_state, "Balanced", random_seed=random_seed
    )

    m = mission_state.payload.mass
    cd = mission_state.payload.drag_coefficient
    area = mission_state.payload.reference_area
    bc = (m / (cd * area)) if (cd and area) else None
    altitude = mission_state.uav_position[2]
    confidence_index = metrics.compute_confidence_index(
        wind_std=cfg.wind_std,
        ballistic_coefficient=bc,
        altitude=altitude,
        telemetry_freshness=None,
    )

    return (
        impact_points,
        P_hit,
        cep50,
        advisory_result,
        mission_state,
        impact_velocity_stats,
        confidence_index,
    )


def compute_reference_area(shape, dims):
    """Compute reference area from shape and dimensions."""
    if shape == "sphere":
        r = dims.get("radius", 0)
        return math.pi * r * r
    elif shape in ["cylinder", "capsule", "blunt_cone"]:
        r = dims.get("radius", 0)
        return math.pi * r * r
    elif shape == "box":
        l = dims.get("length", 0)
        w = dims.get("width", 0)
        return l * w
    return 0.0


def compute_bc(mass, cd, area):
    """Compute ballistic coefficient."""
    if cd and area and cd > 0 and area > 0:
        return mass / (cd * area)
    return None


def _fmt(v, fmt_str="{:.2f}", unit=""):
    """Format a value or return —."""
    if v is None:
        return "—"
    try:
        return fmt_str.format(float(v)) + unit
    except (TypeError, ValueError):
        return "—"

def _analytical_advisory(advisory):
    """Convert directive advisory to analytical phrasing."""
    direction = advisory.suggested_direction
    if not direction or direction == "Hold Position":
        return "Position optimal."
    
    # Map directive -> analytical
    mapping = {
        "Move Forward": "Probability gradient suggests +X region.",
        "Move Backward": "Feasibility improves in negative X direction.",
        "Move Left": "Feasibility improves in negative Y direction.",
        "Move Right": "Probability gradient suggests +Y region.",
        "Unsafe wind conditions": "Environmental uncertainty exceeds limits."
    }
    # Simple heuristic fallback
    for k, v in mapping.items():
        if k in direction:
            return v
    
    # Generic fallback
    return f"Gradient suggests shift: {direction}"

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown(f"""
    <div style="text-align: center; padding: 8px 0 4px 0;">
        <span style="color: {ACCENT_GO}; font-size: 1.6rem; font-family: {FONT_FAMILY}; letter-spacing: 3px;">
            🎯 AIRDROP-X
        </span>
        <br/>
        <span style="color: {TEXT_LABEL}; font-size: 0.75rem; font-family: {FONT_FAMILY};">
            Probabilistic guidance simulation for unpowered payload delivery
        </span>
        <br/>
        <span style="color: {TEXT_LABEL}; font-size: 0.65rem; font-family: {FONT_FAMILY}; opacity: 0.7;">
            {SNAPSHOT_ID}
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Sidebar — Mission Parameters ──
    with st.sidebar:
        st.markdown(f'<div style="color:{ACCENT_GO}; font-family:{FONT_FAMILY}; font-size:1rem; letter-spacing:2px;">⚙ MISSION PARAMETERS</div>', unsafe_allow_html=True)

        st.markdown(f'<div style="color:{TEXT_LABEL}; font-family:{FONT_FAMILY}; font-size:0.8rem; margin-top:12px;">PAYLOAD</div>', unsafe_allow_html=True)
        mass = st.number_input("Mass (kg)", value=float(cfg.mass), min_value=0.1, step=0.1)
        Cd = st.number_input("Drag Coefficient", value=float(cfg.Cd), min_value=0.01, step=0.01)
        if Cd != 0.47: # Check against default or previous context
             st.markdown(f'<span class="user-def-badge">User-defined assumption</span>', unsafe_allow_html=True)

        A = st.number_input("Reference Area (m²)", value=float(cfg.A), min_value=0.001, step=0.001, format="%.4f")

        st.markdown(f'<div style="color:{TEXT_LABEL}; font-family:{FONT_FAMILY}; font-size:0.8rem; margin-top:12px;">UAV STATE</div>', unsafe_allow_html=True)
        uav_x = st.number_input("UAV X (m)", value=float(cfg.uav_pos[0]), step=1.0)
        uav_y = st.number_input("UAV Y (m)", value=float(cfg.uav_pos[1]), step=1.0)
        uav_z = st.number_input("UAV Altitude (m)", value=float(cfg.uav_pos[2]), min_value=10.0, step=10.0)
        uav_vx = st.number_input("UAV Velocity X (m/s)", value=float(cfg.uav_vel[0]), step=1.0)

        st.markdown(f'<div style="color:{TEXT_LABEL}; font-family:{FONT_FAMILY}; font-size:0.8rem; margin-top:12px;">TARGET</div>', unsafe_allow_html=True)
        target_x = st.number_input("Target X (m)", value=float(cfg.target_pos[0]), step=1.0)
        target_y = st.number_input("Target Y (m)", value=float(cfg.target_pos[1]), step=1.0)
        target_radius = st.number_input("Target Radius (m)", value=float(cfg.target_radius), min_value=0.1, step=0.5)

        st.markdown(f'<div style="color:{TEXT_LABEL}; font-family:{FONT_FAMILY}; font-size:0.8rem; margin-top:12px;">ENVIRONMENT</div>', unsafe_allow_html=True)
        wind_x = st.number_input("Wind X (m/s)", value=float(cfg.wind_mean[0]), step=0.5)
        wind_std = st.number_input("Wind Std Dev (m/s)", value=float(cfg.wind_std), min_value=0.0, step=0.1)

        st.markdown(f'<div style="color:{TEXT_LABEL}; font-family:{FONT_FAMILY}; font-size:0.8rem; margin-top:12px;">SIMULATION</div>', unsafe_allow_html=True)
        n_samples = st.number_input("Monte Carlo Samples", value=cfg.n_samples, min_value=50, max_value=1000, step=50)
        random_seed = st.number_input("Random Seed", value=cfg.RANDOM_SEED, min_value=0, step=1)

        st.markdown("---")
        run_button = st.button("🚀 RUN SIMULATION", use_container_width=True)

        st.markdown(f'<div style="color:{TEXT_LABEL}; font-family:{FONT_FAMILY}; font-size:0.8rem; margin-top:12px;">DECISION THRESHOLD</div>', unsafe_allow_html=True)
        threshold_pct = st.slider(
            "Probability Threshold (%)",
            min_value=int(cfg.THRESHOLD_SLIDER_MIN),
            max_value=int(cfg.THRESHOLD_SLIDER_MAX),
            value=int(cfg.THRESHOLD_SLIDER_INIT),
            step=1
        )

    # ── Update config ──
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

    # ── Run sim ──
    if run_button or 'results' not in st.session_state:
        with st.spinner('Running Monte Carlo simulation...'):
            impact_points, P_hit, cep50, advisory_result, mission_state, impact_velocity_stats, confidence_index = run_simulation(random_seed)
            st.session_state.results = {
                'impact_points': impact_points,
                'P_hit': P_hit,
                'cep50': cep50,
                'advisory_result': advisory_result,
                'mission_state': mission_state,
                'impact_velocity_stats': impact_velocity_stats,
                'confidence_index': confidence_index,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    if 'results' not in st.session_state:
        st.info("Click RUN SIMULATION to start.")
        return

    results = st.session_state.results
    P_hit = results['P_hit']
    cep50 = results['cep50']
    impact_points = results['impact_points']
    advisory_result = results['advisory_result']
    impact_velocity_stats = results.get('impact_velocity_stats')
    confidence_index = float(results.get('confidence_index', 0.5))

    threshold = threshold_pct / 100.0
    decision = decision_logic.evaluate_drop_decision(P_hit, threshold)

    # Confidence Index classification
    if confidence_index >= 0.75:
        confidence_class = "High"
    elif confidence_index >= 0.50:
        confidence_class = "Moderate"
    else:
        confidence_class = "Low"

    # ════════════════════════════════════════════════════════════════════════════
    #  5 TABS — matching desktop app
    # ════════════════════════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Mission Overview",
        "Payload Library",
        "Sensor & Telemetry",
        "Analysis",
        "System Status",
    ])

    # ── TAB 1: Mission Overview ───────────────────────────────────────────────
    with tab1:
        # Decision Banner
        is_drop = decision == "DROP"
        banner_class = "decision-drop" if is_drop else "decision-nodrop"
        
        # Determine confidence label color: same as decision or muted? Let's keep it clean white/primary.
        
        st.markdown(f"""
        <div class="decision-banner {banner_class}">
            <div class="decision-text">{decision}</div>
            <div class="confidence-text">Confidence Index: {confidence_index:.2f} ({confidence_class})</div>
            <div class="decision-stats">
                HIT {P_hit*100:.1f}% &nbsp;|&nbsp; THRESH {threshold_pct:.0f}% &nbsp;|&nbsp; CEP50 {cep50:.2f}m
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Decision", decision)
        with c2:
            st.metric("Hit Probability", f"{P_hit*100:.1f}%")
        with c3:
            st.metric("CEP50", f"{cep50:.2f} m")
        with c4:
            st.metric("Threshold", f"{threshold_pct}%")

        st.markdown("---")

        # Two-column: Left = info, Right = impact plot
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown(f"""
            <div class="panel-card">
                <div class="panel-title">KEY METRICS</div>
                <div class="panel-row"><span class="panel-label">Mode</span><span class="panel-value">Balanced</span></div>
                <div class="panel-row"><span class="panel-label">Samples</span><span class="panel-value">{cfg.n_samples}</span></div>
                <div class="panel-row"><span class="panel-label">Seed</span><span class="panel-value">{random_seed}</span></div>
                <div class="panel-row"><span class="panel-label">Hits</span><span class="panel-value">{int(P_hit * len(impact_points))}/{len(impact_points)}</span></div>
                <div class="panel-row"><span class="panel-label">Target Radius</span><span class="panel-value">{cfg.target_radius:.1f} m</span></div>
            </div>
            """, unsafe_allow_html=True)

            analytical_dir = _analytical_advisory(advisory_result)
            st.markdown(f"""
            <div class="panel-card" style="margin-top: 8px;">
                <div class="panel-title">ADVISORY</div>
                <div class="panel-row"><span class="panel-label">Feasibility</span><span class="panel-value-accent">{advisory_result.current_feasibility}</span></div>
                <div class="panel-row"><span class="panel-label">Trend</span><span class="panel-value">{advisory_result.trend_summary}</span></div>
                <div class="panel-row"><span class="panel-label">Analysis</span><span class="panel-value">{analytical_dir}</span></div>
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            fig, ax = plt.subplots(figsize=(7, 7), facecolor=BG_PANEL)
            
            # Symmetric scaling logic
            tp = np.asarray(cfg.target_pos, dtype=float)
            impacts = np.asarray(impact_points, dtype=float)
            
            # Calculate max extent from target to include all points and target circle
            # But let's keep it centered on target for the "crosshair" feel
            if len(impacts) > 0:
                max_dist = np.max(np.linalg.norm(impacts - tp, axis=1))
                view_rad = max(max_dist * 1.1, cfg.target_radius * 2.0, 20.0) # Ensure at least 20m or fits data
            else:
                view_rad = 50.0

            plots.plot_impact_dispersion(
                ax,
                impact_points,
                cfg.target_pos,
                cfg.target_radius,
                cep50,
                release_point=cfg.uav_pos[:2],
                wind_vector=cfg.wind_mean[:2],
            )
            
            # Crosshair
            ax.plot([tp[0]-2, tp[0]+2], [tp[1], tp[1]], color=ACCENT_GO, lw=1, alpha=0.8)
            ax.plot([tp[0], tp[0]], [tp[1]-2, tp[1]+2], color=ACCENT_GO, lw=1, alpha=0.8)
            
            # Symmetric View
            ax.set_xlim(tp[0] - view_rad, tp[0] + view_rad)
            ax.set_ylim(tp[1] - view_rad, tp[1] + view_rad)
            
            ax.set_title("IMPACT DISPERSION", color=TEXT_LABEL, fontsize=10, family="monospace")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.caption(f"📸 Snapshot: {results['timestamp']} | Seed: {random_seed} | Samples: {n_samples}")

    # ── TAB 2: Payload Library ────────────────────────────────────────────────
    with tab2:
        st.markdown(f'<div class="panel-title">STEP 1: IDENTITY</div>', unsafe_allow_html=True)

        # Category filter
        selected_cat = st.selectbox("Category", ["All"] + CATEGORIES, key="pl_cat")

        # Filter payloads
        if selected_cat == "All":
            filtered = PAYLOAD_LIBRARY
        else:
            filtered = [p for p in PAYLOAD_LIBRARY if p["category"] == selected_cat]

        payload_names = [p["name"] for p in filtered]
        selected_name = st.selectbox("Select Payload", payload_names if payload_names else ["—"], key="pl_sel")

        # Find selected payload
        sel_payload = None
        for p in filtered:
            if p["name"] == selected_name:
                sel_payload = p
                break

        if sel_payload:
            st.markdown(f"""
            <div class="panel-card">
                <div class="panel-row"><span class="panel-label">ID</span><span class="panel-value">{sel_payload['id']}</span></div>
                <div class="panel-row"><span class="panel-label">Category</span><span class="panel-value-accent">{sel_payload['category']}</span></div>
                <div class="panel-row"><span class="panel-label">Subcategory</span><span class="panel-value">{sel_payload.get('subcategory', '—')}</span></div>
                <div class="panel-row"><span class="panel-label">Description</span><span class="panel-value">{sel_payload['description']}</span></div>
                <div class="panel-row"><span class="panel-label">Notes</span><span class="panel-value">{sel_payload['notes']}</span></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # Step 2 & 3: Physics
            st.markdown(f'<div class="panel-title">STEP 2 & 3: PHYSICS PARAMETERS</div>', unsafe_allow_html=True)

            col_a, col_b = st.columns(2)

            with col_a:
                pl_mass = st.number_input("Mass (kg)", value=1.0, min_value=0.01, step=0.1, key="pl_mass")

                shape = st.radio("Shape", ["sphere", "cylinder", "box", "capsule", "blunt_cone"],
                                 horizontal=True, key="pl_shape")

                dims = {}
                if shape == "sphere":
                    dims["radius"] = st.number_input("Radius (m)", value=0.1, min_value=0.001, step=0.01, key="pl_r")
                elif shape in ["cylinder", "capsule", "blunt_cone"]:
                    dims["radius"] = st.number_input("Radius (m)", value=0.1, min_value=0.001, step=0.01, key="pl_r2")
                    dims["length"] = st.number_input("Length (m)", value=0.3, min_value=0.001, step=0.01, key="pl_l")
                elif shape == "box":
                    dims["length"] = st.number_input("Length (m)", value=0.3, min_value=0.001, step=0.01, key="pl_bl")
                    dims["width"] = st.number_input("Width (m)", value=0.2, min_value=0.001, step=0.01, key="pl_bw")
                    dims["height"] = st.number_input("Height (m)", value=0.15, min_value=0.001, step=0.01, key="pl_bh")
                pl_safe_text = st.text_input("Max Safe Impact Speed (m/s, optional)", value="", key="pl_safe_speed")
                try:
                    pl_safe_speed = float(pl_safe_text) if str(pl_safe_text).strip() else None
                except ValueError:
                    pl_safe_speed = None
                st.session_state["pl_max_safe_impact_speed"] = pl_safe_speed

            with col_b:
                pl_cd = st.number_input("Drag Coefficient (Cd)", value=0.47, min_value=0.01, step=0.01, key="pl_cd")
                cd_source = st.selectbox(
                    "Cd Source",
                    ["Literature", "Empirical Test", "CFD Estimate", "User Override"],
                    index=0,
                    key="pl_cd_source",
                )
                st.caption("Assumed constant Cd (orientation-averaged)")

                ref_area = compute_reference_area(shape, dims)
                bc = compute_bc(pl_mass, pl_cd, ref_area) if ref_area > 0 else None
                payload_config = {
                    "mass": pl_mass,
                    "reference_area": ref_area,
                    "drag_coefficient": pl_cd,
                    "cd_source": cd_source,
                    "max_safe_impact_speed": pl_safe_speed,
                    "geometry": {"type": shape, "dimensions": dims},
                }

                st.markdown(f"""
                <div class="panel-card">
                    <div class="panel-title">AERODYNAMICS</div>
                    <div class="panel-row"><span class="panel-label">Reference Area</span><span class="panel-value-accent">{ref_area:.4f} m²</span></div>
                    <div class="panel-row"><span class="panel-label">Cd</span><span class="panel-value">{pl_cd:.3f}</span></div>
                    <div class="panel-row"><span class="panel-label">Cd Source</span><span class="panel-value">{cd_source}</span></div>
                </div>
                """, unsafe_allow_html=True)
                if cd_source == "User Override":
                    st.warning("User-specified Cd — verify validity.")

                st.markdown(f"""
                <div class="panel-card" style="margin-top: 8px;">
                    <div class="panel-title">BALLISTIC COEFFICIENT</div>
                    <div style="text-align:center; font-size: 1.8rem; color: {ACCENT_GO}; font-family: {FONT_FAMILY}; padding: 12px 0;">
                        {f'{bc:.1f} kg/m²' if bc else '—'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.caption("BC = m / (Cd × A)")

    # ── TAB 3: Sensor & Telemetry ─────────────────────────────────────────────
    with tab3:
        st.markdown('<span class="snapshot-badge">Snapshot Mode – Not Live Feed</span>', unsafe_allow_html=True)
        col_nav, col_att, col_env = st.columns(3)

        with col_nav:
            st.markdown(f"""
            <div class="panel-card">
                <div class="panel-title">NAVIGATION STATE</div>
                <div style="color:{TEXT_LABEL}; font-size:0.65rem; font-family:{FONT_FAMILY}; margin-bottom:8px;">MEASURED</div>
                <div class="panel-row"><span class="panel-label">Latitude</span><span class="panel-value">Awaiting telemetry input</span></div>
                <div class="panel-row"><span class="panel-label">Longitude</span><span class="panel-value">Awaiting telemetry input</span></div>
                <div class="panel-row"><span class="panel-label">Speed</span><span class="panel-value">Awaiting telemetry input</span></div>
                <div class="panel-row"><span class="panel-label">Heading</span><span class="panel-value">Awaiting telemetry input</span></div>
                <div class="panel-row"><span class="panel-label">Altitude</span><span class="panel-value">Awaiting telemetry input</span></div>
                <div class="panel-row"><span class="panel-label">Fix</span><span class="panel-value" style="color:{ACCENT_NO_GO};">No Fix</span></div>
                <div class="panel-row"><span class="panel-label">Freshness</span><span class="panel-value">Awaiting telemetry input</span></div>
            </div>
            """, unsafe_allow_html=True)

        with col_att:
            st.markdown(f"""
            <div class="panel-card">
                <div class="panel-title">ATTITUDE & MOTION</div>
                <div style="color:{TEXT_LABEL}; font-size:0.65rem; font-family:{FONT_FAMILY}; margin-bottom:8px;">MEASURED</div>
                <div class="panel-row"><span class="panel-label">Roll</span><span class="panel-value">Awaiting telemetry input</span></div>
                <div class="panel-row"><span class="panel-label">Pitch</span><span class="panel-value">Awaiting telemetry input</span></div>
                <div class="panel-row"><span class="panel-label">Yaw</span><span class="panel-value">Awaiting telemetry input</span></div>
                <div class="panel-row"><span class="panel-label">|a|</span><span class="panel-value">Awaiting telemetry input</span></div>
                <div class="panel-row"><span class="panel-label">IMU</span><span class="panel-value">Awaiting telemetry input</span></div>
            </div>
            """, unsafe_allow_html=True)

        with col_env:
            st.markdown(f"""
            <div class="panel-card">
                <div class="panel-title">ENVIRONMENT</div>
                <div style="color:{TEXT_LABEL}; font-size:0.65rem; font-family:{FONT_FAMILY}; margin-bottom:8px;">DERIVED · ESTIMATED</div>
                <div class="panel-row"><span class="panel-label">Wind direction</span><span class="panel-value-warn">Awaiting telemetry input</span></div>
                <div class="panel-row"><span class="panel-label">Wind speed</span><span class="panel-value-warn">Awaiting telemetry input</span></div>
                <div class="panel-row"><span class="panel-label">Wind mean</span><span class="panel-value-warn">{cfg.wind_mean[0]:.2f} m/s</span></div>
                <div class="panel-row"><span class="panel-label">Wind std σ</span><span class="panel-value-warn">{cfg.wind_std:.2f} m/s</span></div>
                <div class="panel-row"><span class="panel-label">Uncertainty</span><span class="panel-value-warn">Awaiting telemetry input</span></div>
                <div class="panel-row"><span class="panel-label">Source</span><span class="panel-value-warn">Awaiting telemetry input</span></div>
                <div class="panel-row"><span class="panel-label">Confidence</span><span class="panel-value-warn">Awaiting telemetry input</span></div>
            </div>
            """, unsafe_allow_html=True)
            st.caption("Using assumed Gaussian wind model.")

    # ── TAB 4: Analysis ───────────────────────────────────────────────────────
    with tab4:
        row_top1, row_top2 = st.columns(2)

        with row_top1:
            st.markdown(f'<div class="panel-card"><div class="panel-title">P(HIT) vs TARGET DISTANCE</div><div style="text-align:center; color:{TEXT_LABEL}; padding:40px; font-family:{FONT_FAMILY}; font-size:0.8rem;">Sensitivity sweep not performed.<br/>Use Opportunity Analysis to generate sweep.</div></div>', unsafe_allow_html=True)

        with row_top2:
            fig_imp, ax_imp = plt.subplots(figsize=(6, 5), facecolor=BG_PANEL)
            plots.plot_impact_dispersion(
                ax_imp,
                impact_points,
                cfg.target_pos,
                cfg.target_radius,
                cep50,
                release_point=cfg.uav_pos[:2],
                wind_vector=cfg.wind_mean[:2],
            )
            ax_imp.set_title("IMPACT DISPERSION", color=TEXT_LABEL, fontsize=9, family="monospace")
            plt.tight_layout()
            st.pyplot(fig_imp)
            plt.close(fig_imp)

        row_bot1, row_bot2, row_bot3 = st.columns(3)

        with row_bot1:
            st.markdown(f'<div class="panel-card"><div class="panel-title">P(HIT) vs WIND UNCERTAINTY</div><div style="text-align:center; color:{TEXT_LABEL}; padding:40px; font-family:{FONT_FAMILY}; font-size:0.8rem;">Sensitivity sweep not performed.<br/>Use Opportunity Analysis to generate sweep.</div></div>', unsafe_allow_html=True)

        with row_bot2:
            st.markdown(f"""
            <div class="panel-card">
                <div class="panel-title">CEP SUMMARY</div>
                <div class="panel-row"><span class="panel-label">CEP50</span><span class="panel-value-accent">{cep50:.2f} m</span></div>
                <div class="panel-row"><span class="panel-label">Hit %</span><span class="panel-value-accent">{P_hit*100:.1f}%</span></div>
                <div style="color:{TEXT_LABEL}; font-size:0.7rem; font-family:{FONT_FAMILY}; text-align:center; margin-top:12px;">Read-only · No recomputation</div>
            </div>
            """, unsafe_allow_html=True)

        with row_bot3:
            ivs = impact_velocity_stats or {}
            mean_s = ivs.get("mean_impact_speed", 0)
            std_s = ivs.get("std_impact_speed", 0)
            p95_s = ivs.get("p95_impact_speed", 0)
            safe_speed = st.session_state.get("pl_max_safe_impact_speed")
            if safe_speed is None:
                survivability_text = "No structural limit defined."
            else:
                ratio = float(p95_s) / max(float(safe_speed), 1e-9)
                if ratio <= 0.80:
                    risk = "LOW"
                elif ratio <= 1.00:
                    risk = "MODERATE"
                else:
                    risk = "HIGH"
                survivability_text = f"Survivability Risk: {risk}"
            st.markdown(f"""
            <div class="panel-card">
                <div class="panel-title">IMPACT DYNAMICS</div>
                <div class="panel-row"><span class="panel-label">Mean impact speed</span><span class="panel-value-accent">{mean_s:.2f} m/s</span></div>
                <div class="panel-row"><span class="panel-label">Std dev</span><span class="panel-value">{std_s:.2f} m/s</span></div>
                <div class="panel-row"><span class="panel-label">95% percentile</span><span class="panel-value">{p95_s:.2f} m/s</span></div>
                <div class="panel-row"><span class="panel-label">Survivability</span><span class="panel-value">{survivability_text}</span></div>
                <div style="color:{TEXT_LABEL}; font-size:0.65rem; font-family:{FONT_FAMILY}; text-align:center; margin-top:12px;">Impact survivability depends on payload structural limits.</div>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 5: System Status ──────────────────────────────────────────────────
    with tab5:
        # Engine Identity
        with st.expander("ENGINE IDENTITY", expanded=True):
            st.markdown(f"""
            <div class="panel-row"><span class="panel-label">System</span><span class="panel-value-accent">AIRDROP-X · v1.0</span></div>
            <div class="panel-row"><span class="panel-label">Physics</span><span class="panel-value">3-DOF point-mass; gravity -Z; quadratic drag (v_rel); explicit Euler.</span></div>
            <div class="panel-row"><span class="panel-label">Monte Carlo</span><span class="panel-value">One wind sample per trajectory; Gaussian uncertainty; fixed seed.</span></div>
            <div class="panel-row"><span class="panel-label">Config source</span><span class="panel-value">configs.mission_configs (ASSUMED)</span></div>
            """, unsafe_allow_html=True)

        # Reproducibility
        with st.expander("REPRODUCIBILITY", expanded=True):
            st.markdown(f"""
            <div class="panel-row"><span class="panel-label">Random seed</span><span class="panel-value">{random_seed}</span></div>
            <div class="panel-row"><span class="panel-label">Sample count</span><span class="panel-value">{n_samples}</span></div>
            <div class="panel-row"><span class="panel-label">Time step</span><span class="panel-value">{cfg.dt} s</span></div>
            <div class="panel-row"><span class="panel-label">Uncertainty</span><span class="panel-value">Gaussian wind (mean + std), one sample per trajectory.</span></div>
            <div class="panel-row"><span class="panel-label">Snapshot</span><span class="panel-value">{results['timestamp']}</span></div>
            """, unsafe_allow_html=True)

        with st.expander("NUMERICAL STABILITY", expanded=True):
            try:
                stab = quick_stability_check(random_seed=int(random_seed), dt=float(cfg.dt), samples=5)
                rel_pct = stab["relative_error"] * 100.0
                st.markdown(f"""
                <div class="panel-row"><span class="panel-label">Integration method</span><span class="panel-value">{stab['integration_method']}</span></div>
                <div class="panel-row"><span class="panel-label">Time step Δt</span><span class="panel-value">{stab['dt']} s</span></div>
                <div class="panel-row"><span class="panel-label">Samples</span><span class="panel-value">{stab['samples']}</span></div>
                <div class="panel-row"><span class="panel-label">Stability status</span><span class="panel-value">{stab['status']} ({rel_pct:.2f}% relative error)</span></div>
                """, unsafe_allow_html=True)
            except Exception:
                st.markdown(f"""
                <div class="panel-row"><span class="panel-label">Integration method</span><span class="panel-value">Explicit Euler</span></div>
                <div class="panel-row"><span class="panel-label">Time step Δt</span><span class="panel-value">{cfg.dt} s</span></div>
                <div class="panel-row"><span class="panel-label">Samples</span><span class="panel-value">5</span></div>
                <div class="panel-row"><span class="panel-label">Stability status</span><span class="panel-value">CAUTION</span></div>
                """, unsafe_allow_html=True)

        # Limitations
        with st.expander("LIMITATIONS & ASSUMPTIONS", expanded=True):
            limitations = [
                "Point-mass payload; no rotation or attitude dynamics.",
                "2D target (X–Y); impact at Z=0 only.",
                "Uniform atmosphere (constant rho); no wind shear or gusts.",
                "Wind uncertainty: isotropic Gaussian; no spatial/temporal correlation.",
                "Single release point per run; no multi-drop or sequencing.",
            ]
            for lim in limitations:
                st.markdown(f'<div style="color:{TEXT_PRIMARY}; font-family:{FONT_FAMILY}; font-size:0.82rem; padding:2px 0;">· {lim}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown(f'<div class="panel-title">MODEL VALIDITY ENVELOPE</div>', unsafe_allow_html=True)
            
            validity_lines = [
                ("50 m – 1500 m AGL", "High confidence"),
                ("1500 m – 3000 m AGL", "Increasing dispersion"),
                (">3000 m AGL", "Use with caution")
            ]
            for v_rng, v_desc in validity_lines:
                 st.markdown(f'<div class="panel-row"><span class="panel-label">{v_rng}</span><span class="panel-value">{v_desc}</span></div>', unsafe_allow_html=True)
            
            st.markdown(f'<div style="font-size:0.75rem; color:{TEXT_LABEL}; margin-top:8px;">Confidence decreases primarily due to environmental uncertainty.</div>', unsafe_allow_html=True)


        # Warnings
        with st.expander("⚠ WARNINGS / STATUS", expanded=True):
            st.markdown(f'<div style="color:{ACCENT_WARN}; font-family:{FONT_FAMILY}; font-size:0.82rem; padding:2px 0;">⚠ No active warnings.</div>', unsafe_allow_html=True)

    # ── Footer ──
    st.markdown("---")
    st.markdown(f"""
    <div class="footer-text">
        AIRDROP-X v1.1<br/>
        Engine Frozen | Operator-in-the-Loop Decision Support<br/>
        Probabilistic Impact & Confidence Modeling Enabled
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
