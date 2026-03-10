"""
Tab modules for AIRDROP-X UI. Each tab exposes render(ax).
"""

from .mission_overview import render as render_tactical_map
from .mission_overview import render_control as render_mission_control
from .payload_library import render as render_mission_setup
from .sensor_telemetry import render as render_telemetry
from .system_status import render as render_system_status

TABS = [
    ("Tactical Map", render_tactical_map),
    ("Mission Control", render_mission_control),
    ("Telemetry", render_telemetry),
    ("Mission Setup", render_mission_setup),
    ("System Status", render_system_status),
]
