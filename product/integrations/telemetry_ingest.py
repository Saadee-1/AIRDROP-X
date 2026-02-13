"""
Live telemetry ingest. Read-only; parses UAV state and forwards to product pipeline.
No control or command output. No physics or Monte Carlo.
"""

from .uav_state import UAVStateSnapshot


def parse_uav_state(raw):
    """
    Parse a raw update (dict) into UAVStateSnapshot.
    Expects keys: time_s (or time_boot_ms -> s), position or (x,y,z), velocity or (vx,vy,vz).
    """
    if isinstance(raw, UAVStateSnapshot):
        return raw
    if not isinstance(raw, dict):
        raise TypeError("raw must be dict or UAVStateSnapshot")
    t = raw.get("time_s")
    if t is None:
        ms = raw.get("time_boot_ms")
        t = float(ms) / 1000.0 if ms is not None else 0.0
    else:
        t = float(t)
    p = raw.get("position")
    if isinstance(p, (list, tuple)) and len(p) >= 3:
        pos = (float(p[0]), float(p[1]), float(p[2]))
    elif isinstance(p, dict):
        pos = (
            float(p.get("x", 0)),
            float(p.get("y", 0)),
            float(p.get("z", p.get("altitude", 0))),
        )
    else:
        pos = (
            float(raw.get("x", 0)),
            float(raw.get("y", 0)),
            float(raw.get("z", raw.get("altitude", 0))),
        )

    # Advisory range check (telemetry might be valid but extreme)
    if pos[2] < -500.0:
        print(
            f"[TELEM WARNING] Altitude {pos[2]:.1f} m is very low (<-500m). "
            f"Check coordinate frame (NED vs ENU)."
        )

    v = raw.get("velocity")
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        vel = (float(v[0]), float(v[1]), float(v[2]))
    elif isinstance(v, dict):
        vel = (
            float(v.get("vx", 0)),
            float(v.get("vy", 0)),
            float(v.get("vz", 0)),
        )
    else:
        vel = (
            float(raw.get("vx", 0)),
            float(raw.get("vy", 0)),
            float(raw.get("vz", 0)),
        )

    # Advisory velocity check (supersonic?)
    speed_sq = vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]
    if speed_sq > 340.0 * 340.0:
        print(
            "[TELEM WARNING] Velocity magnitude > 340 m/s. "
            "Supersonic input detected."
        )

    return UAVStateSnapshot(t, pos, vel)


def ingest_stream(raw_stream, parse=None):
    """
    Consume an iterator of raw telemetry updates; yield UAVStateSnapshot.
    raw_stream: iterator of dicts (or MAVLink-derived dicts).
    parse: optional callable(raw) -> UAVStateSnapshot. Default: parse_uav_state.
    Read-only; no control output.
    """
    if parse is None:
        parse = parse_uav_state
    for raw in raw_stream:
        try:
            snapshot = parse(raw)
            if snapshot is not None:
                yield snapshot
        except (TypeError, ValueError, KeyError):
            continue
