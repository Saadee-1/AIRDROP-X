from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple


@dataclass
class TacticalMapState:
    vehicle_position: Optional[Tuple[float, float]] = None
    vehicle_heading: Optional[float] = None

    target_position: Optional[Tuple[float, float]] = None

    impact_mean: Optional[Tuple[float, float]] = None
    impact_covariance: Optional[Any] = None

    ellipse_axes: Optional[Tuple[float, float]] = None
    ellipse_angle: Optional[float] = None

    release_corridor: Optional[Iterable[Tuple[float, float]]] = None

    guidance_vector: Optional[Tuple[float, float]] = None

    wind_vector: Optional[Tuple[float, float]] = None

    uncertainty_breakdown: Optional[Dict[str, float]] = None
