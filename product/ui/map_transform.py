from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math


@dataclass
class MapTransform:
    pixels_per_meter: float = 1.4

    def __post_init__(self) -> None:
        self._validate_scale(self.pixels_per_meter)

    @property
    def scale(self) -> float:
        return self.pixels_per_meter

    @scale.setter
    def scale(self, value: float) -> None:
        self._validate_scale(value)
        self.pixels_per_meter = value

    def world_to_scene(self, x: float, y: float) -> Tuple[float, float]:
        return float(x), float(y)

    def scene_to_world(self, x: float, y: float) -> Tuple[float, float]:
        return float(x), float(y)

    def apply_to_view(self, view) -> None:
        center = view.mapToScene(view.viewport().rect().center())
        view.resetTransform()
        view.scale(self.pixels_per_meter, -self.pixels_per_meter)
        view.centerOn(center)

    @staticmethod
    def _validate_scale(scale: float) -> None:
        if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid map scale")
