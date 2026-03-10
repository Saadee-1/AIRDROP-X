from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math


@dataclass
class MapTransform:
    scale: float = 1.0
    origin_x: float = 0.0
    origin_y: float = 0.0

    def __post_init__(self) -> None:
        self._validate_scale(self.scale)

    def world_to_scene(self, x: float, y: float) -> Tuple[float, float]:
        self._validate_scale(self.scale)
        scene_x = (float(x) - self.origin_x) * self.scale
        scene_y = (float(y) - self.origin_y) * self.scale
        return scene_x, scene_y

    def scene_to_world(self, x: float, y: float) -> Tuple[float, float]:
        self._validate_scale(self.scale)
        world_x = float(x) / self.scale + self.origin_x
        world_y = float(y) / self.scale + self.origin_y
        return world_x, world_y

    @staticmethod
    def _validate_scale(scale: float) -> None:
        if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid map scale")
