from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping
import numpy as np
from .config import PAPER


class Role(IntEnum):
    SV = 0
    FV = 1
    NLV = 2
    NPV = 3
    RLV = 4
    RPV = 5


@dataclass(frozen=True)
class Neighbor:
    distance: float
    relative_speed: float
    available: bool = True


@dataclass(frozen=True)
class LocalObservation:
    lane: int
    speed: float
    merge_distance: float
    neighbors: tuple[Neighbor, ...]

    def __post_init__(self):
        if self.lane not in (1, 2, 3) or len(self.neighbors) != 6:
            raise ValueError("lane, neighbors")

    @classmethod
    def from_roles(cls, lane: int, speed: float, merge_distance: float, roles: Mapping[Role, Neighbor]):
        return cls(lane, speed, merge_distance, tuple(roles.get(r, Neighbor(0, 0, False)) for r in Role))

    def vector(self) -> np.ndarray:
        values = [float(self.lane), self.speed, self.merge_distance]
        for neighbor in self.neighbors:
            present = neighbor.available and abs(neighbor.distance) <= PAPER.perception_range
            values.extend((neighbor.distance, neighbor.relative_speed, 1.0) if present else (0.0, 0.0, 0.0))
        return np.asarray(values, dtype=np.float32)
