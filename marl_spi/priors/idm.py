from dataclasses import dataclass
from math import sqrt
from ..config import PAPER


@dataclass(frozen=True)
class IDMConfig:
    standstill_gap: float = 2.0
    desired_speed: float = PAPER.max_speed
    exponent: float = 4.0
    time_headway: float = PAPER.time_headway
    max_acceleration: float = PAPER.max_acceleration
    comfortable_deceleration: float = PAPER.comfortable_deceleration

    def __post_init__(self):
        if min(self.standstill_gap, self.desired_speed, self.exponent, self.time_headway, self.max_acceleration, self.comfortable_deceleration) <= 0:
            raise ValueError("IDM parameters")


def desired_gap(speed: float, leader_relative_speed: float, config: IDMConfig = IDMConfig()) -> float:
    return config.standstill_gap + speed * config.time_headway + speed * leader_relative_speed / (2 * sqrt(config.max_acceleration * config.comfortable_deceleration))


def acceleration(speed: float, gap: float, leader_relative_speed: float, config: IDMConfig = IDMConfig()) -> float:
    if speed < 0 or gap <= 0:
        raise ValueError("speed, gap")
    return config.max_acceleration * (1 - (speed / config.desired_speed) ** config.exponent - (desired_gap(speed, leader_relative_speed, config) / gap) ** 2)
