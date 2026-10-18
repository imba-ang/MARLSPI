from dataclasses import dataclass
from math import exp, log
from .config import PAPER


@dataclass(frozen=True)
class RewardConfig:
    speed_threshold: float = 10.0
    merge_end_distance: float = 300.0
    passing_incentive: float = 1.0
    driving_incentive: float = 1.0

    def __post_init__(self):
        if not 0 <= self.speed_threshold < PAPER.max_speed or self.merge_end_distance <= 0:
            raise ValueError("reward configuration")
        if min(self.passing_incentive, self.driving_incentive) <= 0:
            raise ValueError("lane incentives")


def reward_components(speed: float, position: float, headway: float, collision: bool, destination: bool, lane: int, ramp_neighbor: bool, safe_gap: bool, merge_space_unblocked: bool, config: RewardConfig = RewardConfig()) -> tuple[float, ...]:
    if speed < 0 or headway <= 0:
        raise ValueError("speed, headway")
    speed_reward = (speed - config.speed_threshold) / (PAPER.max_speed - config.speed_threshold)
    urgency = exp(-(position - config.merge_end_distance) ** 2 / (20 * config.merge_end_distance))
    safety = min(log(headway / (speed * PAPER.time_headway)), 1.0) if speed > 0 else 1.0
    lane_reward = 0.0
    if lane == 3 and speed > config.speed_threshold:
        lane_reward = config.passing_incentive
    elif lane == 2 and ramp_neighbor and safe_gap and merge_space_unblocked:
        lane_reward = config.driving_incentive
    return speed_reward, urgency, safety, -float(collision), float(destination), lane_reward


def total_reward(components: tuple[float, ...]) -> float:
    if len(components) != len(PAPER.reward_weights):
        raise ValueError("reward components")
    return sum(w * r for w, r in zip(PAPER.reward_weights, components))
