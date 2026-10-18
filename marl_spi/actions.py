from dataclasses import dataclass
from enum import IntEnum
from .config import PAPER, RUNTIME


class Action(IntEnum):
    ACCELERATE = 0
    DECELERATE = 1
    MAINTAIN = 2
    LEFT = 3
    RIGHT = 4


@dataclass(frozen=True)
class Command:
    acceleration: float
    target_lane: int


def semantic_action(acceleration: float, lane_delta: int = 0) -> Action:
    if lane_delta not in (-1, 0, 1):
        raise ValueError("lane_delta")
    if lane_delta:
        return Action.LEFT if lane_delta > 0 else Action.RIGHT
    if acceleration > PAPER.acceleration_threshold:
        return Action.ACCELERATE
    if acceleration < -PAPER.acceleration_threshold:
        return Action.DECELERATE
    return Action.MAINTAIN


def execute_action(action: Action, lane: int, feasible, longitudinal_commands: tuple[float, float, float] = RUNTIME.longitudinal_commands, nominal_acceleration: float = RUNTIME.nominal_acceleration) -> Command:
    action = Action(action)
    if not feasible[int(action)]:
        raise ValueError("infeasible action")
    acceleration = longitudinal_commands[int(action)] if action < Action.LEFT else nominal_acceleration
    if not PAPER.control_min <= acceleration <= PAPER.control_max:
        raise ValueError("acceleration")
    target = lane + int(action == Action.LEFT) - int(action == Action.RIGHT)
    return Command(acceleration, target)
