from dataclasses import dataclass
from typing import Literal, Iterable
from ..actions import Action, semantic_action


@dataclass(frozen=True)
class MOBILConfig:
    politeness: float = 0.3
    safe_braking: float = 4.0
    incentive_threshold: float = 0.2
    safety_subject: Literal["ego", "new_follower"] = "new_follower"

    def __post_init__(self):
        if self.safe_braking <= 0 or self.politeness < 0 or self.safety_subject not in ("ego", "new_follower"):
            raise ValueError("MOBIL parameters")


@dataclass(frozen=True)
class LaneChange:
    direction: int
    ego_before: float
    ego_after: float
    previous_follower_before: float
    previous_follower_after: float
    new_follower_before: float
    new_follower_after: float


def incentive(candidate: LaneChange, config: MOBILConfig = MOBILConfig()) -> float:
    return candidate.ego_after - candidate.ego_before + config.politeness * (candidate.previous_follower_after - candidate.previous_follower_before + candidate.new_follower_after - candidate.new_follower_before)


def accepted(candidate: LaneChange, config: MOBILConfig = MOBILConfig()) -> bool:
    if candidate.direction not in (-1, 1):
        raise ValueError("direction")
    safe_acceleration = candidate.ego_after if config.safety_subject == "ego" else candidate.new_follower_after
    return safe_acceleration >= -config.safe_braking and incentive(candidate, config) > config.incentive_threshold


def model_action(longitudinal_acceleration: float, candidates: Iterable[LaneChange], feasible, config: MOBILConfig = MOBILConfig()) -> Action:
    for candidate in candidates:
        action = semantic_action(longitudinal_acceleration, candidate.direction)
        if feasible[int(action)] and accepted(candidate, config):
            return action
    return semantic_action(longitudinal_acceleration)


def build_model_prior(samples, abstraction=None, config: MOBILConfig = MOBILConfig()):
    from .abstraction import StateAbstraction
    from .hybrid import CountPriorBuilder
    abstraction = abstraction if abstraction is not None else StateAbstraction()
    builder = CountPriorBuilder()
    for observation, longitudinal_acceleration, candidates, feasible in samples:
        action = model_action(longitudinal_acceleration, candidates, feasible, config)
        builder.add(abstraction(observation), action)
    return builder.freeze()
