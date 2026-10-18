from bisect import bisect_right
from ..config import PAPER, RUNTIME
from ..observation import LocalObservation, Role


class StateAbstraction:
    def __init__(self, absent_leader_gap: float = RUNTIME.absent_leader_gap, absent_leader_relative_speed: float = RUNTIME.absent_leader_relative_speed):
        self.absent_leader_gap = absent_leader_gap
        self.absent_leader_relative_speed = absent_leader_relative_speed

    @staticmethod
    def quantize(value: float, boundaries: tuple[float, ...]) -> int:
        if not boundaries[0] <= value < boundaries[-1]:
            raise ValueError((value, boundaries))
        return bisect_right(boundaries, value)

    def __call__(self, observation: LocalObservation) -> tuple[int, ...]:
        leader = observation.neighbors[Role.SV]
        present = leader.available and abs(leader.distance) <= PAPER.perception_range
        gap = leader.distance - PAPER.vehicle_length if present else self.absent_leader_gap
        relative_speed = leader.relative_speed if present else self.absent_leader_relative_speed
        return (self.quantize(observation.speed, PAPER.speed_bins), self.quantize(gap, PAPER.gap_bins), self.quantize(relative_speed, PAPER.relative_speed_bins), self.quantize(observation.merge_distance, PAPER.merge_distance_bins), observation.lane)
