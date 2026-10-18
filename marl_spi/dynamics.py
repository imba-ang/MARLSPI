from dataclasses import dataclass
from math import atan, cos, sin, tan, hypot
from .config import PAPER, RUNTIME


@dataclass(frozen=True)
class KinematicState:
    x: float
    y: float
    speed: float
    heading: float


def bicycle_step(state: KinematicState, acceleration: float, steering: float, wheelbase: float = RUNTIME.wheelbase, dt: float = PAPER.simulation_dt) -> KinematicState:
    if wheelbase <= 0 or dt <= 0:
        raise ValueError("wheelbase, dt")
    beta = atan(0.5 * tan(steering))
    return KinematicState(state.x + dt * state.speed * cos(state.heading + beta), state.y + dt * state.speed * sin(state.heading + beta), state.speed + dt * acceleration, state.heading + dt * state.speed / wheelbase * sin(beta))


def geometric_clearance(dx: float, dy: float, length_i: float = PAPER.vehicle_length, length_j: float = PAPER.vehicle_length, width_i: float = PAPER.vehicle_width, width_j: float = PAPER.vehicle_width) -> float:
    gx = abs(dx) - (length_i + length_j) / 2
    gy = abs(dy) - (width_i + width_j) / 2
    return hypot(max(gx, 0), max(gy, 0)) if gx > 0 or gy > 0 else max(gx, gy)


def closest_approach(dx: float, dy: float, dvx: float, dvy: float, velocity_epsilon: float = RUNTIME.velocity_epsilon) -> tuple[float, float, bool]:
    if velocity_epsilon < 0:
        raise ValueError("velocity_epsilon")
    norm = dvx * dvx + dvy * dvy
    dot = dx * dvx + dy * dvy
    moving = norm > velocity_epsilon ** 2
    time = max(0.0, -dot / norm) if moving else 0.0
    clearance = geometric_clearance(dx + time * dvx, dy + time * dvy)
    return time, clearance, moving and dot < 0
