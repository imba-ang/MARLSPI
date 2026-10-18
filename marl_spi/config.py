from dataclasses import dataclass


@dataclass(frozen=True)
class PaperConfig:
    simulation_dt: float = 0.1
    decision_dt: float = 0.2
    road_length: float = 2000.0
    episode_seconds: float = 3600.0
    perception_range: float = 100.0
    observation_size: int = 21
    action_size: int = 5
    hidden_size: int = 128
    vehicle_length: float = 4.7
    vehicle_width: float = 1.8
    max_speed: float = 20.0
    max_acceleration: float = 3.0
    comfortable_deceleration: float = 2.0
    control_min: float = -3.0
    control_max: float = 2.0
    time_headway: float = 1.5
    gamma: float = 0.99
    batch_size: int = 128
    learning_rate: float = 5e-4
    reward_weights: tuple[float, ...] = (1.0, 4.0, 8.0, 200.0, 1.0, 2.0)
    mu_min: float = 1e-5
    alpha_d: float = 1.0
    epsilon_pi: float = 1e-6
    alpha_pi: float = 1.0
    fusion_weight: float = 0.35
    kl_initial: float = 0.20
    kl_min: float = 0.10
    kl_max: float = 0.50
    acceleration_threshold: float = 0.2
    speed_bins: tuple[float, ...] = (0, 5, 10, 15, 20, 33.3)
    gap_bins: tuple[float, ...] = (0, 5, 10, 20, 40, 80, 200)
    relative_speed_bins: tuple[float, ...] = (-10, -5, -2, 0, 2, 5, 10)
    merge_distance_bins: tuple[float, ...] = (0, 20, 50, 100, 200, 500, 1000, 2000)
    peak_inflow: tuple[int, ...] = (1227, 1491, 555)
    off_peak_inflow: tuple[int, ...] = (1167, 779, 365)


@dataclass(frozen=True)
class LearningConfig:
    ppo_clip: float = 0.2
    gae_lambda: float = 0.95
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    update_epochs: int = 10
    kl_low_threshold: float = 0.01
    kl_high_threshold: float = 0.05
    kl_adjustment_rate: float = 0.1

    def __post_init__(self):
        if not 0 < self.ppo_clip < 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("ppo_clip, gae_lambda")
        if self.update_epochs < 1 or self.value_coefficient < 0 or self.entropy_coefficient < 0:
            raise ValueError("update_epochs, loss coefficients")
        if not 0 <= self.kl_low_threshold < self.kl_high_threshold:
            raise ValueError("KL thresholds")
        if not 0 < self.kl_adjustment_rate < 1:
            raise ValueError("kl_adjustment_rate")


@dataclass(frozen=True)
class RuntimeConfig:
    wheelbase: float = 2.7
    safe_clearance: float = 2.0
    velocity_epsilon: float = 1e-6
    absent_leader_gap: float = 100.0
    absent_leader_relative_speed: float = 0.0
    longitudinal_commands: tuple[float, float, float] = (2.0, -3.0, 0.0)
    nominal_acceleration: float = 0.0
    role_input_size: int = 4
    phase_input_size: int = 2


@dataclass(frozen=True)
class DQNConfig:
    capacity: int = 50000
    temperature: float = 1.0
    target_update_interval: int = 500
    epsilon: float = 0.05

    def __post_init__(self):
        if self.capacity < PaperConfig().batch_size or self.temperature <= 0 or self.target_update_interval < 1 or not 0 <= self.epsilon <= 1:
            raise ValueError("DQN configuration")


PAPER = PaperConfig()
RUNTIME = RuntimeConfig()
DQN = DQNConfig()
