from dataclasses import dataclass
from math import isfinite
from ..config import PAPER, LearningConfig


@dataclass
class AdaptiveKL:
    config: LearningConfig = LearningConfig()
    coefficient: float = PAPER.kl_initial

    def update(self, deviation: float) -> float:
        if not isfinite(deviation):
            raise ValueError("deviation")
        if deviation > self.config.kl_high_threshold:
            self.coefficient = min(PAPER.kl_max, self.coefficient * (1 + self.config.kl_adjustment_rate))
        elif deviation < self.config.kl_low_threshold:
            self.coefficient = max(PAPER.kl_min, self.coefficient * (1 - self.config.kl_adjustment_rate))
        return self.coefficient
