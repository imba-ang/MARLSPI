from .ippo import IndependentPPO
from ..config import LearningConfig


class IndependentA2C(IndependentPPO):
    def __init__(self, network, config: LearningConfig | None = None, use_spi: bool = False):
        config = config if config is not None else LearningConfig(update_epochs=1)
        if config.update_epochs != 1:
            raise ValueError("IA2C requires one on-policy update epoch")
        super().__init__(network, config, use_spi)

    def policy_loss(self, distribution, actions, behavior_log_probabilities, advantages):
        return -(distribution.log_prob(actions) * advantages).mean()
