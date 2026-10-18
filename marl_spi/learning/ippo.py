import torch
from torch.distributions import Categorical
from ..config import PAPER, LearningConfig
from ..policies.adaptive_kl import AdaptiveKL
from ..policies.soft_mask import masked_distribution, prior_kl
from .advantages import generalized_advantage


class IndependentPPO:
    def __init__(self, network, config: LearningConfig | None = None, use_spi: bool = False):
        self.network = network
        self.config = config if config is not None else LearningConfig()
        self.use_spi = use_spi
        self.adaptive_kl = AdaptiveKL(self.config)
        self.optimizer = torch.optim.Adam(network.parameters(), lr=PAPER.learning_rate)

    def policy_loss(self, distribution, actions, behavior_log_probabilities, advantages):
        ratio = (distribution.log_prob(actions) - behavior_log_probabilities).exp()
        clipped = ratio.clamp(1 - self.config.ppo_clip, 1 + self.config.ppo_clip)
        return -torch.minimum(ratio * advantages, clipped * advantages).mean()

    def update(self, buffer):
        batch = buffer.batch()
        if self.use_spi and batch.priors is None:
            raise ValueError("SPI prior")
        advantages, returns = generalized_advantage(batch.rewards, batch.values, batch.next_values, batch.terminated, batch.truncated, self.config.gae_lambda)
        total_loss = 0.0
        updates = 0
        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(len(batch.actions), device=batch.actions.device)
            for indices in permutation.split(PAPER.batch_size):
                context = None if batch.contexts is None else batch.contexts[indices]
                logits, values = self.network(batch.observations[indices], context)
                deployed = masked_distribution(logits, batch.feasible[indices], batch.weights[indices])
                policy_loss = self.policy_loss(deployed, batch.actions[indices], batch.behavior_log_probabilities[indices], advantages[indices])
                value_loss = (values - returns[indices]).square().mean()
                entropy = Categorical(logits=logits).entropy().mean()
                kl = prior_kl(logits, batch.priors[indices]).mean() if self.use_spi else logits.new_zeros(())
                coefficient = self.adaptive_kl.coefficient if self.use_spi else 0.0
                loss = policy_loss + coefficient * kl + self.config.value_coefficient * value_loss - self.config.entropy_coefficient * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite learning objective")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.detach())
                updates += 1
        with torch.no_grad():
            logits, _ = self.network(batch.observations, batch.contexts)
            deviation = float(prior_kl(logits, batch.priors).mean()) if self.use_spi else 0.0
        if self.use_spi:
            self.adaptive_kl.update(deviation)
        buffer.clear()
        return {"loss": total_loss / updates, "kl": deviation, "kl_coefficient": self.adaptive_kl.coefficient if self.use_spi else 0.0}
