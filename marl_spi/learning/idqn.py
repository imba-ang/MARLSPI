import copy
from collections import deque
from dataclasses import dataclass
import random
import torch
from ..config import PAPER, DQN, LearningConfig
from ..policies.adaptive_kl import AdaptiveKL
from ..policies.soft_mask import masked_distribution, prior_kl


@dataclass(frozen=True)
class QTransition:
    observation: torch.Tensor
    action: int
    reward: float
    next_observation: torch.Tensor
    terminated: bool
    context: torch.Tensor | None
    next_context: torch.Tensor | None
    prior: torch.Tensor | None
    next_feasible: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int = DQN.capacity):
        if capacity < 1:
            raise ValueError("capacity")
        self._rows = deque(maxlen=capacity)

    @property
    def capacity(self):
        return self._rows.maxlen

    def __len__(self):
        return len(self._rows)

    def append(self, transition: QTransition):
        self._rows.append(transition)

    def sample(self, batch_size: int = PAPER.batch_size):
        return random.sample(list(self._rows), batch_size)


class IndependentDQN:
    def __init__(self, network, config: LearningConfig | None = None, use_spi: bool = False, temperature: float = DQN.temperature, target_update_interval: int = DQN.target_update_interval):
        if temperature <= 0 or target_update_interval < 1:
            raise ValueError("DQN configuration")
        self.network = network
        self.target = copy.deepcopy(network).requires_grad_(False)
        self.config = config if config is not None else LearningConfig()
        self.use_spi = use_spi
        self.temperature = temperature
        self.target_update_interval = target_update_interval
        self.update_count = 0
        self.adaptive_kl = AdaptiveKL(self.config)
        self.optimizer = torch.optim.Adam(network.parameters(), lr=PAPER.learning_rate)

    @torch.no_grad()
    def select(self, observation, feasible, context=None, weights=None, epsilon=DQN.epsilon):
        q = self.network(observation, context)
        feasible = feasible.to(device=q.device, dtype=torch.bool)
        if self.use_spi:
            if weights is None:
                raise ValueError("SPI weights")
            return int(masked_distribution(q / self.temperature, feasible, weights).sample())
        if epsilon is None or not 0 <= epsilon <= 1:
            raise ValueError("epsilon")
        choices = feasible.nonzero().flatten()
        if choices.numel() == 0:
            raise ValueError("empty feasible action set")
        if torch.rand((), device=q.device) < epsilon:
            return int(choices[torch.randint(len(choices), (), device=q.device)])
        return int(q.masked_fill(~feasible, -torch.inf).argmax())

    def update(self, replay: ReplayBuffer):
        rows = replay.sample()
        device = rows[0].observation.device
        def stack(name):
            return torch.stack([getattr(row, name).detach() for row in rows])
        def optional(name):
            if all(getattr(row, name) is None for row in rows):
                return None
            return stack(name)
        observations = stack("observation")
        contexts = optional("context")
        q = self.network(observations, contexts)
        actions = torch.tensor([row.action for row in rows], device=device)
        rewards = torch.tensor([row.reward for row in rows], device=device)
        terminated = torch.tensor([row.terminated for row in rows], device=device)
        with torch.no_grad():
            next_q = self.target(stack("next_observation"), optional("next_context"))
            next_feasible = stack("next_feasible").bool()
            if not next_feasible.any(-1)[~terminated].all():
                raise ValueError("empty next feasible action set")
            next_max = next_q.masked_fill(~next_feasible, -torch.inf).amax(-1)
            targets = rewards + PAPER.gamma * torch.where(terminated, torch.zeros_like(next_max), next_max)
        td_loss = (q.gather(-1, actions[:, None]).squeeze(-1) - targets).square().mean()
        priors = optional("prior")
        if self.use_spi and priors is None:
            raise ValueError("SPI prior")
        kl = prior_kl(q / self.temperature, priors).mean() if self.use_spi else q.new_zeros(())
        loss = td_loss + (self.adaptive_kl.coefficient * kl if self.use_spi else 0)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite learning objective")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.target_update_interval == 0:
            self.target.load_state_dict(self.network.state_dict())
        with torch.no_grad():
            deviation = float(prior_kl(self.network(observations, contexts) / self.temperature, priors).mean()) if self.use_spi else 0.0
        if self.use_spi:
            self.adaptive_kl.update(deviation)
        return {"loss": float(loss.detach()), "kl": deviation}
