import torch
from torch import nn
from ..config import PAPER
from .modulation import FeatureModulation


class ModulatedNetwork(nn.Module):
    def __init__(self, output_size: int = PAPER.action_size, modulated: bool = True):
        super().__init__()
        self.hidden = nn.Linear(PAPER.observation_size, PAPER.hidden_size)
        self.output = nn.Linear(PAPER.hidden_size, output_size)
        self.modulation = FeatureModulation() if modulated else None

    def forward(self, observation, context=None):
        hidden = torch.relu(self.hidden(observation))
        if self.modulation is not None:
            if context is None:
                raise ValueError("context")
            hidden = self.modulation(hidden, context)
        return self.output(hidden)

    def generated_parameters(self):
        return tuple(self.hidden.parameters()) + tuple(self.output.parameters())

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.generated_parameters())

    def initialize(self, flat):
        if flat.ndim != 1 or flat.numel() != self.parameter_count:
            raise ValueError("generated parameters")
        offset = 0
        with torch.no_grad():
            for parameter in self.generated_parameters():
                size = parameter.numel()
                parameter.copy_(flat[offset:offset + size].reshape_as(parameter))
                offset += size


class ActorCritic(nn.Module):
    def __init__(self, modulated: bool = True):
        super().__init__()
        self.actor = ModulatedNetwork(PAPER.action_size, modulated)
        self.critic = ModulatedNetwork(1, modulated)

    def forward(self, observation, context=None):
        return self.actor(observation, context), self.critic(observation, context).squeeze(-1)
