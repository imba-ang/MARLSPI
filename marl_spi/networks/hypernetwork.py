import copy
import torch
from torch import nn
from .actor_critic import ActorCritic
from .context import ContextEncoder
from .modulation import FeatureModulation
from ..config import PAPER, RUNTIME


class ContextualInitializer(nn.Module):
    def __init__(self, role_input_size: int = RUNTIME.role_input_size, phase_input_size: int = RUNTIME.phase_input_size):
        super().__init__()
        self.encoder = ContextEncoder(role_input_size, phase_input_size)
        template = ActorCritic()
        self.actor_generator = nn.Sequential(nn.Linear(48, 128), nn.ReLU(), nn.Linear(128, template.actor.parameter_count))
        self.critic_generator = nn.Sequential(nn.Linear(48, 128), nn.ReLU(), nn.Linear(128, template.critic.parameter_count))
        self.actor_modulation = FeatureModulation()
        self.critic_modulation = FeatureModulation()

    def forward(self, role, observation, phase):
        context = self.encoder(role, observation, phase)
        return self.actor_generator(context), self.critic_generator(context), context

    def initialize(self, role, observation, phase):
        actor, critic, context = self(role, observation, phase)
        network = ActorCritic().to(device=actor.device, dtype=actor.dtype)
        network.actor.initialize(actor)
        network.critic.initialize(critic)
        network.actor.modulation = copy.deepcopy(self.actor_modulation)
        network.critic.modulation = copy.deepcopy(self.critic_modulation)
        return network, context.detach()

    @staticmethod
    def generated_forward(observation, flat, context, modulation, output_size=PAPER.action_size):
        hidden_weight = flat[..., :2688].reshape(*flat.shape[:-1], 128, 21)
        hidden_bias = flat[..., 2688:2816]
        output_end = 2816 + output_size * 128
        output_weight = flat[..., 2816:output_end].reshape(*flat.shape[:-1], output_size, 128)
        output_bias = flat[..., output_end:]
        hidden = torch.relu((hidden_weight @ observation.unsqueeze(-1)).squeeze(-1) + hidden_bias)
        hidden = modulation(hidden, context)
        return (output_weight @ hidden.unsqueeze(-1)).squeeze(-1) + output_bias

    def policy_value(self, role, observation, phase):
        actor, critic, context = self(role, observation, phase)
        logits = self.generated_forward(observation, actor, context, self.actor_modulation, 5)
        value = self.generated_forward(observation, critic, context, self.critic_modulation, 1).squeeze(-1)
        return logits, value
