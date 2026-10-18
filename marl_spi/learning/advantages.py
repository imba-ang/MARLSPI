import torch
from ..config import PAPER, LearningConfig


def generalized_advantage(rewards, values, next_values, terminated, truncated, gae_lambda: float = LearningConfig().gae_lambda, gamma: float = PAPER.gamma):
    if not 0 <= gae_lambda <= 1 or not 0 <= gamma <= 1:
        raise ValueError("GAE parameters")
    if len({tuple(x.shape) for x in (rewards, values, next_values, terminated, truncated)}) != 1:
        raise ValueError("trajectory shape")
    deltas = rewards + gamma * (~terminated).to(values.dtype) * next_values - values
    advantages = torch.zeros_like(values)
    carry = torch.zeros((), dtype=values.dtype, device=values.device)
    for index in reversed(range(len(rewards))):
        continuation = (~(terminated[index] | truncated[index])).to(values.dtype)
        carry = deltas[index] + gamma * gae_lambda * continuation * carry
        advantages[index] = carry
    return advantages.detach(), (advantages + values).detach()
