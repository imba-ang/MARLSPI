import torch
from torch.nn import functional as F
from ..config import PAPER


class BehaviorCloning:
    def __init__(self, actor):
        self.actor = actor
        self.optimizer = torch.optim.Adam(actor.parameters(), lr=PAPER.learning_rate)

    def update(self, observations, demonstrated_actions, contexts=None):
        logits = self.actor(observations, contexts)
        loss = F.cross_entropy(logits, demonstrated_actions)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite cloning objective")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())
