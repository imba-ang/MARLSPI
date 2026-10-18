from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Transition:
    observation: torch.Tensor
    action: int
    reward: float
    next_observation: torch.Tensor
    terminated: bool
    truncated: bool
    abstract_state: tuple[int, ...] | None
    feasible: torch.Tensor
    weights: torch.Tensor
    behavior_log_probability: torch.Tensor
    value: torch.Tensor
    next_value: torch.Tensor
    context: torch.Tensor | None
    prior: torch.Tensor | None


@dataclass(frozen=True)
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    feasible: torch.Tensor
    weights: torch.Tensor
    behavior_log_probabilities: torch.Tensor
    values: torch.Tensor
    next_values: torch.Tensor
    contexts: torch.Tensor | None
    priors: torch.Tensor | None


class RolloutBuffer:
    def __init__(self):
        self._transitions = []

    def __len__(self):
        return len(self._transitions)

    def append(self, transition: Transition):
        self._transitions.append(transition)

    def clear(self):
        self._transitions.clear()

    def batch(self) -> RolloutBatch:
        if not self._transitions:
            raise ValueError("empty rollout")
        rows = self._transitions
        device = rows[0].observation.device
        def stack(name):
            return torch.stack([getattr(row, name).detach() for row in rows])
        def optional_stack(name):
            values = [getattr(row, name) for row in rows]
            if all(value is None for value in values):
                return None
            if any(value is None for value in values):
                raise ValueError(name)
            return stack(name)
        return RolloutBatch(stack("observation"), torch.tensor([r.action for r in rows], device=device), torch.tensor([r.reward for r in rows], device=device), torch.tensor([r.terminated for r in rows], device=device), torch.tensor([r.truncated for r in rows], device=device), stack("feasible"), stack("weights"), stack("behavior_log_probability"), stack("value"), stack("next_value"), optional_stack("context"), optional_stack("prior"))
