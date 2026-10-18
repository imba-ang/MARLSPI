from collections import defaultdict
from types import MappingProxyType
from typing import Mapping
import numpy as np
from ..config import PAPER


class FrozenPrior:
    def __init__(self, table: Mapping[tuple[int, ...], tuple[float, ...]]):
        normalized = {}
        for state, probabilities in table.items():
            values = np.asarray(probabilities, dtype=float)
            if values.shape != (PAPER.action_size,) or not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
                raise ValueError("prior distribution")
            normalized[tuple(state)] = tuple(values / values.sum())
        self._table = MappingProxyType(normalized)

    @property
    def states(self):
        return frozenset(self._table)

    def __call__(self, state):
        return np.array(self._table[tuple(state)], dtype=np.float32)




class CountPriorBuilder:
    def __init__(self):
        self._counts = defaultdict(lambda: np.zeros(PAPER.action_size, dtype=np.int64))

    def add(self, state, action):
        if not 0 <= int(action) < PAPER.action_size:
            raise ValueError("action")
        self._counts[tuple(state)][int(action)] += 1

    def extend(self, samples):
        for state, action in samples:
            self.add(state, action)

    def freeze(self) -> FrozenPrior:
        return FrozenPrior(self._counts)


class HybridPrior(FrozenPrior):
    def __init__(self, model: FrozenPrior, empirical: FrozenPrior, weight: float = PAPER.fusion_weight):
        if not 0 <= weight <= 1:
            raise ValueError("weight")
        states = model.states if weight == 1 else empirical.states if weight == 0 else model.states & empirical.states
        super().__init__({state: model(state) if weight == 1 else empirical(state) if weight == 0 else weight * model(state) + (1 - weight) * empirical(state) for state in states})

