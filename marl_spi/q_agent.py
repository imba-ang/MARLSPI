import copy
from dataclasses import dataclass
import torch
from .actions import Action
from .config import DQN
from .learning.idqn import IndependentDQN, ReplayBuffer, QTransition
from .networks.actor_critic import ModulatedNetwork
from .policies.soft_mask import mask_weights
from .priors.abstraction import StateAbstraction


@dataclass(frozen=True)
class QDecision:
    action: Action
    observation: torch.Tensor
    context: torch.Tensor | None
    prior: torch.Tensor | None


class LocalQAgent:
    def __init__(self, learner: IndependentDQN, capacity: int = DQN.capacity, prior=None, abstraction=None, context_encoder=None, role=None, phase=None):
        if learner.use_spi != (prior is not None):
            raise ValueError("SPI configuration")
        self.learner = learner
        self.prior = prior
        self.abstraction = abstraction if abstraction is not None else StateAbstraction() if prior is not None else None
        self.replay = ReplayBuffer(capacity)
        self.context_encoder = copy.deepcopy(context_encoder).to(self.device).requires_grad_(False) if context_encoder is not None else None
        self.role = role.detach().clone() if role is not None else None
        self.phase = phase.detach().clone() if phase is not None else None
        if self.context_encoder is not None and (self.role is None or self.phase is None):
            raise ValueError("role, phase")

    @property
    def device(self):
        return next(self.learner.network.parameters()).device

    def tensor(self, observation):
        return torch.as_tensor(observation.vector(), dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def context(self, observation):
        return None if self.context_encoder is None else self.context_encoder(self.role.to(self.device), observation, self.phase.to(self.device))

    @torch.no_grad()
    def act(self, observation, feasible, risk=None, epsilon=DQN.epsilon):
        vector = self.tensor(observation)
        context = self.context(vector)
        prior = None if self.prior is None else torch.as_tensor(self.prior(self.abstraction(observation)), device=self.device)
        weights = None
        if prior is not None:
            if risk is None:
                raise ValueError("distance risk")
            weights = mask_weights(prior, torch.as_tensor(risk, dtype=vector.dtype, device=self.device))
        action = self.learner.select(vector, torch.as_tensor(feasible, device=self.device), context, weights, epsilon)
        return QDecision(Action(action), vector.detach().clone(), context, prior)

    @torch.no_grad()
    def observe(self, decision: QDecision, reward: float, next_observation, next_feasible, terminated: bool):
        vector = self.tensor(next_observation)
        self.replay.append(QTransition(decision.observation, int(decision.action), reward, vector, terminated, decision.context, self.context(vector), decision.prior, torch.as_tensor(next_feasible, dtype=torch.bool, device=self.device).clone()))

    def update(self):
        return self.learner.update(self.replay)




def create_q_agent(config=None, capacity: int = DQN.capacity, temperature: float = DQN.temperature, target_update_interval: int = DQN.target_update_interval, prior=None, abstraction=None, initializer=None, initial_observation=None, role=None, phase=None, device="cpu"):
    encoder = None
    if initializer is None:
        network = ModulatedNetwork(5, modulated=False).to(device)
    else:
        if initial_observation is None or role is None or phase is None:
            raise ValueError("initial context")
        initializer = copy.deepcopy(initializer).to(device)
        role = torch.as_tensor(role, dtype=torch.float32, device=device)
        phase = torch.as_tensor(phase, dtype=torch.float32, device=device)
        actor_critic, _ = initializer.initialize(role, torch.as_tensor(initial_observation.vector(), device=device), phase)
        network = actor_critic.actor
        encoder = initializer.encoder
    learner = IndependentDQN(network, config, prior is not None, temperature, target_update_interval)
    return LocalQAgent(learner, capacity, prior, abstraction, encoder, role, phase)
