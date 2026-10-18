import copy
from dataclasses import dataclass
import torch
from .actions import Action
from .config import PAPER
from .learning.rollout import RolloutBuffer, Transition
from .policies.soft_mask import mask_weights, masked_distribution
from .priors.abstraction import StateAbstraction


@dataclass(frozen=True)
class Decision:
    action: Action
    observation: torch.Tensor
    abstract_state: tuple[int, ...] | None
    feasible: torch.Tensor
    weights: torch.Tensor
    log_probability: torch.Tensor
    value: torch.Tensor
    context: torch.Tensor | None
    prior: torch.Tensor | None
    sampled: bool


class LocalAgent:
    def __init__(self, network, learner, prior=None, abstraction=None, context_encoder=None, role=None, phase=None):
        if learner.network is not network:
            raise ValueError("learner network")
        if learner.use_spi != (prior is not None):
            raise ValueError("SPI configuration")
        self.network = network
        self.learner = learner
        self.prior = prior
        self.abstraction = abstraction if abstraction is not None else StateAbstraction() if prior is not None else None
        self.context_encoder = copy.deepcopy(context_encoder).to(self.device).requires_grad_(False) if context_encoder is not None else None
        self.role = role.detach().clone() if role is not None else None
        self.phase = phase.detach().clone() if phase is not None else None
        if self.context_encoder is not None and (self.role is None or self.phase is None):
            raise ValueError("role, phase")
        self.rollout = RolloutBuffer()

    @property
    def device(self):
        return next(self.network.parameters()).device

    def tensor(self, observation):
        return torch.as_tensor(observation.vector(), dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def context(self, observation):
        if self.context_encoder is None:
            return None
        return self.context_encoder(self.role.to(self.device), observation, self.phase.to(self.device))

    @torch.no_grad()
    def act(self, observation, feasible, risk=None, deterministic: bool = False) -> Decision:
        vector = self.tensor(observation)
        context = self.context(vector)
        logits, value = self.network(vector, context)
        feasible = torch.as_tensor(feasible, dtype=torch.bool, device=self.device)
        state = self.abstraction(observation) if self.prior is not None else None
        prior = torch.as_tensor(self.prior(state), device=self.device) if self.prior is not None else None
        if prior is not None:
            if risk is None:
                raise ValueError("distance risk")
            weights = mask_weights(prior, torch.as_tensor(risk, dtype=logits.dtype, device=self.device))
        else:
            weights = torch.ones(PAPER.action_size, device=self.device)
        distribution = masked_distribution(logits, feasible, weights)
        action = distribution.probs.argmax() if deterministic else distribution.sample()
        return Decision(Action(int(action)), vector.detach().clone(), state, feasible.detach().clone(), weights.detach().clone(), distribution.log_prob(action).detach(), value.detach(), context, prior, not deterministic)

    @torch.no_grad()
    def observe(self, decision: Decision, reward: float, next_observation, terminated: bool, truncated: bool):
        if not decision.sampled:
            raise ValueError("on-policy rollout requires sampled actions")
        vector = self.tensor(next_observation)
        if terminated:
            next_value = torch.zeros((), device=self.device)
        else:
            _, next_value = self.network(vector, self.context(vector))
        self.rollout.append(Transition(decision.observation, int(decision.action), reward, vector, terminated, truncated, decision.abstract_state, decision.feasible, decision.weights, decision.log_probability, decision.value, next_value, decision.context, decision.prior))

    def update(self):
        return self.learner.update(self.rollout)




def create_actor_critic_agent(algorithm="ippo", config=None, prior=None, abstraction=None, initializer=None, initial_observation=None, role=None, phase=None, device="cpu"):
    from .learning.ippo import IndependentPPO
    from .learning.ia2c import IndependentA2C
    from .networks.actor_critic import ActorCritic
    algorithms = {"ippo": IndependentPPO, "ia2c": IndependentA2C}
    if algorithm not in algorithms:
        raise ValueError("algorithm")
    encoder = None
    if initializer is None:
        network = ActorCritic(modulated=False).to(device)
    else:
        if initial_observation is None or role is None or phase is None:
            raise ValueError("initial context")
        initializer = copy.deepcopy(initializer).to(device)
        role = torch.as_tensor(role, dtype=torch.float32, device=device)
        phase = torch.as_tensor(phase, dtype=torch.float32, device=device)
        observation = torch.as_tensor(initial_observation.vector(), device=device)
        network, _ = initializer.initialize(role, observation, phase)
        encoder = initializer.encoder
    learner = algorithms[algorithm](network, config, use_spi=prior is not None)
    return LocalAgent(network, learner, prior, abstraction, encoder, role, phase)
