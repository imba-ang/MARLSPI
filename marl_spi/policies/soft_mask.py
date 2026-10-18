import torch
from torch.distributions import Categorical
from ..config import PAPER, RUNTIME


def distance_risk(clearances, safe_clearance: float = RUNTIME.safe_clearance):
    if safe_clearance <= 0:
        raise ValueError("safe_clearance")
    if clearances.shape[-1] == 0:
        return torch.zeros(clearances.shape[:-1], dtype=clearances.dtype, device=clearances.device)
    return (1 - clearances / safe_clearance).clamp(0, 1).amax(dim=-1)


def mask_weights(prior, risk):
    if prior.shape != risk.shape or (risk < 0).any() or (risk > 1).any():
        raise ValueError("prior, risk")
    return ((prior + PAPER.epsilon_pi).pow(PAPER.alpha_pi) * torch.exp(-PAPER.alpha_d * risk)).clamp_min(PAPER.mu_min)


def masked_distribution(logits, feasible, weights):
    feasible = feasible.to(device=logits.device, dtype=torch.bool)
    weights = weights.to(device=logits.device, dtype=logits.dtype)
    if logits.shape != feasible.shape or logits.shape != weights.shape:
        raise ValueError("mask shape")
    if not feasible.any(dim=-1).all() or not torch.isfinite(weights).all() or (weights[feasible] <= 0).any():
        raise ValueError("mask support")
    return Categorical(logits=(logits + weights.clamp_min(torch.finfo(logits.dtype).tiny).log()).masked_fill(~feasible, -torch.inf))


def prior_kl(logits, prior):
    if prior.shape != logits.shape or (prior < 0).any() or not torch.isfinite(prior).all():
        raise ValueError("prior")
    if not torch.allclose(prior.sum(-1), torch.ones_like(prior.sum(-1)), atol=1e-5):
        raise ValueError("prior normalization")
    if (prior == 0).any():
        raise ValueError("KL requires positive SPI support")
    log_policy = logits.log_softmax(dim=-1)
    return (log_policy.exp() * (log_policy - prior.log())).sum(dim=-1)
