import torch
from torch import nn
from ..config import PAPER, RUNTIME


class ContextEncoder(nn.Module):
    def __init__(self, role_input_size: int = RUNTIME.role_input_size, phase_input_size: int = RUNTIME.phase_input_size):
        super().__init__()
        self.role = nn.Sequential(nn.Linear(role_input_size, 32), nn.ReLU(), nn.Linear(32, 16))
        self.interaction = nn.Sequential(nn.Linear(PAPER.observation_size, 64), nn.ReLU(), nn.Linear(64, 32))
        self.phase = nn.Sequential(nn.Linear(phase_input_size, 32), nn.ReLU(), nn.Linear(32, 16))
        self.role_projection = nn.Linear(16, 32)
        self.gate = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 32), nn.Sigmoid())

    def forward(self, role, observation, phase):
        r = self.role(role)
        m = self.interaction(observation)
        f = self.phase(phase)
        gate = self.gate(torch.cat((r, m, f), dim=-1))
        return torch.cat((gate * m + (1 - gate) * self.role_projection(r), f), dim=-1)
