from torch import nn


class FeatureModulation(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 128))
        self.shift = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 128))

    def forward(self, hidden, context):
        return self.scale(context) * hidden + self.shift(context)
