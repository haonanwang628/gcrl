"""ADMPO/dynamics/arm.py ResBlock structure, without GRU/actions/dropout."""
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.activation = nn.SiLU()
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(x + self.activation(self.linear(x)))


def residual_mlp(input_dim, output_dim, width=256, depth=3):
    return nn.Sequential(nn.Linear(input_dim, width), nn.SiLU(),
                         *[ResBlock(width) for _ in range(depth)], nn.Linear(width, output_dim))
