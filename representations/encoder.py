"""PyTorch adaptation of TempDATA/src/special_networks.py (see SOURCES.md)."""
import torch
from torch import nn


def representation_mlp(input_dim, hidden_dims, output_dim):
    layers = []
    for width in hidden_dims:
        layers.extend([nn.Linear(input_dim, width), nn.GELU(approximate='tanh'),
                       nn.LayerNorm(width, eps=1e-6)])
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    net = nn.Sequential(*layers)
    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return net


class TemporalEncoder(nn.Module):
    """Twin independent phi networks; downstream representation uses head zero."""
    def __init__(self, observation_dim, latent_dim=32, hidden_dims=(512, 512, 512)):
        super().__init__()
        self.heads = nn.ModuleList([representation_mlp(observation_dim, hidden_dims, latent_dim)
                                    for _ in range(2)])

    def forward(self, observations):
        return self.heads[0](observations)

    def value(self, observations, goals):
        return torch.stack([-torch.sqrt(((head(observations) - head(goals)) ** 2)
                                         .sum(-1).clamp_min(1e-6)) for head in self.heads])
