"""Standalone state decoder; latent dynamics."""
from torch import nn
from .encoder import representation_mlp


class Decoder(nn.Module):
    def __init__(self, observation_dim, latent_dim=32, hidden_dims=(512, 512, 512)):
        super().__init__()
        self.net = representation_mlp(latent_dim, hidden_dims, observation_dim)

    def forward(self, latent):
        return self.net(latent)
