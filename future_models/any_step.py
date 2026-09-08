"""A single forward call predicts the requested endpoint, without rollout."""
import torch
from torch import nn
from .networks import residual_mlp
from .horizon_embedding import HorizonEmbedding


class AnyStepModel(nn.Module):
    def __init__(self, state_dim, horizons=(1, 5, 10, 20), width=256, depth=3, embedding_dim=16):
        super().__init__()
        self.horizon = HorizonEmbedding(horizons, embedding_dim)
        self.delta = residual_mlp(2 * state_dim + embedding_dim, state_dim, width, depth)

    def forward(self, current, goal, k):
        return current + self.delta(torch.cat([current, goal, self.horizon(k)], dim=-1))
