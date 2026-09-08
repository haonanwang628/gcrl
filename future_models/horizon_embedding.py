"""New explicit lookup conditioning; not ADMPO's implicit sequence length."""
import torch
from torch import nn


class HorizonEmbedding(nn.Module):
    def __init__(self, horizons, embedding_dim=16):
        super().__init__()
        if not horizons or min(horizons) < 1 or len(set(horizons)) != len(horizons):
            raise ValueError('Invalid horizon vocabulary')
        self.register_buffer('horizons', torch.tensor(horizons, dtype=torch.long))
        self.embedding = nn.Embedding(len(horizons), embedding_dim)

    def forward(self, k):
        matches = k[:, None] == self.horizons[None]
        if not bool(matches.any(1).all()):
            raise ValueError('Unconfigured horizon; embedding interpolation is not defined')
        return self.embedding(matches.to(torch.long).argmax(1))
