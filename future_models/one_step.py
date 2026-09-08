"""One-step parameterization, recursive endpoint supervision, fixed goal."""
import torch
from torch import nn
from .networks import residual_mlp


class OneStepModel(nn.Module):
    def __init__(self, state_dim, horizons=(1, 5, 10, 20), width=256, depth=3, embedding_dim=16):
        super().__init__()
        self.horizons = tuple(horizons)
        self.delta = residual_mlp(2 * state_dim, state_dim, width, depth)

    def step(self, current, goal):
        return current + self.delta(torch.cat([current, goal], dim=-1))

    def forward(self, current, goal, k):
        if not all(int(x) in self.horizons for x in k.unique()):
            raise ValueError('Unconfigured horizon')
        predicted = current
        for step in range(int(k.max())):
            # Inactive rows are held fixed; active rows only see their own prediction.
            proposal = self.step(predicted, goal)
            predicted = torch.where((k > step)[:, None], proposal, predicted)
        return predicted
