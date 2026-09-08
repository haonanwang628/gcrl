import numpy as np
import pytest
import torch
from any_step_mher_ogbench.datasets.trajectories import reconstruct


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


@pytest.fixture
def trajectories():
    def make(split='train', shift=0., lengths=(25, 31, 3)):
        states = [np.column_stack([np.arange(n + 1) + shift + 100 * i,
                                    np.arange(n + 1) * .5]) for i, n in enumerate(lengths)]
        arrays = dict(observations=np.concatenate([s[:-1] for s in states]),
                      next_observations=np.concatenate([s[1:] for s in states]),
                      terminals=np.concatenate([np.r_[np.zeros(n - 1), 1] for n in lengths]))
        return reconstruct(arrays, split, 'test-source-' + split)
    return make
