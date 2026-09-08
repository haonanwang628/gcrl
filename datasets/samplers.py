"""Sampling laws are independent of model type and never expose actions."""
from __future__ import annotations
import numpy as np


class FutureSampler:
    def __init__(self, data, horizons=(1, 5, 10, 20), seed=0):
        self.data = data
        self.horizons = tuple(int(k) for k in horizons)
        if not self.horizons or min(self.horizons) < 1 or len(set(self.horizons)) != len(self.horizons):
            raise ValueError("Horizons must be distinct positive integers")
        self.rng = np.random.default_rng(seed)
        # Uniform k, then uniform eligible start state, then uniform H in [k+1,L-t].
        self.counts = {k: np.maximum(data.lengths - k, 0) for k in self.horizons}
        self.cumulative = {k: np.cumsum(v) for k, v in self.counts.items()}
        for k, cumulative in self.cumulative.items():
            if cumulative[-1] == 0:
                raise ValueError(f"No trajectory supports strict k < H for k={k}")

    def sample(self, batch_size, horizon=None):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if horizon is not None and horizon not in self.horizons:
            raise ValueError("Unconfigured horizon")
        ks = (self.rng.choice(self.horizons, batch_size) if horizon is None
              else np.full(batch_size, horizon, dtype=np.int64))
        trajectory = np.empty(batch_size, dtype=np.int64)
        ts = np.empty_like(trajectory)
        hs = np.empty_like(trajectory)
        for k in np.unique(ks):
            mask = ks == k
            cumulative = self.cumulative[k]
            draws = self.rng.integers(cumulative[-1], size=mask.sum())
            ids = np.searchsorted(cumulative, draws, side='right')
            before = np.r_[0, cumulative[:-1]][ids]
            t = draws - before
            remaining = self.data.lengths[ids] - t
            trajectory[mask], ts[mask] = ids, t
            hs[mask] = self.rng.integers(k + 1, remaining + 1)
        start = self.data.offsets[trajectory] + ts
        target, goal = start + ks, start + hs
        result = dict(observations=self.data.states[start], goals=self.data.states[goal],
                      targets=self.data.states[target], k=ks, H=hs, t=ts,
                      trajectory_id=self.data.trajectory_ids[trajectory],
                      current_index=start, target_index=target, goal_index=goal,
                      split=np.full(batch_size, self.data.split),
                      source_id=np.full(batch_size, self.data.source_id))
        for name, idx in [('current', start), ('target', target), ('goal', goal)]:
            result[name + '_source_row'] = self.data.source_rows[idx]
            result[name + '_is_next'] = self.data.is_next[idx]
        return result

    def validation_table(self, samples_per_horizon):
        batches = [self.sample(samples_per_horizon, k) for k in self.horizons]
        result = {key: np.concatenate([b[key] for b in batches]) for key in batches[0]}
        result['dataset_fingerprint'] = np.full(len(result['k']), self.data.fingerprint)
        return result


class TemporalSampler:
    """TempDATA representation-only goal sampling, NOT learner relabeling.

    Random goals may belong to another trajectory of the SAME split, as in
    TempDATA. Only future-model tuples require all three states in one trajectory.
    Success is current STATE INDEX equality, preserving the representation loss.
    """
    def __init__(self, data, seed=0, discount=0.99, p_current=0.0, p_future=0.625):
        if not (0 < discount < 1 and 0 <= p_current <= 1 and 0 <= p_future <= 1 - p_current):
            raise ValueError("Invalid temporal sampling probabilities")
        self.data, self.rng = data, np.random.default_rng(seed)
        self.discount, self.p_current, self.p_future = discount, p_current, p_future
        self.cumulative = np.cumsum(data.lengths)

    def sample(self, batch_size):
        draws = self.rng.integers(self.cumulative[-1], size=batch_size)
        ids = np.searchsorted(self.cumulative, draws, side='right')
        t = draws - np.r_[0, self.cumulative[:-1]][ids]
        current = self.data.offsets[ids] + t
        final = self.data.offsets[ids + 1] - 1
        future = np.minimum(current + self.rng.geometric(1 - self.discount, batch_size), final)
        random = self.rng.integers(len(self.data.states), size=batch_size)
        choice = self.rng.random(batch_size)
        goal = np.where(choice < self.p_current, current,
                        np.where(choice < self.p_current + self.p_future, future, random))
        return dict(observations=self.data.states[current], next_observations=self.data.states[current + 1],
                    goals=self.data.states[goal], success=(current == goal).astype(np.float32))
