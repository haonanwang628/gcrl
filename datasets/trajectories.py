"""Reconstruct states from explicit REAL (observation, next_observation) pairs.

State provenance is (source_id, source_row, is_next). Final next observations
are retained even when they have no corresponding observation/action row.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np


@dataclass(frozen=True)
class Trajectories:
    states: np.ndarray
    offsets: np.ndarray  # Half-open state intervals, one extra final offset.
    source_rows: np.ndarray
    is_next: np.ndarray
    trajectory_ids: np.ndarray  # Stable within the original source.
    split: str
    source_id: str

    def __post_init__(self):
        if self.states.ndim != 2 or len(self.states) == 0 or not np.isfinite(self.states).all():
            raise ValueError("Expected nonempty, finite vector observations")
        if (self.offsets[0] != 0 or self.offsets[-1] != len(self.states)
                or np.any(np.diff(self.offsets) < 2)):
            raise ValueError("Each trajectory must contain at least one real transition")
        if len(self.trajectory_ids) != len(self.offsets) - 1:
            raise ValueError("Invalid trajectory IDs")
        if len(self.source_rows) != len(self.states) or len(self.is_next) != len(self.states):
            raise ValueError("Invalid state provenance")
        for array in (self.states, self.offsets, self.source_rows, self.is_next, self.trajectory_ids):
            array.flags.writeable = False

    @property
    def lengths(self):
        return np.diff(self.offsets) - 1  # Transition counts, not state counts.

    @property
    def fingerprint(self):
        h = hashlib.sha256()
        h.update(self.source_id.encode())
        h.update(self.split.encode())
        for x in (self.states, self.offsets, self.source_rows, self.is_next, self.trajectory_ids):
            h.update(np.ascontiguousarray(x).tobytes())
        return h.hexdigest()

    def save(self, path):
        np.savez_compressed(path, states=self.states, offsets=self.offsets,
                            source_rows=self.source_rows, is_next=self.is_next,
                            trajectory_ids=self.trajectory_ids, split=self.split,
                            source_id=self.source_id)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            return cls(**{k: str(f[k]) if k in ('split', 'source_id') else f[k]
                          for k in f.files})

    def subset(self, ids, split):
        ids = np.asarray(ids, dtype=np.int64)
        if not len(ids) or len(np.unique(ids)) != len(ids):
            raise ValueError("Select a nonempty set of distinct trajectories")
        indices = np.concatenate([np.arange(self.offsets[i], self.offsets[i + 1]) for i in ids])
        offsets = np.r_[0, np.cumsum(np.diff(self.offsets)[ids])]
        return Trajectories(self.states[indices], offsets, self.source_rows[indices],
                            self.is_next[indices], self.trajectory_ids[ids], split, self.source_id)


def reconstruct(dataset, split, source_id):
    """Accept regular transitions. Compact arrays must first use OGBench's loader.

    Boundary flags refer to the outgoing transition. Never infer next state by
    shifting a regular array: doing so discards final states or crosses resets.
    """
    if 'next_observations' not in dataset:
        raise ValueError("Explicit next_observations required; load OGBench with compact_dataset=False")
    obs = np.asarray(dataset['observations'], dtype=np.float32)
    nxt = np.asarray(dataset['next_observations'], dtype=np.float32)
    if obs.shape != nxt.shape or obs.ndim != 2 or not len(obs):
        raise ValueError("Invalid observation/next_observation shapes")
    n = len(obs)
    end = np.zeros(n, dtype=bool)
    found = False
    for key in ('terminals', 'terminated', 'truncated', 'timeouts', 'truncations'):
        if key in dataset:
            flags = np.asarray(dataset[key]).reshape(-1)
            if len(flags) != n or not np.isin(flags, [0, 1]).all():
                raise ValueError(f"Invalid {key} flags")
            end |= flags.astype(bool)
            found = True
    if not found:
        raise ValueError("At least one terminal/truncation flag array is required")
    end[-1] = True  # File boundary is always a boundary; final nxt is still real.
    if np.any(np.any(nxt[:-1] != obs[1:], axis=1) & ~end[:-1]):
        raise ValueError("Unmarked observation discontinuity; cannot establish a real trajectory")
    rows = np.asarray(dataset.get('source_rows', np.arange(n)), dtype=np.int64)
    if rows.shape != (n,):
        raise ValueError("Invalid source_rows")
    starts = np.r_[0, np.flatnonzero(end)[:-1] + 1]
    ends = np.flatnonzero(end)
    states, provenance, kinds = [], [], []
    for start, last in zip(starts, ends):
        states.append(np.concatenate([obs[start:last + 1], nxt[last:last + 1]]))
        provenance.append(np.r_[rows[start:last + 1], rows[last]])
        kinds.append(np.r_[np.zeros(last - start + 1, dtype=bool), True])
    return Trajectories(np.concatenate(states), np.r_[0, np.cumsum(ends - starts + 2)],
                        np.concatenate(provenance), np.concatenate(kinds),
                        np.arange(len(starts)), split, str(source_id))


def split_trajectories(data, validation_fraction=0.1, seed=0):
    """Fallback ONLY for a source without official validation; split whole episodes."""
    if not 0 < validation_fraction < 1 or len(data.lengths) < 2:
        raise ValueError("Need >=2 trajectories and a validation fraction in (0,1)")
    order = np.random.default_rng(seed).permutation(len(data.lengths))
    count = min(len(order) - 1, max(1, int(round(len(order) * validation_fraction))))
    return data.subset(np.sort(order[count:]), 'train'), data.subset(np.sort(order[:count]), 'validation')


def assert_split_isolation(train, validation):
    if train.split != 'train' or validation.split != 'validation':
        raise ValueError("Incorrect split labels")
    if train.source_id == validation.source_id:
        if np.intersect1d(train.trajectory_ids, validation.trajectory_ids).size:
            raise ValueError("A source trajectory appears in both splits")
    if train.states.shape[1:] != validation.states.shape[1:]:
        raise ValueError("Train/validation observation dimensions differ")
