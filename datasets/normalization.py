"""Fixed train-only, invertible standardization (no clipping or whitening)."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class StateNormalizer:
    mean: np.ndarray
    scale: np.ndarray
    train_fingerprint: str

    def __post_init__(self):
        if (self.mean.shape != self.scale.shape or self.mean.ndim != 1
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all()
                or np.any(self.scale <= 0)):
            raise ValueError('Invalid normalizer statistics')
        self.mean.flags.writeable = False
        self.scale.flags.writeable = False

    @classmethod
    def fit(cls, data):
        if data.split != 'train':
            raise ValueError("Normalizer may only be fitted on the training split")
        return cls(data.states.mean(0, dtype=np.float64).astype(np.float32),
                   np.maximum(data.states.std(0, dtype=np.float64), 1e-6).astype(np.float32),
                   data.fingerprint)

    def normalize(self, states):
        return (states - self.mean) / self.scale

    def unnormalize(self, states):
        return states * self.scale + self.mean

    def save(self, path):
        np.savez(path, mean=self.mean, scale=self.scale, train_fingerprint=self.train_fingerprint)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            return cls(f['mean'], f['scale'], str(f['train_fingerprint']))
