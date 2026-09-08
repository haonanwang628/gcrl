"""Configuration, reproducibility, artifacts, and frozen module helpers."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import random
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent


def merge(base, override):
    result = dict(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and key in result else value
    return result


def load_config(path=None):
    with open(ROOT / 'configs/default.yaml') as f:
        config = yaml.safe_load(f)
    if path:
        with open(path) as f:
            config = merge(config, yaml.safe_load(f))
    validate_config(config)
    return config


def validate_config(config):
    horizons = config['horizons']
    if not horizons or any(type(k) is not int or k < 1 for k in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError('horizons must contain distinct positive integers')
    positive = {'representation': ('latent_dim', 'batch_size', 'encoder_steps', 'decoder_steps',
                                  'validation_samples', 'log_every'),
                'future_model': ('width', 'depth', 'embedding_dim', 'batch_size', 'steps', 'log_every'),
                'evaluation': ('batch_size', 'timing_batch_size', 'timing_repeats'),
                'dataset': ('validation_samples_per_horizon',)}
    for section, keys in positive.items():
        for key in keys:
            if type(config[section][key]) is not int or config[section][key] < 1:
                raise ValueError(f'{section}.{key} must be a positive integer')
    if config['evaluation']['warmup'] < 0:
        raise ValueError('warmup must be nonnegative')


def save_json(path, content):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(content, f, indent=2, allow_nan=False)


def read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def prepared_manifest(directory, config, *splits):
    manifest = read_json(Path(directory) / 'manifest.json')
    if not manifest.get('fixture', False) and manifest['env_name'] != config['dataset']['env_name']:
        raise ValueError('Prepared dataset environment does not match the training/evaluation config')
    for data in splits:
        if manifest[data.split]['fingerprint'] != data.fingerprint:
            raise ValueError('Prepared dataset differs from its saved manifest')
    return manifest


def seed_all(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def freeze(module):
    module.eval()
    module.requires_grad_(False)
    for p in module.parameters():
        p.grad = None
    return module


def module_hash(module):
    h = hashlib.sha256()
    for name, p in module.state_dict().items():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def tensor(array, device):
    return torch.as_tensor(np.array(array, copy=True), dtype=torch.float32, device=device)


def versions():
    result = {}
    for name in ('torch', 'numpy', 'ogbench', 'matplotlib', 'PyYAML', 'pytest'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def check_finite(loss):
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite training loss; checkpoint not accepted')


def metrics_to_float(metrics):
    return {key: float(value.detach().cpu()) for key, value in metrics.items()}
