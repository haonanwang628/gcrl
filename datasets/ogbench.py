"""Official OGBench dataset-only loading; no environment is created or stepped."""
from pathlib import Path
import hashlib
from urllib.error import URLError
import numpy as np
from .trajectories import reconstruct, assert_split_isolation

SUPPORTED_ENVS = ('pointmaze-medium-v0', 'pointmaze-large-v0',
                  'antmaze-medium-v0', 'antmaze-large-v0')


def dataset_name(env_name, dataset_type='navigate'):
    if env_name not in SUPPORTED_ENVS:
        raise ValueError(f"Unsupported V1 environment: {env_name}")
    if dataset_type not in ('navigate', 'stitch', 'explore'):
        raise ValueError("Specify an official dataset type, distinct from the environment ID")
    prefix, version = env_name.rsplit('-', 1)
    return f'{prefix}-{dataset_type}-{version}'


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_official(env_name, dataset_type='navigate', dataset_dir='~/.ogbench/data'):
    import ogbench
    name = dataset_name(env_name, dataset_type)
    root = Path(dataset_dir).expanduser()
    try:
        train, validation = ogbench.make_env_and_datasets(
            name, dataset_dir=str(root), dataset_only=True, compact_dataset=False)
    except URLError as exc:
        raise RuntimeError(
            f'Official OGBench download failed: {exc}. Place the original {name}.npz '
            f'and {name}-val.npz in {root}, then retry with --dataset-dir. '
            'No substitute data was generated.') from exc
    result, sources = [], []
    for split, suffix, arrays in [('train', '', train), ('validation', '-val', validation)]:
        path = root / f'{name}{suffix}.npz'
        digest = file_hash(path)
        # Official regular loader removes terminal sentinel rows; retain RAW file indices.
        with np.load(path, allow_pickle=False) as raw:
            if not bool(raw['terminals'][-1]):
                raise ValueError('Official compact file must end in a terminal sentinel state')
            mask = ~raw['terminals'].astype(bool)
            arrays['source_rows'] = np.flatnonzero(mask)
            # Preserve extra transition boundaries if a local file supplies them.
            for key in ('timeouts', 'truncated', 'truncations', 'terminated'):
                if key in raw:
                    arrays[key] = raw[key][mask]
        result.append(reconstruct(arrays, split, digest))
        sources.append(dict(split=split, path=str(path.resolve()), sha256=digest))
    assert_split_isolation(*result)
    return *result, dict(env_name=env_name, dataset_name=name, sources=sources,
                         loader='ogbench.make_env_and_datasets(dataset_only=True, compact_dataset=False)')
