"""All four controls consume the SAME tuple batch at every optimizer step."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from ..common import (seed_all, freeze, module_hash, tensor, save_json, read_json, check_finite,
                      prepared_manifest)
from ..datasets.samplers import FutureSampler
from ..datasets.trajectories import Trajectories
from ..representations.training import load_representation
from .any_step import AnyStepModel
from .one_step import OneStepModel
from .losses import endpoint_loss

MODEL_NAMES = ('raw_one_step', 'raw_any_step', 'latent_one_step', 'latent_any_step')


def build_models(state_dim, latent_dim, config):
    cfg = config['future_model']
    models = {}
    for name in MODEL_NAMES:
        cls = AnyStepModel if 'any_step' in name else OneStepModel
        models[name] = cls(state_dim=latent_dim if name.startswith('latent') else state_dim,
                           horizons=config['horizons'], width=cfg['width'], depth=cfg['depth'],
                           embedding_dim=cfg['embedding_dim'])
    return models


def load_models(directory, device='cpu'):
    directory = Path(directory)
    meta = read_json(directory / 'metadata.json')
    models = build_models(meta['state_dim'], meta['latent_dim'], meta['config'])
    for name, model in models.items():
        saved = torch.load(directory / f'{name}.pt', map_location=device, weights_only=True)
        if saved['tuple_stream_sha256'] != meta['tuple_stream_sha256']:
            raise ValueError('Model checkpoints were not trained with the same tuple stream')
        if saved['encoder_hash'] != meta['encoder_hash']:
            raise ValueError('Model checkpoint representation mismatch')
        model.load_state_dict(saved['state_dict'])
        models[name] = freeze(model.to(device))
    return models, meta


def train_future_models(data_dir, representation_dir, output_dir, config, device='cpu'):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite future-model artifacts: {output_dir}')
    train = Trajectories.load(data_dir / 'train.npz')
    manifest = prepared_manifest(data_dir, config, train)
    encoder, decoder, normalizer = load_representation(representation_dir, device)
    if normalizer.train_fingerprint != train.fingerprint:
        raise ValueError('Representation and future model must use the same training dataset')
    seed_all(config['seed'], config['threads'])
    encoder_hash, decoder_hash = module_hash(encoder), module_hash(decoder)
    latent_dim = encoder(tensor(normalizer.normalize(train.states[:1]), device)).shape[-1]
    models = {k: m.to(device) for k, m in build_models(train.states.shape[1], latent_dim, config).items()}
    cfg = config['future_model']
    optimizers = {k: torch.optim.Adam(m.parameters(), lr=cfg['lr']) for k, m in models.items()}
    sampler = FutureSampler(train, config['horizons'], config['seed'] + 21)
    tuple_hash = hashlib.sha256()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Stream a full audit to disk: memory stays O(batch), even for long runs.
    audit = np.lib.format.open_memmap(output_dir / 'training_tuple_indices.npy', mode='w+',
              dtype=np.int64, shape=(cfg['steps'], cfg['batch_size'], 6))
    history = []
    for step in range(1, cfg['steps'] + 1):
        batch = sampler.sample(cfg['batch_size'])
        indices = np.column_stack([batch[key] for key in ('trajectory_id', 'current_index',
                                                         'target_index', 'goal_index', 'k', 'H')])
        tuple_hash.update(indices.tobytes())
        # Full tuple audit is compact integer metadata, not synthetic transitions.
        audit[step - 1] = indices
        raw = [tensor(normalizer.normalize(batch[key]), device) for key in ('observations', 'goals', 'targets')]
        with torch.no_grad():
            latent = [encoder(x) for x in raw]
        horizons = torch.as_tensor(batch['k'], dtype=torch.long, device=device)
        losses = {}
        for name, model in models.items():
            current, goal, target = latent if name.startswith('latent') else raw
            optimizers[name].zero_grad(set_to_none=True)
            loss = endpoint_loss(model(current, goal, horizons), target)
            check_finite(loss)
            loss.backward()
            optimizers[name].step()
            losses[name] = float(loss.detach())
        if step == 1 or step % cfg['log_every'] == 0 or step == cfg['steps']:
            history.append(dict(step=step, **losses))
            print(history[-1], flush=True)
    assert module_hash(encoder) == encoder_hash and module_hash(decoder) == decoder_hash
    assert all(p.grad is None for m in (encoder, decoder) for p in m.parameters())
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        freeze(model)
        torch.save(dict(state_dict=model.state_dict(), tuple_stream_sha256=tuple_hash.hexdigest(),
                        encoder_hash=encoder_hash), output_dir / f'{name}.pt')
    audit.flush()
    save_json(output_dir / 'training_tuple_columns.json',
              dict(columns=['trajectory_id', 'current_index', 'target_index', 'goal_index', 'k', 'H'],
                   dataset_fingerprint=train.fingerprint, shape=list(audit.shape)))
    save_json(output_dir / 'training_metrics.json', history)
    save_json(output_dir / 'metadata.json', dict(config=config, state_dim=train.states.shape[1],
              latent_dim=latent_dim, encoder_hash=encoder_hash, decoder_hash=decoder_hash,
              train_fingerprint=train.fingerprint, tuple_stream_sha256=tuple_hash.hexdigest(),
              dataset_manifest=manifest))
    return models
