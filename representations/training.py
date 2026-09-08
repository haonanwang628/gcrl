"""Train temporal value, freeze encoder, train decoder, freeze both."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from ..common import (freeze, module_hash, tensor, save_json, seed_all,
                      check_finite, metrics_to_float, prepared_manifest)
from ..datasets.normalization import StateNormalizer
from ..datasets.samplers import TemporalSampler
from ..datasets.trajectories import Trajectories, assert_split_isolation
from .encoder import TemporalEncoder
from .decoder import Decoder
from .losses import temporal_loss, update_target


def normalized_batch(batch, normalizer, device):
    return {key: tensor(normalizer.normalize(x) if key != 'success' else x, device)
            for key, x in batch.items()}


def load_representation(directory, device='cpu'):
    directory = Path(directory)
    saved = torch.load(directory / 'encoder.pt', map_location=device, weights_only=True)
    decoder_saved = torch.load(directory / 'decoder.pt', map_location=device, weights_only=True)
    encoder = TemporalEncoder(**saved['architecture']).to(device)
    decoder = Decoder(**saved['architecture']).to(device)
    encoder.load_state_dict(saved['state_dict'])
    decoder.load_state_dict(decoder_saved['state_dict'])
    normalizer = StateNormalizer.load(directory / 'normalizer.npz')
    if decoder_saved['encoder_hash'] != module_hash(encoder):
        raise ValueError('Decoder belongs to a different encoder')
    if saved['train_fingerprint'] != normalizer.train_fingerprint:
        raise ValueError('Normalizer belongs to a different training dataset')
    return freeze(encoder), freeze(decoder), normalizer


def train_representation(data_dir, output_dir, config, device='cpu'):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite representation artifacts: {output_dir}')
    train, validation = (Trajectories.load(data_dir / f'{s}.npz') for s in ('train', 'validation'))
    assert_split_isolation(train, validation)
    manifest = prepared_manifest(data_dir, config, train, validation)
    cfg = config['representation']
    seed_all(config['seed'], config['threads'])
    normalizer = StateNormalizer.fit(train)
    arch = dict(observation_dim=train.states.shape[1], latent_dim=cfg['latent_dim'],
                hidden_dims=list(cfg['hidden_dims']))
    encoder = TemporalEncoder(**arch).to(device)
    target = freeze(deepcopy(encoder))
    optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg['lr'], eps=1e-8)
    kwargs = dict(discount=cfg['discount'], p_current=cfg['p_current'], p_future=cfg['p_future'])
    sampler = TemporalSampler(train, config['seed'] + 11, **kwargs)
    val_sampler = TemporalSampler(validation, config['seed'] + 12, **kwargs)
    # Fixed representation validation batch size: native smooth loss depends on it.
    val_batch = normalized_batch(val_sampler.sample(cfg['batch_size']), normalizer, device)
    loss_kwargs = {k: cfg[k] for k in ('discount', 'expectile', 'smoothing_coef')}
    history = []
    for step in range(1, cfg['encoder_steps'] + 1):
        batch = normalized_batch(sampler.sample(cfg['batch_size']), normalizer, device)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = temporal_loss(encoder, target, batch, **loss_kwargs)
        check_finite(loss)
        loss.backward()
        update_target(target, encoder, cfg['tau'])
        optimizer.step()
        if step == 1 or step % cfg['log_every'] == 0 or step == cfg['encoder_steps']:
            with torch.no_grad():
                _, metrics = temporal_loss(encoder, target, val_batch, **loss_kwargs)
            record = dict(stage='encoder', step=step, **metrics_to_float(metrics))
            history.append(record)
            print(record, flush=True)
    freeze(encoder)
    frozen_hash = module_hash(encoder)
    decoder = Decoder(**arch).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=cfg['lr'], eps=1e-8)
    rng = np.random.default_rng(config['seed'] + 13)
    val_indices = rng.integers(len(validation.states), size=cfg['validation_samples'])
    val_states = tensor(normalizer.normalize(validation.states[val_indices]), device)
    with torch.no_grad():
        val_latents = encoder(val_states)
    for step in range(1, cfg['decoder_steps'] + 1):
        ids = rng.integers(len(train.states), size=cfg['batch_size'])
        states = tensor(normalizer.normalize(train.states[ids]), device)
        with torch.no_grad():
            latent = encoder(states)
        loss = (decoder(latent) - states).square().mean()
        check_finite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % cfg['log_every'] == 0 or step == cfg['decoder_steps']:
            with torch.no_grad():
                mse = (decoder(val_latents) - val_states).square().mean()
            record = dict(stage='decoder', step=step, normalized_state_mse=float(mse))
            history.append(record)
            print(record, flush=True)
    freeze(decoder)
    assert module_hash(encoder) == frozen_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    normalizer.save(output_dir / 'normalizer.npz')
    torch.save(dict(state_dict=encoder.state_dict(), architecture=arch,
                    target_state_dict=target.state_dict(), train_fingerprint=train.fingerprint),
               output_dir / 'encoder.pt')
    torch.save(dict(state_dict=decoder.state_dict(), encoder_hash=frozen_hash), output_dir / 'decoder.pt')
    save_json(output_dir / 'config.json', config)
    save_json(output_dir / 'validation_metrics.json', history)
    save_json(output_dir / 'provenance.json', dict(train_fingerprint=train.fingerprint,
              validation_fingerprint=validation.fingerprint, encoder_hash=frozen_hash,
              decoder_hash=module_hash(decoder), dataset_manifest=manifest))
    return encoder, decoder, normalizer
