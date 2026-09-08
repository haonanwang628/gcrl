"""Matched validation evaluation; no environment creation or policy evaluation."""
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from ..common import tensor, module_hash, save_json, seed_all, prepared_manifest
from ..datasets.trajectories import Trajectories
from ..representations.training import load_representation
from ..future_models.training import load_models
from .reporting import summarize, endpoint_errors, write_csv


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark(fn, device, warmup, repeats, count):
    for _ in range(warmup):
        fn()
    synchronize(device)
    elapsed = []
    for _ in range(repeats):
        start = perf_counter()
        fn()
        synchronize(device)
        elapsed.append((perf_counter() - start) * 1000)
    return dict(batch_median_ms=float(np.median(elapsed)),
                amortized_ms_per_sample=float(np.median(elapsed) / count), batch_size=count)


def validate_table(table, validation, horizons):
    if set(np.unique(table['k'])) != set(horizons):
        raise ValueError('Validation table must cover exactly the configured horizons')
    if not np.all(table['split'] == 'validation'):
        raise ValueError('Evaluation requires validation tuples only')
    if not np.all(table['dataset_fingerprint'] == validation.fingerprint):
        raise ValueError('Validation tuple dataset fingerprint mismatch')
    current, target, goal = (table[k] for k in ('current_index', 'target_index', 'goal_index'))
    if any(np.any(index < 0) or np.any(index >= len(validation.states))
           for index in (current, target, goal)):
        raise ValueError('Validation state index out of bounds')
    ids = np.searchsorted(validation.offsets[1:], current, side='right')
    if (np.any(current < 0) or np.any(goal >= len(validation.states))
            or np.any(table['k'] < 1) or np.any(table['k'] >= table['H'])
            or not np.array_equal(target, current + table['k'])
            or not np.array_equal(goal, current + table['H'])
            or np.any(goal >= validation.offsets[ids + 1])):
        raise ValueError('Invalid ordering or cross-trajectory validation tuple')
    for field, indices in [('observations', current), ('targets', target), ('goals', goal)]:
        if not np.array_equal(table[field], validation.states[indices]):
            raise ValueError('Validation tuple states differ from real dataset')


@torch.inference_mode()
def evaluate(data_dir, representation_dir, model_dir, output_dir, config, device='cpu'):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite evaluation artifacts: {output_dir}')
    seed_all(config['seed'], config['threads'])
    encoder, decoder, normalizer = load_representation(representation_dir, device)
    models, meta = load_models(model_dir, device)
    if module_hash(encoder) != meta['encoder_hash'] or module_hash(decoder) != meta['decoder_hash']:
        raise ValueError('Models and representation checkpoints do not match')
    if normalizer.train_fingerprint != meta['train_fingerprint']:
        raise ValueError('Normalizer/future-model dataset mismatch')
    validation = Trajectories.load(data_dir / 'validation.npz')
    manifest = prepared_manifest(data_dir, config, validation)
    if meta['dataset_manifest']['validation']['fingerprint'] != validation.fingerprint:
        raise ValueError('Future model and evaluation validation split do not match')
    with np.load(data_dir / 'validation_tuples.npz', allow_pickle=False) as f:
        table = dict(f)
    validate_table(table, validation, meta['config']['horizons'])
    cfg = config['evaluation']
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, report, prediction_arrays = [], {}, {}
    scale = tensor(normalizer.scale, device)
    mean = tensor(normalizer.mean, device)
    for name, model in models.items():
        report[name] = {}
        all_predictions = np.empty_like(table['targets'])
        for k in meta['config']['horizons']:
            ids = np.flatnonzero(table['k'] == k)
            collected = {}
            for start in range(0, len(ids), cfg['batch_size']):
                selection = ids[start:start + cfg['batch_size']]
                current, goal, target = [tensor(normalizer.normalize(table[key][selection]), device)
                                         for key in ('observations', 'goals', 'targets')]
                horizons = torch.full((len(selection),), k, dtype=torch.long, device=device)
                latent_model = name.startswith('latent')
                if latent_model:
                    latent_target = encoder(target)
                    pred_latent = model(encoder(current), encoder(goal), horizons)
                    predicted = decoder(pred_latent)
                else:
                    predicted = model(current, goal, horizons)
                physical = (predicted * scale + mean).cpu().numpy()
                target_physical = table['targets'][selection]
                all_predictions[selection] = physical
                errors = endpoint_errors(predicted.cpu().numpy(), target.cpu().numpy(), 'normalized_state')
                errors.update(endpoint_errors(physical, target_physical, 'physical_state'))
                errors.update(endpoint_errors(physical[:, :2], target_physical[:, :2], 'maze_xy'))
                if latent_model:
                    errors.update(endpoint_errors(predicted.cpu().numpy(), target.cpu().numpy(),
                                                  'full_prediction_decoder_normalized'))
                    errors.update(endpoint_errors(physical, target_physical, 'full_prediction_decoder_physical'))
                    errors.update(endpoint_errors(physical[:, :2], target_physical[:, :2], 'full_prediction_decoder_xy'))
                    errors.update(endpoint_errors(pred_latent.cpu().numpy(), latent_target.cpu().numpy(), 'latent'))
                    reconstruction = decoder(latent_target)
                    errors.update(endpoint_errors(reconstruction.cpu().numpy(), target.cpu().numpy(), 'decoder_only_normalized'))
                    errors.update(endpoint_errors((reconstruction * scale + mean).cpu().numpy(),
                                                  target_physical, 'decoder_only_physical'))
                    errors.update(endpoint_errors((reconstruction * scale + mean).cpu().numpy()[:, :2],
                                                  target_physical[:, :2], 'decoder_only_xy'))
                for key, values in errors.items():
                    collected.setdefault(key, []).append(values)
            distributions = {key: summarize(np.concatenate(values)) for key, values in collected.items()}
            # Full decoded error is the latent model's normalized_state / physical_state / maze_xy error.
            # It is not computed by subtracting decoder-only errors.
            timing_ids = ids[:min(len(ids), cfg['timing_batch_size'])]
            current, goal = [tensor(normalizer.normalize(table[key][timing_ids]), device)
                             for key in ('observations', 'goals')]
            horizons = torch.full((len(timing_ids),), k, dtype=torch.long, device=device)
            if name.startswith('latent'):
                z, zg = encoder(current), encoder(goal)
                model_fn = lambda: model(z, zg, horizons)
                full_fn = lambda: decoder(model(encoder(current), encoder(goal), horizons)) * scale + mean
            else:
                model_fn = lambda: model(current, goal, horizons)
                full_fn = lambda: model(current, goal, horizons) * scale + mean
            timing = {label: benchmark(fn, device, cfg['warmup'], cfg['timing_repeats'], len(timing_ids))
                      for label, fn in [('model', model_fn), ('model_plus_representation', full_fn)]}
            parameters = dict(future_model=sum(p.numel() for p in model.parameters()),
                              frozen_encoder=sum(p.numel() for p in encoder.parameters()) if name.startswith('latent') else 0,
                              frozen_decoder=sum(p.numel() for p in decoder.parameters()) if name.startswith('latent') else 0)
            parameters['active_encoder_head'] = (sum(p.numel() for p in encoder.heads[0].parameters())
                                                  if name.startswith('latent') else 0)
            parameters['total_active_inference'] = (parameters['future_model'] + parameters['active_encoder_head']
                                                    + parameters['frozen_decoder'])
            report[name][str(k)] = dict(n=len(ids), errors=distributions, timing=timing, parameters=parameters)
            for metric, stats in distributions.items():
                rows.append(dict(model=name, k=k, n=len(ids), metric=metric, **stats))
            prediction_arrays[f'{name}_k{k}_indices'] = ids
            for metric, values in collected.items():
                prediction_arrays[f'{name}_k{k}_{metric}'] = np.concatenate(values)
        prediction_arrays[name] = all_predictions
    save_json(output_dir / 'metrics.json', report)
    write_csv(output_dir / 'metrics.csv', rows)
    np.savez_compressed(output_dir / 'predictions.npz', **prediction_arrays)
    np.savez_compressed(output_dir / 'validation_tuples.npz', **table)
    save_json(output_dir / 'evaluation_config.json', dict(config=config, device=device,
              dataset_manifest=manifest,
              timing_note='Synchronized inference; excludes host data transfer and NumPy normalization. '
                          'Full includes encoder(s), encoder(g), predictor, decoder, inverse normalization. '
                          'Per-sample time is amortized batched time, not single-request latency.',
              tail_definition='p90/p95/p99 and mean of errors >= p95',
              full_prediction_decoder_error='Latent model normalized_state/physical_state/maze_xy errors',
              decoder_only='D(E(target)); same real targets as full prediction'))
    return report
