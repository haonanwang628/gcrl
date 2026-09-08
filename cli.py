"""Commands implement phases 1-4 only. Training never creates an environment."""
from pathlib import Path
import argparse
import numpy as np
from .common import ROOT, load_config, save_json, versions
from .datasets.ogbench import load_official
from .datasets.trajectories import reconstruct, assert_split_isolation
from .datasets.samplers import FutureSampler


def prepare(output, config, fixture=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite prepared dataset: {output}')
    cfg = config['dataset']
    if fixture:
        # Explicitly marked numerical test fixture, never a benchmark result.
        # No environment is used to generate this diagnostic-only input.
        data = []
        for split, shift in [('train', 0.), ('validation', 100.)]:
            states = [np.column_stack([np.linspace(0, 3, 33) + i + shift,
                                        np.sin(np.linspace(0, 3, 33) + i)]) for i in range(4)]
            arrays = dict(observations=np.concatenate([s[:-1] for s in states]),
                          next_observations=np.concatenate([s[1:] for s in states]),
                          terminals=np.tile(np.r_[np.zeros(31), 1], 4))
            data.append(reconstruct(arrays, split, 'fixture-' + split))
        train, validation = data
        manifest = dict(fixture=True, benchmark_result=False, env_name='fixture-pointmaze-shape',
                        source='Deterministic numerical test fixture; not OGBench data')
    else:
        train, validation, manifest = load_official(cfg['env_name'], cfg['dataset_type'], cfg['dataset_dir'])
    assert_split_isolation(train, validation)
    # Validate support before writing partial artifacts.
    FutureSampler(train, config['horizons'], config['seed'])
    tuples = FutureSampler(validation, config['horizons'], config['seed'] + 1).validation_table(
        cfg['validation_samples_per_horizon'])
    output.mkdir(parents=True, exist_ok=True)
    train.save(output / 'train.npz')
    validation.save(output / 'validation.npz')
    np.savez_compressed(output / 'validation_tuples.npz', **tuples)
    manifest.update(config=config, versions=versions(),
                    train=dict(trajectories=len(train.lengths), states=len(train.states),
                               transitions=int(train.lengths.sum()), fingerprint=train.fingerprint),
                    validation=dict(trajectories=len(validation.lengths), states=len(validation.states),
                                    transitions=int(validation.lengths.sum()), fingerprint=validation.fingerprint),
                    observation_dim=train.states.shape[1])
    save_json(output / 'manifest.json', manifest)
    print({k: manifest[k] for k in ('train', 'validation', 'observation_dim')}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'train-representation', 'train-future', 'evaluate', 'smoke'):
        p = sub.add_parser(command)
        p.add_argument('--config')
        p.add_argument('--output', required=True)
        p.add_argument('--device', default='cpu')
        if command in ('train-representation', 'train-future', 'evaluate'):
            p.add_argument('--data', required=True)
        if command in ('train-future', 'evaluate'):
            p.add_argument('--representation', required=True)
        if command == 'evaluate':
            p.add_argument('--models', required=True)
        if command in ('prepare', 'smoke'):
            p.add_argument('--dataset-dir')
            p.add_argument('--env')
        if command == 'smoke':
            p.add_argument('--fixture', action='store_true', help='Numerical test only; not real OGBench verification')
    p = sub.add_parser('plot')
    p.add_argument('--evaluation', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--horizon', type=int, default=20)
    p.add_argument('--count', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'plot':
        from .evaluation.visualize_pointmaze import plot_endpoints
        plot_endpoints(args.evaluation, args.output, args.horizon, args.count)
        return
    config = load_config(args.config or (ROOT / 'configs/smoke.yaml' if args.command == 'smoke' else None))
    if getattr(args, 'dataset_dir', None):
        config['dataset']['dataset_dir'] = args.dataset_dir
    if getattr(args, 'env', None):
        config['dataset']['env_name'] = args.env
    from .representations.training import train_representation
    from .future_models.training import train_future_models
    from .evaluation.future_model_eval import evaluate
    if args.command == 'prepare':
        prepare(args.output, config)
    elif args.command == 'train-representation':
        train_representation(args.data, args.output, config, args.device)
    elif args.command == 'train-future':
        train_future_models(args.data, args.representation, args.output, config, args.device)
    elif args.command == 'evaluate':
        evaluate(args.data, args.representation, args.models, args.output, config, args.device)
    elif args.command == 'smoke':
        root = Path(args.output)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f'Refusing to overwrite smoke artifacts: {root}')
        prepare(root / 'data', config, fixture=args.fixture)
        train_representation(root / 'data', root / 'representation', config, args.device)
        train_future_models(root / 'data', root / 'representation', root / 'models', config, args.device)
        evaluate(root / 'data', root / 'representation', root / 'models', root / 'evaluation', config, args.device)
        if config['dataset']['env_name'].startswith('pointmaze'):
            from .evaluation.visualize_pointmaze import plot_endpoints
            plot_endpoints(root / 'evaluation', root / 'evaluation/endpoints.png', count=2)
        save_json(root / 'smoke_result.json', dict(passed=True, fixture=args.fixture,
                  device=args.device, versions=versions(), phases=[1, 2, 3, 4],
                  note='Execution check only; three optimization steps do not establish model quality.'))


if __name__ == '__main__':
    main()
