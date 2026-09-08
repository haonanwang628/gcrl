import json
import numpy as np
import pytest
import torch
from any_step_mher_ogbench.common import ROOT, load_config, module_hash
from any_step_mher_ogbench.cli import prepare
from any_step_mher_ogbench.representations.training import train_representation, load_representation
from any_step_mher_ogbench.future_models.training import train_future_models, load_models, MODEL_NAMES
from any_step_mher_ogbench.evaluation.future_model_eval import evaluate, validate_table
from any_step_mher_ogbench.evaluation.reporting import summarize
from any_step_mher_ogbench.datasets.trajectories import Trajectories
from any_step_mher_ogbench.datasets.samplers import FutureSampler


def test_phase_1_to_4_pipeline(tmp_path):
    config = load_config(ROOT / 'configs/smoke.yaml')
    prepare(tmp_path / 'data', config, fixture=True)
    encoder, decoder, _ = train_representation(tmp_path / 'data', tmp_path / 'repr', config)
    hashes = module_hash(encoder), module_hash(decoder)
    train_future_models(tmp_path / 'data', tmp_path / 'repr', tmp_path / 'models', config)
    restored_encoder, restored_decoder, _ = load_representation(tmp_path / 'repr')
    assert hashes == (module_hash(restored_encoder), module_hash(restored_decoder))
    models, meta = load_models(tmp_path / 'models')
    train = Trajectories.load(tmp_path / 'data/train.npz')
    expected = FutureSampler(train, config['horizons'], config['seed'] + 21).sample(
        config['future_model']['batch_size'])
    audit = np.load(tmp_path / 'models/training_tuple_indices.npy', mmap_mode='r')
    np.testing.assert_array_equal(audit[0], np.column_stack([expected[k] for k in
        ('trajectory_id', 'current_index', 'target_index', 'goal_index', 'k', 'H')]))
    assert set(models) == set(MODEL_NAMES)
    for name in models:
        saved = torch.load(tmp_path / 'models' / f'{name}.pt', weights_only=True)
        assert saved['tuple_stream_sha256'] == meta['tuple_stream_sha256']
    report = evaluate(tmp_path / 'data', tmp_path / 'repr', tmp_path / 'models', tmp_path / 'eval', config)
    for name in MODEL_NAMES:
        assert set(report[name]) == {'1', '5', '10', '20'}
        for result in report[name].values():
            assert result['n'] == 8
            assert result['errors']['physical_state_l2']['p95'] >= 0
            assert ('latent_l2' in result['errors']) == name.startswith('latent')
            assert result['parameters']['future_model'] > 0
    # Both latent controls have exactly the same decoder floor on the same targets.
    for k in config['horizons']:
        assert (report['latent_one_step'][str(k)]['errors']['decoder_only_normalized_l2'] ==
                report['latent_any_step'][str(k)]['errors']['decoder_only_normalized_l2'])
    assert (tmp_path / 'eval/metrics.csv').exists()
    with np.load(tmp_path / 'data/validation_tuples.npz') as f:
        a = dict(f)
    with np.load(tmp_path / 'eval/validation_tuples.npz') as f:
        for key in a:
            np.testing.assert_array_equal(a[key], f[key])
    a['targets'] = a['targets'].copy()
    a['targets'][0] += 1
    with pytest.raises(ValueError, match='differ from real'):
        validate_table(a, Trajectories.load(tmp_path / 'data/validation.npz'), config['horizons'])
    with pytest.raises(FileExistsError):
        train_representation(tmp_path / 'data', tmp_path / 'repr', config)


def test_error_summary_matches_hand_calculation():
    result = summarize(np.arange(101))
    assert result['mean'] == 50 and result['median'] == 50
    assert result['p95'] == 95 and result['worst_5_percent_mean'] == 97.5
