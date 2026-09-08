import numpy as np
import pytest
from any_step_mher_ogbench.datasets.trajectories import (
    reconstruct, split_trajectories, assert_split_isolation, Trajectories)
from any_step_mher_ogbench.datasets.samplers import FutureSampler, TemporalSampler
from any_step_mher_ogbench.datasets.normalization import StateNormalizer
from any_step_mher_ogbench.datasets.ogbench import dataset_name


def test_final_next_and_provenance(trajectories, tmp_path):
    data = trajectories()
    np.testing.assert_array_equal(data.lengths, [25, 31, 3])
    final = data.offsets[1:] - 1
    np.testing.assert_array_equal(data.states[final, 0], [25, 131, 203])
    assert data.is_next[final].all()
    np.testing.assert_array_equal(data.source_rows[final], [24, 55, 58])
    data.save(tmp_path / 'data.npz')
    restored = Trajectories.load(tmp_path / 'data.npz')
    assert restored.fingerprint == data.fingerprint
    assert not restored.states.flags.writeable


@pytest.mark.parametrize('boundary_key', ['terminals', 'terminated', 'truncated', 'timeouts', 'truncations'])
def test_every_boundary_type(boundary_key):
    data = reconstruct(dict(observations=np.array([[0.], [1.], [100.], [101.]]),
                            next_observations=np.array([[1.], [2.], [101.], [102.]]),
                            **{boundary_key: [0, 1, 0, 1]}), 'train', 'source')
    np.testing.assert_array_equal(data.offsets, [0, 3, 6])
    batch = FutureSampler(data, [1]).sample(100)
    assert np.all(batch['H'] == 2)
    assert np.all(batch['goals'][:, 0] - batch['observations'][:, 0] == 2)


def test_union_of_terminal_and_timeout():
    data = reconstruct(dict(observations=np.arange(6.)[:, None], next_observations=np.arange(1., 7.)[:, None],
                            terminals=[0, 1, 0, 0, 0, 0], timeouts=[0, 0, 0, 1, 0, 0]), 'train', 'source')
    np.testing.assert_array_equal(data.lengths, [2, 2, 2])


def test_discontinuity_and_compact_rejected():
    with pytest.raises(ValueError, match='Explicit'):
        reconstruct(dict(observations=np.zeros((2, 2)), terminals=[0, 1]), 'train', 'source')
    with pytest.raises(ValueError, match='discontinuity'):
        reconstruct(dict(observations=np.array([[0.], [100.]]), next_observations=np.array([[1.], [101.]]),
                         terminals=[0, 1]), 'train', 'source')


def test_horizon_ordering_indices_and_boundaries(trajectories):
    data = trajectories()
    batch = FutureSampler(data, seed=23).sample(5000)
    assert set(batch['k']) == {1, 5, 10, 20}
    assert np.all((1 <= batch['k']) & (batch['k'] < batch['H']))
    for i in range(len(batch['k'])):
        trajectory = batch['trajectory_id'][i]
        current, target, goal = [batch[key][i] for key in ('current_index', 'target_index', 'goal_index')]
        assert data.offsets[trajectory] <= current < target < goal < data.offsets[trajectory + 1]
        assert target - current == batch['k'][i]
        assert goal - current == batch['H'][i]
        np.testing.assert_array_equal(batch['targets'][i], data.states[target])
        np.testing.assert_array_equal(batch['goals'][i], data.states[goal])
    assert batch['goal_is_next'].any(), 'Final valid next state must be reachable as a goal'
    assert 'actions' not in batch and 'action_sequence' not in batch


def test_split_isolation_and_normalization(trajectories):
    train, val = split_trajectories(trajectories(lengths=(25, 25, 25, 25)), .5, 42)
    assert_split_isolation(train, val)
    assert not set(train.trajectory_ids) & set(val.trajectory_ids)
    assert not set(zip(train.source_rows, train.is_next)) & set(zip(val.source_rows, val.is_next))
    for data in (train, val):
        batch = FutureSampler(data).sample(200)
        assert set(batch['split']) == {data.split}
        assert set(batch['trajectory_id']) <= set(data.trajectory_ids)
    normalizer = StateNormalizer.fit(train)
    np.testing.assert_allclose(normalizer.unnormalize(normalizer.normalize(val.states)), val.states, atol=1e-5)
    with pytest.raises(ValueError, match='training split'):
        StateNormalizer.fit(val)
    wrong_val = train.subset([0], 'validation')
    with pytest.raises(ValueError, match='both splits'):
        assert_split_isolation(train, wrong_val)


def test_short_trajectory_strictness(trajectories):
    with pytest.raises(ValueError, match='k=20'):
        FutureSampler(trajectories(lengths=(20,)), [20])
    batch = FutureSampler(trajectories(lengths=(21,)), [20]).sample(10)
    assert np.all(batch['t'] == 0) and np.all(batch['H'] == 21)


def test_temporal_sampler_preserves_real_next_and_split(trajectories):
    data = trajectories('validation', shift=1000)
    batch = TemporalSampler(data, p_current=1., p_future=0).sample(1000)
    assert batch['success'].all()
    np.testing.assert_array_equal(batch['observations'], batch['goals'])
    np.testing.assert_allclose(batch['next_observations'] - batch['observations'],
                               np.tile([1., .5], (1000, 1)))
    assert np.all(batch['goals'][:, 0] >= 1000)


@pytest.mark.parametrize('env', ['pointmaze-medium-v0', 'pointmaze-large-v0', 'antmaze-medium-v0', 'antmaze-large-v0'])
def test_official_name_mapping(env):
    assert dataset_name(env) == env.replace('-v0', '-navigate-v0')


def test_official_loader_regular_final_state(tmp_path):
    from ogbench.utils import load_dataset
    raw_states = np.array([[0., 0.], [1., 0.], [2., 0.], [10., 0.], [11., 0.], [12., 0.]], np.float32)
    path = tmp_path / 'official-format.npz'
    np.savez(path, observations=raw_states, actions=np.zeros((6, 2)), terminals=[0, 0, 1, 0, 0, 1])
    regular = load_dataset(str(path), compact_dataset=False)
    data = reconstruct(regular, 'train', 'official-fixture')
    np.testing.assert_array_equal(data.states, raw_states)
    np.testing.assert_array_equal(data.lengths, [2, 2])


@pytest.mark.parametrize('env,dim', [('pointmaze-medium-v0', 2), ('pointmaze-large-v0', 2),
                                    ('antmaze-medium-v0', 29), ('antmaze-large-v0', 29)])
def test_official_adapter_never_creates_env(tmp_path, monkeypatch, env, dim):
    import gymnasium
    from any_step_mher_ogbench.datasets.ogbench import load_official
    def forbidden(*args, **kwargs):
        raise AssertionError('Environment creation is forbidden in phases 1-4')
    monkeypatch.setattr(gymnasium, 'make', forbidden)
    for suffix, offset in [('', 0.), ('-val', 10.)]:
        np.savez(tmp_path / f'{dataset_name(env)}{suffix}.npz',
                 observations=np.arange(3 * dim, dtype=np.float32).reshape(3, dim) + offset,
                 actions=np.zeros((3, 2)), terminals=[0, 0, 1])
    train, val, manifest = load_official(env, dataset_dir=str(tmp_path))
    assert_split_isolation(train, val)
    assert train.is_next[-1] and val.is_next[-1]
    assert manifest['dataset_name'] == dataset_name(env)
    assert train.states.shape == (3, dim)
