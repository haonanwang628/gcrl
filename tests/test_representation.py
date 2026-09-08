from copy import deepcopy
import numpy as np
import torch
from any_step_mher_ogbench.common import freeze, module_hash
from any_step_mher_ogbench.representations.encoder import TemporalEncoder
from any_step_mher_ogbench.representations.decoder import Decoder
from any_step_mher_ogbench.representations.losses import temporal_loss, update_target


def test_original_loss_equations_and_stopped_branch():
    torch.manual_seed(3)
    encoder = TemporalEncoder(2, 3, [8])
    target = freeze(deepcopy(encoder))
    obs, nxt, goal = [torch.randn(5, 2) for _ in range(3)]
    success = torch.tensor([0., 1., 0., 0., 1.])
    batch = dict(observations=obs, next_observations=nxt, goals=goal, success=success)
    loss, metrics = temporal_loss(encoder, target, batch, .99, .95, .01)
    with torch.no_grad():
        nv = target.value(nxt, goal).numpy()
        tv = target.value(obs, goal).numpy()
        v = encoder.value(obs, goal).numpy()
        r, m = success.numpy() - 1, 1 - success.numpy()
        adv = r + .99 * m * nv.min(0) - tv.mean(0)
        differences = r[None] + .99 * m[None] * nv - v
        expected_td = (np.where(adv >= 0, .95, .05)[None] * differences ** 2).mean(1).sum()
        expected_smooth = .01 * max(np.linalg.norm((encoder(nxt) - encoder(obs)).numpy()) - 1, 0)
    np.testing.assert_allclose(float(loss.detach()), expected_td + expected_smooth, rtol=2e-6)
    loss.backward()
    assert any(p.grad is not None for p in encoder.parameters())
    assert all(p.grad is None for p in target.parameters())
    # The smooth term propagates through next_phi only, exactly as in TempDATA.
    encoder.zero_grad(set_to_none=True)
    _, metrics = temporal_loss(encoder, target, batch, smoothing_coef=1.)
    actual = torch.autograd.grad(metrics['local_constraint'], tuple(encoder.parameters()), allow_unused=True)
    expected = torch.relu(torch.linalg.vector_norm(encoder(nxt) - encoder(obs).detach()) - 1)
    wanted = torch.autograd.grad(expected, tuple(encoder.parameters()), allow_unused=True)
    for a, b in zip(actual, wanted):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b)


def test_polyak_uses_pre_optimizer_weights():
    online, target = torch.nn.Linear(1, 1, bias=False), torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        online.weight.fill_(2); target.weight.fill_(0)
    opt = torch.optim.SGD(online.parameters(), lr=.1)
    online(torch.ones(1, 1)).sum().backward()
    update_target(target, online, .25)
    opt.step()
    torch.testing.assert_close(target.weight, torch.tensor([[.5]]))
    torch.testing.assert_close(online.weight, torch.tensor([[1.9]]))


def test_decoder_cannot_change_frozen_encoder():
    encoder, decoder = freeze(TemporalEncoder(2, 3, [8])), Decoder(2, 3, [8])
    before = module_hash(encoder)
    optim = torch.optim.Adam(decoder.parameters())
    states = torch.randn(8, 2)
    loss = (decoder(encoder(states)) - states).square().mean()
    loss.backward(); optim.step()
    assert before == module_hash(encoder)
    assert all(p.grad is None for p in encoder.parameters())
