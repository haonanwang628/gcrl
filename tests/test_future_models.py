import inspect
import torch
import pytest
from any_step_mher_ogbench.future_models.any_step import AnyStepModel
from any_step_mher_ogbench.future_models.one_step import OneStepModel


def test_any_step_single_forward_explicit_horizon():
    torch.manual_seed(1)
    model = AnyStepModel(3, width=8, depth=1, embedding_dim=4)
    calls = []
    hook = model.delta.register_forward_hook(lambda *args: calls.append(1))
    current, goal = torch.zeros(4, 3), torch.ones(4, 3)
    output = model(current, goal, torch.tensor([1, 5, 10, 20]))
    assert calls == [1]
    assert not torch.allclose(output[0], output[3])
    output.square().sum().backward()
    assert torch.all(model.horizon.embedding.weight.grad.abs().sum(1) > 0)
    hook.remove()
    assert list(inspect.signature(model.forward).parameters) == ['current', 'goal', 'k']
    with pytest.raises(ValueError, match='Unconfigured'):
        model(current[:1], goal[:1], torch.tensor([2]))


def test_recursive_baseline_uses_own_predictions_and_fixed_goal():
    model = OneStepModel(1, horizons=[1, 5, 10, 20], width=4, depth=1)
    seen = []
    def deterministic_step(current, goal):
        seen.append((current.detach().clone(), goal.detach().clone()))
        return current * 2 + goal
    model.step = deterministic_step
    current, goal = torch.ones(2, 1), torch.full((2, 1), 3.)
    output = model(current, goal, torch.tensor([1, 5]))
    torch.testing.assert_close(output, torch.tensor([[5.], [125.]]))
    assert len(seen) == 5
    torch.testing.assert_close(seen[1][0][1], torch.tensor([5.]))
    assert all(torch.equal(g, goal) for _, g in seen)


@pytest.mark.parametrize('cls', [OneStepModel, AnyStepModel])
def test_endpoint_supervision_backward_and_shapes(cls):
    model = cls(3, width=8, depth=1)
    output = model(torch.randn(4, 3), torch.randn(4, 3), torch.tensor([1, 5, 10, 20]))
    assert output.shape == (4, 3)
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
