"""TempDATA compute_value_loss and PRE-optimizer Polyak target update."""
import torch


def expectile_loss(advantage, difference, expectile):
    return torch.where(advantage >= 0, expectile, 1 - expectile) * difference.square()


def temporal_loss(encoder, target_encoder, batch, discount=0.99, expectile=0.95,
                  smoothing_coef=0.01):
    obs, nxt, goal = batch['observations'], batch['next_observations'], batch['goals']
    success = batch['success']
    reward, mask = success - 1, 1 - success
    with torch.no_grad():
        next_values = target_encoder.value(nxt, goal)
        target_values = target_encoder.value(obs, goal)
        q = reward + discount * mask * next_values.min(0).values
        advantage = q - target_values.mean(0)
        twin_targets = reward[None] + discount * mask[None] * next_values
        # Original phi(s) call omits differentiable network_params.
        current_phi = encoder(obs)
    values = encoder.value(obs, goal)
    td = expectile_loss(advantage[None], twin_targets - values, expectile).mean(1).sum()
    # Original has NO axis: preserve the whole-batch Frobenius norm.
    smooth = smoothing_coef * torch.relu(torch.linalg.vector_norm(encoder(nxt) - current_phi) - 1)
    loss = td + smooth
    return loss, dict(loss=loss, expectile_td=td, local_constraint=smooth,
                      value_mean=values.mean())


@torch.no_grad()
def update_target(target, online, tau):
    """Call AFTER loss/backward but BEFORE optimizer.step: use pre-step weights."""
    if not 0 <= tau <= 1:
        raise ValueError('tau must lie in [0,1]')
    for tp, p in zip(target.parameters(), online.parameters()):
        tp.lerp_(p, tau)
