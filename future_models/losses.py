def endpoint_loss(predicted, target):
    """Same per-coordinate endpoint MSE; target always comes from real data."""
    return (predicted - target.detach()).square().mean()
