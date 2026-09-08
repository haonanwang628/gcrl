"""PointMaze endpoint scatter. No endpoint-connecting lines imply a trajectory."""
from pathlib import Path
import numpy as np


def plot_endpoints(evaluation_dir, output_path, horizon=20, count=4, seed=0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ..future_models.training import MODEL_NAMES
    root = Path(evaluation_dir)
    with np.load(root / 'validation_tuples.npz', allow_pickle=False) as f:
        table = dict(f)
    with np.load(root / 'predictions.npz', allow_pickle=False) as f:
        predictions = {name: f[name] for name in MODEL_NAMES}
    if table['observations'].shape[-1] != 2:
        raise ValueError('This visualization is restricted to PointMaze state observations')
    eligible = np.flatnonzero(table['k'] == horizon)
    if not len(eligible):
        raise ValueError('No tuples for the requested horizon')
    selection = np.random.default_rng(seed).choice(eligible, min(count, len(eligible)), replace=False)
    fig, axes = plt.subplots(len(selection), 4, figsize=(16, 4 * len(selection)), squeeze=False)
    for row, index in enumerate(selection):
        # Identical limits across four controls for each tuple.
        points = np.stack([table['observations'][index], table['goals'][index], table['targets'][index],
                           *[predictions[name][index] for name in MODEL_NAMES]])
        lo, hi = points.min(0) - 1, points.max(0) + 1
        center, radius = (lo + hi) / 2, (hi - lo).max() / 2
        lo, hi = center - radius, center + radius
        for col, name in enumerate(MODEL_NAMES):
            ax = axes[row, col]
            markers = [
                (table['observations'][index], 'o', 'black', 'Current'),
                (table['goals'][index], '*', 'green', 'Final goal'),
                (table['targets'][index], 's', 'blue', 'True k-step state'),
                (predictions[name][index], 'x', 'red', 'Predicted k-step state')]
            for point, marker, color, label in markers:
                ax.scatter(*point[:2], marker=marker, color=color, s=75, label=label)
            ax.set(title=f'{name}\nk={horizon}, H={table["H"][index]}, tuple={index}', xlabel='X', ylabel='Y')
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_aspect('equal', adjustable='box')
            # Zoom only when a poor prediction would collapse real endpoints visually.
            real = points[:3]
            real_span = max(float(np.ptp(real, axis=0).max()), .5)
            if 2 * radius > 4 * real_span:
                zoom = ax.inset_axes([.06, .56, .36, .36])
                for point, marker, color, label in markers:
                    zoom.scatter(*point[:2], marker=marker, color=color, s=30)
                mid = (real.min(0) + real.max(0)) / 2
                zoom.set_xlim(mid[0] - real_span * .65, mid[0] + real_span * .65)
                zoom.set_ylim(mid[1] - real_span * .65, mid[1] + real_span * .65)
                zoom.set_aspect('equal')
                zoom.set_title('Real endpoints (zoom)', fontsize=6)
                zoom.tick_params(labelsize=5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9)
    fig.suptitle('Endpoint comparison only; markers and distances do not show executed paths or maze connectivity')
    fig.tight_layout(rect=(0, .06, 1, .94))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
