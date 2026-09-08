"""Per-sample endpoint errors, distributions, and machine-readable tables."""
import csv
import numpy as np


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Cannot summarize empty or non-finite errors')
    threshold = np.quantile(values, .95)
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p90=float(np.quantile(values, .90)), p95=float(threshold),
                p99=float(np.quantile(values, .99)),
                worst_5_percent_mean=float(values[values >= threshold].mean()))


def endpoint_errors(predicted, target, prefix):
    difference = np.asarray(predicted) - np.asarray(target)
    return {prefix + '_l2': np.linalg.norm(difference, axis=-1),
            prefix + '_mse': np.mean(difference ** 2, axis=-1)}


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
