"""Population stability index (PSI) on reference deciles."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

EPS = 1e-4  # floor for empty bins, so an empty bin contributes a finite (large) term


def bin_edges(reference: ArrayLike, bins: int = 10) -> list[float]:
    """Inner edges at the reference quantiles (duplicates dropped: discrete features get fewer bins)."""
    values = np.asarray(reference, float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return []
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1], method="lower"))  # observed values
    return [float(e) for e in edges]


def proportions(values: ArrayLike, edges: list[float]) -> list[float]:
    """Share of non-missing values in each bin (len(edges) + 1 bins), plus the missing share as the last entry."""
    v = np.asarray(values, float)
    missing = np.isnan(v)
    counts = np.bincount(np.searchsorted(edges, v[~missing], side="right"), minlength=len(edges) + 1)
    total = len(v)
    if total == 0:
        return [0.0] * (len(edges) + 2)
    return [float(c) / total for c in counts] + [float(missing.sum()) / total]


def psi(expected: list[float], actual: list[float]) -> float:
    """sum((a - e) * ln(a / e)) over bins; < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 large shift (rule of thumb)."""
    e = np.maximum(np.asarray(expected, float), EPS)
    a = np.maximum(np.asarray(actual, float), EPS)
    return float(np.sum((a - e) * np.log(a / e)))
