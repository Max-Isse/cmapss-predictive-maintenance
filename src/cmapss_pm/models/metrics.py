"""Evaluation metrics: regression (RMSE, MAE, NASA score), classification, calibration, interval coverage."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score


def rmse(y: ArrayLike, pred: ArrayLike) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred, float) - np.asarray(y, float)) ** 2)))


def mae(y: ArrayLike, pred: ArrayLike) -> float:
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(y, float))))


def nasa_score(y: ArrayLike, pred: ArrayLike) -> float:
    """PHM08 scoring function, summed over engines: late predictions (pred > true) cost more than early ones.

    d = pred - true; s = exp(-d/13) - 1 when d < 0, exp(d/10) - 1 otherwise.
    """
    d = np.asarray(pred, float) - np.asarray(y, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def regression_metrics(y: ArrayLike, pred: ArrayLike) -> dict[str, float]:
    return {"rmse": rmse(y, pred), "mae": mae(y, pred), "bias": float(np.mean(np.asarray(pred) - np.asarray(y)))}


def threshold_at_precision(y: ArrayLike, p: ArrayLike, target: float) -> float:
    """Lowest threshold whose precision reaches `target` (maximises recall subject to the precision floor)."""
    precision, _, thresholds = precision_recall_curve(y, p)
    ok = np.where(precision[:-1] >= target)[0]
    return float(thresholds[ok[0]]) if len(ok) else 1.0


def classification_metrics(y: ArrayLike, p: ArrayLike, threshold: float) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    alert = p >= threshold
    tp = int(np.sum(alert & (y == 1)))
    out = {
        "pr_auc": float(average_precision_score(y, p)) if y.any() else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "prevalence": float(y.mean()),
        "threshold": float(threshold),
        "precision_at_threshold": tp / int(alert.sum()) if alert.any() else float("nan"),
        "recall_at_threshold": tp / int(y.sum()) if y.any() else float("nan"),
    }
    return out


def calibration_table(y: ArrayLike, p: ArrayLike, bins: int = 10) -> pd.DataFrame:
    """Mean predicted probability vs observed rate in equal-width probability bins."""
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    df["bin"] = np.minimum((df.p * bins).astype(int), bins - 1)
    t = df.groupby("bin").agg(rows=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
    return t.reset_index()


def expected_calibration_error(y: ArrayLike, p: ArrayLike, bins: int = 10) -> float:
    t = calibration_table(y, p, bins)
    return float(np.sum(t.rows * np.abs(t.mean_predicted - t.observed_rate)) / t.rows.sum())


def interval_metrics(y: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> dict[str, float]:
    y = np.asarray(y, float)
    inside = (y >= lower) & (y <= upper)
    return {"coverage": float(inside.mean()), "mean_width": float(np.mean(np.asarray(upper) - np.asarray(lower)))}


def rul_band(rul: pd.Series, edges: list[int]) -> pd.Series:
    """Label true RUL with its band, e.g. edges [0, 30, 60, 125, 1000] -> '0-30', '31-60', '61-125', '>125'."""
    labels = [f"{lo if i == 0 else lo + 1}-{hi}" for i, (lo, hi) in enumerate(itertools.pairwise(edges[:-1]))]
    labels.append(f">{edges[-2]}")
    return pd.cut(rul, bins=[-1, *edges[1:]], labels=labels).astype(str)
