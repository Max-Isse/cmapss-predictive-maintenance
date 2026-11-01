"""Monitoring baseline, captured when a model is trained and stored with its MLflow run.

Contents (D-028): reference deciles and proportions of every input feature (training rows) and of the model's
outputs (out-of-fold predictions), and the out-of-fold performance per true-RUL band. Out-of-fold, because the
replayed engines that monitoring watches are the official test engines: a test-set baseline would be circular.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from mlflow.tracking import MlflowClient

from cmapss_pm.config import Config
from cmapss_pm.models.data import ModelData
from cmapss_pm.models.metrics import interval_metrics, regression_metrics, rul_band
from cmapss_pm.models.train import TrainResult
from cmapss_pm.monitoring.drift import bin_edges, proportions

ARTIFACT = "monitoring/monitoring_baseline.json"
OUTPUTS = {"rul_pred": "reg_lightgbm", "p_fail_within_n": "clf_lightgbm_calibrated"}


def band_performance(df: Any, pred: str, lower: str, upper: str, edges: list[int]) -> dict[str, dict[str, float]]:
    out = {}
    bands = rul_band(df["rul"], edges)
    for band in sorted(bands.unique()):
        g = df[bands == band]
        out[band] = {
            "rows": len(g),
            **regression_metrics(g["rul_capped"], g[pred]),
            **interval_metrics(g["rul_capped"], g[lower], g[upper]),
        }
    return out


def build_baseline(res: TrainResult, data: ModelData, cfg: Config) -> dict[str, Any]:
    bins = int(cfg.section("monitoring")["psi_bins"])
    edges_rul = [int(e) for e in cfg.section("models")["rul_bands"]]
    features = {}
    for f in data.features:
        e = bin_edges(data.train[f].to_numpy(), bins)
        features[f] = {"edges": e, "proportions": proportions(data.train[f].to_numpy(), e)}
    outputs = {}
    for name, col in OUTPUTS.items():
        e = bin_edges(res.oof[col].to_numpy(), bins)
        outputs[name] = {"edges": e, "proportions": proportions(res.oof[col].to_numpy(), e)}
    return {
        "data": dict(data.meta),
        "psi_bins": bins,
        "features": features,
        "outputs": outputs,
        "performance_by_band": band_performance(res.oof, "reg_lightgbm", "rul_lower", "rul_upper", edges_rul),
        "interval_nominal_coverage": 1 - res.bundle.alpha,
        "fail_rate_oof": float(np.mean(res.oof["fail_within_n"])),
    }


def save_baseline(baseline: dict[str, Any], cfg: Config, client: MlflowClient, run_id: str) -> Path:
    path = cfg.path("artifacts_dir") / "monitoring_baseline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline), encoding="utf-8")
    client.log_artifact(run_id, str(path), "monitoring")
    return path


def load_baseline(client: MlflowClient, run_id: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        local = client.download_artifacts(run_id, ARTIFACT, tmp)
        return dict(json.loads(Path(local).read_text(encoding="utf-8")))
