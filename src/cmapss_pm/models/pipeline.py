"""One training run end to end: data -> CV and final models -> evaluation -> MLflow (+ monitoring baseline) -> report.

Used by `cmapss train` and by the retraining path of `cmapss monitor` (a retrain is just another run; the champion
alias moves only if it improves the primary CV metric, D-023).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession

from cmapss_pm.config import Config
from cmapss_pm.models.data import load_model_data
from cmapss_pm.models.registry import log_and_register
from cmapss_pm.models.report import evaluate, write_model_report
from cmapss_pm.models.train import train_all
from cmapss_pm.monitoring.baseline import build_baseline, save_baseline

log = logging.getLogger(__name__)


def train_and_register(spark: SparkSession, cfg: Config, write_report: bool = True) -> dict[str, Any]:
    data = load_model_data(spark, cfg)
    log.info("model data: %s", json.dumps(data.meta))
    res = train_all(data, cfg)
    ev = evaluate(res, cfg)
    log.info("metrics: %s", json.dumps({k: round(v, 4) for k, v in ev["flat"].items()}))
    info = log_and_register(cfg, res.bundle, ev["flat"], data.test)
    client = MlflowClient()
    save_baseline(build_baseline(res, data, cfg), cfg, client, info["run_id"])
    if write_report:
        path = write_model_report(res, ev, cfg, info)
        client.log_artifact(info["run_id"], str(path), "reports")
        log.info("report %s", path)
    return {**info, "metrics": ev["flat"]}
