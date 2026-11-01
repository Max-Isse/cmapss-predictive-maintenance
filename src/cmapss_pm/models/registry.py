"""MLflow tracking and registry: log a training run, register the bundle, move the "champion" alias on improvement.

Every run records the data it was trained on: the gold Delta version, the feature spec version and the config
(D-023). The registered model holds the full pyfunc bundle, so serving loads `models:/<name>@champion`.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from cmapss_pm.config import Config
from cmapss_pm.models.pyfunc import MaintenanceBundle, MaintenancePyfunc

log = logging.getLogger(__name__)


def setup_mlflow(cfg: Config) -> tuple[MlflowClient, str]:
    mc = cfg.section("models")["mlflow"]
    uri = str(mc["tracking_uri"])
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):  # relative to the project root
        db = cfg.root / uri.removeprefix("sqlite:///")
        db.parent.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{db.as_posix()}"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    name = str(mc["experiment"])
    exp = client.get_experiment_by_name(name)
    if exp is None:
        artifacts = (cfg.root / mc["artifact_location"]).resolve()
        artifacts.mkdir(parents=True, exist_ok=True)
        exp_id = client.create_experiment(name, artifact_location=artifacts.as_uri())
    else:
        exp_id = exp.experiment_id
    return client, exp_id


def champion_metric(client: MlflowClient, name: str, alias: str, metric: str) -> float | None:
    try:
        version = client.get_model_version_by_alias(name, alias)
    except mlflow.exceptions.MlflowException:
        return None
    if version.run_id is None:
        return None
    value = client.get_run(version.run_id).data.metrics.get(metric)
    return None if value is None else float(value)


def promote_if_better(
    client: MlflowClient, name: str, version: str, alias: str, metric: str, new_value: float
) -> tuple[bool, float | None]:
    """Point `alias` at `version` when there is no champion or the new value is lower (better)."""
    current = champion_metric(client, name, alias, metric)
    if current is None or new_value < current:
        client.set_registered_model_alias(name, alias, version)
        return True, current
    return False, current


def log_and_register(
    cfg: Config,
    bundle: MaintenanceBundle,
    metrics: dict[str, float],
    example: pd.DataFrame,
    artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    mc = cfg.section("models")["mlflow"]
    client, exp_id = setup_mlflow(cfg)
    name, alias, primary = str(mc["registered_model"]), str(mc["champion_alias"]), str(mc["primary_metric"])
    with mlflow.start_run(experiment_id=exp_id) as run:
        mlflow.set_tags(
            {
                "gold_delta_version": str(bundle.meta["gold_delta_version"]),
                "feature_spec_version": str(bundle.meta["feature_spec_version"]),
                "data": "NASA C-MAPSS FD001-FD004 (simulation); training engines only",
            }
        )
        mlflow.log_params(
            {
                **{f"lgbm_{k}": v for k, v in cfg.section("models")["lightgbm"].items()},
                "cv_folds": cfg.section("models")["cv_folds"],
                "rul_cap": bundle.rul_cap,
                "horizon": bundle.horizon,
                "interval_alpha": bundle.alpha,
                "conformal_q_hat": json.dumps([round(q, 3) for q in bundle.q_hat]),
                "interval_group_bounds": json.dumps(bundle.group_bounds),
                "alert_threshold": round(bundle.threshold, 4),
                "n_features": len(bundle.features),
                **{k: v for k, v in bundle.meta.items() if k.endswith(("rows", "engines"))},
            }
        )
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})  # NaN-free
        mlflow.log_dict(cfg.raw, "config.json")
        for local, dest in (artifacts or {}).items():
            mlflow.log_artifact(local, dest)
        x = example[bundle.features].astype("float32")
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=MaintenancePyfunc(bundle),
            signature=infer_signature(x, bundle.predict_frame(x)),
            input_example=x.head(3),
            registered_model_name=name,
        )
    version = str(info.registered_model_version)
    promoted, previous = promote_if_better(client, name, version, alias, primary, float(metrics[primary]))
    result = {
        "run_id": run.info.run_id,
        "model": name,
        "version": version,
        "alias": alias,
        "promoted": promoted,
        "primary_metric": primary,
        "value": float(metrics[primary]),
        "previous_champion_value": previous,
    }
    log.info("mlflow: %s", json.dumps(result))
    return result
