"""What the API reads and writes: the champion model (MLflow), gold features and the prediction log (Delta, no JVM).

- Model: `models:/<name>@champion` resolved through MLflow; the served threshold comes from artifacts/policy.json when
  it was chosen for that model version (D-025), otherwise the bundle's fixed-precision threshold.
- Features: the API scores gold rows built by `features/build.py`, the one feature implementation (D-026); it looks
  them up by engine with delta-rs, so the image needs neither Spark nor Java.
- Prediction log: one Delta row per prediction (D-027).
"""

from __future__ import annotations

import json
import logging
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from deltalake import DeltaTable, write_deltalake
from mlflow.tracking import MlflowClient

from cmapss_pm.config import Config
from cmapss_pm.models.pyfunc import MaintenanceBundle
from cmapss_pm.models.registry import resolve_tracking_uri

log = logging.getLogger("cmapss.api")
KEY = ["dataset", "split", "unit", "cycle"]


@dataclass
class LoadedModel:
    bundle: MaintenanceBundle
    name: str
    alias: str
    version: str
    run_id: str
    threshold: float
    threshold_source: str
    lead_time: int | None
    metrics: dict[str, float] = field(default_factory=dict)
    loaded_at: str = ""


def load_champion_model(cfg: Config) -> LoadedModel:
    mc = cfg.section("models")["mlflow"]
    mlflow.set_tracking_uri(resolve_tracking_uri(cfg))
    client = MlflowClient()
    name, alias = str(mc["registered_model"]), str(mc["champion_alias"])
    mv = client.get_model_version_by_alias(name, alias)
    # load into a private copy: MLflow writes a metadata file next to the model, and the artifact mount is read-only
    model = mlflow.pyfunc.load_model(f"models:/{name}/{mv.version}", dst_path=tempfile.mkdtemp(prefix="model-"))
    bundle: MaintenanceBundle = model.unwrap_python_model().bundle
    threshold, source, lead = bundle.threshold, "fixed precision (no cost policy for this version)", None
    policy_file = cfg.path("artifacts_dir") / "policy.json"
    if policy_file.exists():
        policy = json.loads(policy_file.read_text(encoding="utf-8"))
        if str(policy.get("champion_version")) == str(mv.version):
            threshold, source, lead = (
                float(policy["threshold"]),
                "cost policy (artifacts/policy.json)",
                int(policy["lead_time"]),
            )
    metrics = client.get_run(mv.run_id).data.metrics if mv.run_id else {}
    return LoadedModel(
        bundle=bundle,
        name=name,
        alias=alias,
        version=str(mv.version),
        run_id=str(mv.run_id),
        threshold=threshold,
        threshold_source=source,
        lead_time=lead,
        metrics={k: float(v) for k, v in metrics.items()},
        loaded_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


class GoldFeatures:
    """Latest (or a given) gold feature row of an engine, read with delta-rs; refreshed to the newest table version."""

    def __init__(self, path: Path, features: list[str]) -> None:
        self.path = path
        self.columns = [*KEY, *[f for f in features if f not in KEY]]
        self._table: DeltaTable | None = None
        self._lock = threading.Lock()

    def _current(self) -> DeltaTable:
        with self._lock:
            if self._table is None:
                self._table = DeltaTable(str(self.path))
            else:
                self._table.update_incremental()  # the stream keeps rewriting gold; read its latest version
            return self._table

    def version(self) -> int:
        return int(self._current().version())

    def lookup(self, dataset: str, split: str, unit: int, cycle: int | None = None) -> pd.DataFrame:
        flt = (pc.field("dataset") == dataset) & (pc.field("split") == split) & (pc.field("unit") == unit)
        if cycle is not None:
            flt = flt & (pc.field("cycle") == cycle)
        table = self._current().to_pyarrow_dataset().to_table(columns=self.columns, filter=flt)
        df = table.to_pandas()
        return df.sort_values("cycle").tail(1).reset_index(drop=True) if len(df) else df


LOG_SCHEMA = pa.schema(
    [
        ("request_id", pa.string()),
        ("logged_at", pa.timestamp("us", tz="UTC")),
        ("source", pa.string()),  # api-lookup | api-features | batch
        ("model_name", pa.string()),
        ("model_version", pa.string()),
        ("threshold", pa.float64()),
        ("dataset", pa.string()),
        ("split", pa.string()),
        ("unit", pa.int64()),
        ("cycle", pa.int64()),
        ("rul_pred", pa.float64()),
        ("rul_lower", pa.float64()),
        ("rul_upper", pa.float64()),
        ("p_fail_within_n", pa.float64()),
        ("book_shop_visit", pa.bool_()),
        ("latency_ms", pa.float64()),
        ("features_json", pa.string()),  # only for posted feature rows; looked-up rows are in gold by key
    ]
)


class PredictionLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def append(self, rows: list[dict[str, Any]]) -> None:
        if rows:
            self._write(pa.Table.from_pylist(rows, schema=LOG_SCHEMA))

    def append_frame(self, df: pd.DataFrame) -> None:
        if len(df):
            self._write(pa.Table.from_pandas(df[LOG_SCHEMA.names], schema=LOG_SCHEMA, preserve_index=False))

    def _write(self, table: pa.Table) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            write_deltalake(str(self.path), table, mode="append", partition_by=None)

    def read(self) -> pd.DataFrame:
        return DeltaTable(str(self.path)).to_pandas() if (self.path / "_delta_log").exists() else pd.DataFrame()


def prediction_log_path(cfg: Config) -> Path:
    """Written by the API only (in compose: a named volume owned by the API's non-root user)."""
    return cfg.path("lakehouse_dir") / "serving" / "prediction_log"


def batch_predictions_path(cfg: Config) -> Path:
    """Written by `cmapss score` (pipeline container); same schema as the prediction log."""
    return cfg.path("lakehouse_dir") / "serving" / "batch_predictions"


def gold_features_path(cfg: Config) -> Path:
    return cfg.path("lakehouse_dir") / "gold" / "features"
