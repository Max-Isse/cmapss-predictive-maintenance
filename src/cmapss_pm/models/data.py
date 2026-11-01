"""Training and holdout frames from gold, read at a pinned Delta version so every model run is reproducible.

Training rows: every cycle of the training engines. Holdout: every cycle of the official NASA test engines (their
RUL is known from the RUL files); the official C-MAPSS score uses each test engine's last cycle. Replayed rows are
copies of test rows and are not used here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from delta.tables import DeltaTable
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from cmapss_pm.config import Config
from cmapss_pm.features.build import LABELS, feature_columns
from cmapss_pm.features.gold import gold_path
from cmapss_pm.features.spec import load_spec, spec_path

ID = ["dataset", "split", "unit", "cycle"]


@dataclass
class ModelData:
    train: pd.DataFrame
    test: pd.DataFrame
    features: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def engine_id(df: pd.DataFrame) -> pd.Series:
        return df["dataset"] + "/" + df["unit"].astype(str)


def load_model_data(spark: SparkSession, cfg: Config) -> ModelData:
    spec = load_spec(spec_path(cfg))
    features = feature_columns(spec)
    path = gold_path(cfg, "features")
    version = int(DeltaTable.forPath(spark, path).history(1).first()["version"])  # type: ignore[index]
    gold = spark.read.format("delta").option("versionAsOf", version).load(path)
    versions = [r[0] for r in gold.select("_spec_version").distinct().collect()]
    if versions != [spec.version]:
        raise SystemExit(f"gold was built with spec {versions}, artifacts hold {spec.version}: rerun `cmapss gold`")
    cols = [*ID, *[f for f in features if f not in ID], *LABELS]  # `cycle` is both a key and a feature
    frames = {}
    for split in ("train", "test"):
        pdf = gold.where(F.col("split") == split).select(*cols).orderBy(*ID).toPandas()
        pdf[features] = pdf[features].astype("float32")
        frames[split] = pdf
    meta = {
        "gold_delta_version": version,
        "feature_spec_version": spec.version,
        "rul_cap": spec.rul_cap,
        "horizon": spec.horizon,
        "train_rows": len(frames["train"]),
        "train_engines": int(ModelData.engine_id(frames["train"]).nunique()),
        "test_rows": len(frames["test"]),
        "test_engines": int(ModelData.engine_id(frames["test"]).nunique()),
    }
    return ModelData(frames["train"], frames["test"], features, meta)
