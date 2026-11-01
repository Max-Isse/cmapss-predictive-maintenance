"""The feature spec: every constant the feature code needs, fitted once on TRAINING engines and saved as JSON.

Training (gold tables), the replay and serving all load the same spec, so a test or live engine is never used to
fit anything (D-013). Contents:

- operating conditions: k-means centroids of the 3 settings (standardised for the distance), ordered by setting;
- near-constant sensors to drop: <= `constant_max_distinct` distinct values within every condition;
- per-condition mean and standard deviation of each kept sensor (the normalisation);
- rolling windows, the RUL cap and the failure horizon N.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from cmapss_pm.config import Config


@dataclass(frozen=True)
class FeatureSpec:
    settings: list[str]
    settings_mean: list[float]
    settings_std: list[float]
    centroids: list[list[float]]  # one per condition, original setting units; condition id = list index
    kept_sensors: list[str]
    dropped_sensors: dict[str, str]  # sensor -> reason
    norm_mean: dict[str, list[float]]  # sensor -> mean per condition
    norm_std: dict[str, list[float]]  # sensor -> standard deviation per condition (1.0 where constant)
    windows: list[int]
    rul_cap: int
    horizon: int
    fitted_on: dict[str, Any] = field(default_factory=dict)

    @property
    def version(self) -> str:
        """Content hash (ignores `fitted_on`): identical constants give the identical version."""
        d = asdict(self)
        d.pop("fitted_on")
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def to_json(self) -> str:
        return json.dumps({**asdict(self), "version": self.version}, indent=2)

    @classmethod
    def from_json(cls, text: str) -> FeatureSpec:
        d = json.loads(text)
        d.pop("version", None)
        return cls(**d)


def save_spec(spec: FeatureSpec, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.to_json(), encoding="utf-8")
    return path


def load_spec(path: Path) -> FeatureSpec:
    return FeatureSpec.from_json(path.read_text(encoding="utf-8"))


def spec_path(cfg: Config) -> Path:
    return cfg.path("artifacts_dir") / "feature_spec.json"


def fit_conditions(train: DataFrame, cfg: Config) -> tuple[list[float], list[float], list[list[float]]]:
    """K-means on the standardised settings; returns (mean, std, centroids in original units, sorted)."""
    settings = list(cfg.raw["schema"]["settings"])
    k = int(cfg.section("features")["n_conditions"])
    stats = train.agg(*[F.avg(c) for c in settings], *[F.stddev_pop(c) for c in settings]).first()
    assert stats is not None
    mean = [float(stats[i]) for i in range(len(settings))]
    std = [float(stats[len(settings) + i]) or 1.0 for i in range(len(settings))]
    z = train.select(*[((F.col(c) - m) / s).alias(c) for c, m, s in zip(settings, mean, std, strict=True)])
    vec = VectorAssembler(inputCols=settings, outputCol="x").transform(z).select("x")
    if k == 1:  # a single condition (e.g. FD001 alone): its centre is the mean; Spark's KMeans needs k >= 2
        centres: list[Any] = [[0.0] * len(settings)]
    else:
        vec = vec.cache()
        # k-means can stop in a local optimum (two conditions merged, one split): keep the best of several seeds
        restarts = int(cfg.section("features").get("kmeans_restarts", 1))
        models = [KMeans(k=k, seed=cfg.seed + i, featuresCol="x", maxIter=50).fit(vec) for i in range(restarts)]
        centres = min(models, key=lambda m: m.summary.trainingCost).clusterCenters()
        vec.unpersist()
    centroids = {
        tuple(round(float(v) * s + m, 6) for v, m, s in zip(c, mean, std, strict=True)) for c in centres
    }  # k-means|| may return fewer than k distinct centres when the data has fewer distinct points
    return mean, std, [list(c) for c in sorted(centroids)]


def fit_spec(train: DataFrame, cfg: Config) -> FeatureSpec:
    """Fit every constant on the training rows of silver.cycles (or any frame with the same columns)."""
    from cmapss_pm.features.build import assign_condition  # build imports this module

    fc = cfg.section("features")
    settings = list(cfg.raw["schema"]["settings"])
    mean, std, centroids = fit_conditions(train, cfg)
    probe = FeatureSpec(settings, mean, std, centroids, [], {}, {}, {}, [], 0, 0)
    labelled = assign_condition(train, probe).cache()
    sensors = cfg.sensor_columns
    per_cond = (
        labelled.groupBy("op_condition")
        .agg(
            *[F.countDistinct(s).alias(f"{s}__n") for s in sensors],
            *[F.avg(s).alias(f"{s}__mean") for s in sensors],
            *[F.stddev_samp(s).alias(f"{s}__std") for s in sensors],
            F.max("op_condition_dist").alias("max_dist"),
            F.count("*").alias("rows"),
        )
        .orderBy("op_condition")
        .collect()
    )
    limit = int(fc["constant_max_distinct"])
    kept, dropped = [], {}
    for s in sensors:
        most = max(int(r[f"{s}__n"]) for r in per_cond)
        if most <= limit:
            dropped[s] = f"at most {most} distinct value(s) within every operating condition"
        else:
            kept.append(s)
    n_cond = len(centroids)
    by_id = {int(r["op_condition"]): r for r in per_cond}
    norm_mean = {s: [float(by_id[i][f"{s}__mean"]) if i in by_id else 0.0 for i in range(n_cond)] for s in kept}
    norm_std = {s: [float(by_id[i][f"{s}__std"] or 0.0) if i in by_id else 0.0 for i in range(n_cond)] for s in kept}
    norm_std = {s: [v if v > 0 else 1.0 for v in vals] for s, vals in norm_std.items()}  # constant: z = 0
    engines = labelled.groupBy("dataset").agg(F.countDistinct("unit").alias("n")).orderBy("dataset").collect()
    fitted_on = {
        "rows": int(sum(int(r["rows"]) for r in per_cond)),
        "engines": {r["dataset"]: int(r["n"]) for r in engines},
        "rows_per_condition": [int(by_id[i]["rows"]) if i in by_id else 0 for i in range(n_cond)],
        "max_distance_to_centroid": float(max(float(r["max_dist"]) for r in per_cond)),
    }
    labelled.unpersist()
    return FeatureSpec(
        settings=settings,
        settings_mean=mean,
        settings_std=std,
        centroids=centroids,
        kept_sensors=kept,
        dropped_sensors=dropped,
        norm_mean=norm_mean,
        norm_std=norm_std,
        windows=[int(w) for w in fc["windows"]],
        rul_cap=int(fc["rul_cap"]),
        horizon=int(cfg.section("business")["maintenance_lead_time_cycles"]),
        fitted_on=fitted_on,
    )
