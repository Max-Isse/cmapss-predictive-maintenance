"""Gold layer: fit the feature spec on training engines, build features + labels for every silver row.

gold.features is rebuilt in full on each run, like silver (D-008), partitioned by dataset, one row per
engine x cycle. The spec is saved to artifacts/feature_spec.json (read by the replay and serving) and copied to
reports/tables/feature_spec.json for review.
"""

from __future__ import annotations

import operator
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import reduce
from typing import Any

import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from cmapss_pm.config import Config
from cmapss_pm.features.build import KEY, add_labels, build_features, end_of_life, feature_columns
from cmapss_pm.features.spec import FeatureSpec, fit_spec, save_spec, spec_path
from cmapss_pm.quality.silver import silver_path


def gold_path(cfg: Config, table: str) -> str:
    return str(cfg.path("lakehouse_dir") / "gold" / table)


@dataclass
class GoldResult:
    run_id: str
    spec: FeatureSpec
    counts: dict[str, int]
    labels: list[dict[str, Any]]
    point_in_time: dict[str, Any]


def point_in_time_check(cycles: DataFrame, full: DataFrame, spec: FeatureSpec, seed: int) -> dict[str, Any]:
    """Recompute features with each engine's history cut at a random cycle t and compare with `full` at cycle <= t.

    Equal values (and equal null patterns) prove that no feature reads rows after its own cycle.
    """
    engines = cycles.groupBy("dataset", "split", "unit").agg(F.max("cycle").alias("_n"))
    cut = engines.withColumn("_cut", (F.floor(F.rand(seed) * F.col("_n")) + 1).cast("int")).drop("_n")
    truncated = cycles.join(cut, ["dataset", "split", "unit"]).where(F.col("cycle") <= F.col("_cut")).drop("_cut")
    again = build_features(truncated, spec)
    cols = feature_columns(spec)[1:]
    a = full.select(*KEY, *[F.col(c).alias(f"a_{c}") for c in cols])
    b = again.select(*KEY, *[F.col(c).alias(f"b_{c}") for c in cols])
    both = a.join(b, KEY, "inner")
    diffs = [F.max(F.abs(F.col(f"a_{c}") - F.col(f"b_{c}"))) for c in cols]
    null_mismatch = [F.sum((F.col(f"a_{c}").isNull() != F.col(f"b_{c}").isNull()).cast("int")) for c in cols]
    row = both.agg(
        F.count("*").alias("rows"),
        F.greatest(*diffs).alias("max_abs_diff"),
        reduce(operator.add, null_mismatch).alias("nulls"),
    ).first()
    assert row is not None
    n_truncated = truncated.count()
    return {
        "engines": engines.count(),
        "rows_compared": int(row["rows"]),
        "rows_expected": n_truncated,
        "features_compared": len(cols),
        "max_abs_diff": float(row["max_abs_diff"] or 0.0),
        "null_pattern_mismatches": int(row["nulls"] or 0),
    }


def build_gold(
    spark: SparkSession, cfg: Config, run_id: str | None = None, check_engines: int | None = None
) -> GoldResult:
    run_id = run_id or "gold-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    cycles = spark.read.format("delta").load(silver_path(cfg, "cycles"))
    rul = spark.read.format("delta").load(silver_path(cfg, "rul"))
    spec = fit_spec(cycles.where(F.col("split") == "train"), cfg)
    path = save_spec(spec, spec_path(cfg))
    tables = cfg.path("tables_dir")
    tables.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, tables / "feature_spec.json")

    gold = add_labels(build_features(cycles, spec), end_of_life(cycles, rul), spec).select(
        "*", F.lit(spec.version).alias("_spec_version"), F.lit(run_id).alias("_gold_run_id")
    )
    gold.write.format("delta").mode("overwrite").option("overwriteSchema", "true").partitionBy("dataset").save(
        gold_path(cfg, "features")
    )
    written = spark.read.format("delta").load(gold_path(cfg, "features"))
    labels = (
        written.groupBy("dataset", "split")
        .agg(
            F.countDistinct("unit").alias("engines"),
            F.count("*").alias("rows"),
            F.avg("fail_within_n").alias("fail_within_n_rate"),
            F.avg(F.when(F.col("is_last_cycle"), F.col("fail_within_n"))).alias("fail_within_n_rate_last_cycle"),
            F.avg((F.col("rul") >= spec.rul_cap).cast("int")).alias("share_at_cap"),
            F.sum(F.col("rul").isNull().cast("int")).alias("rows_without_rul"),
        )
        .orderBy("dataset", "split")
        .toPandas()
    )
    sample = cycles
    if check_engines:  # a random subset of engines keeps the check cheap on large data
        units = cycles.select("dataset", "split", "unit").distinct().orderBy(F.rand(cfg.seed)).limit(check_engines)
        sample = cycles.join(units, ["dataset", "split", "unit"])
    pit = point_in_time_check(sample, written, spec, cfg.seed)
    counts = {
        "silver_cycles": cycles.count(),
        "gold_features": written.count(),
        "feature_columns": len(feature_columns(spec)),
    }
    return GoldResult(
        run_id=run_id,
        spec=spec,
        counts=counts,
        labels=[{str(k): v for k, v in r.items()} for r in labels.replace({np.nan: None}).to_dict("records")],
        point_in_time=pit,
    )
