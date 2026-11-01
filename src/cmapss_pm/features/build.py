"""Feature computation shared by training (the gold tables) and serving: one function, Spark DataFrames in and out.

`build_features(df, spec)` takes rows with the silver.cycles columns (dataset, split, unit, cycle, settings,
sensors) and returns one feature row per input row. Labels are added separately (`add_labels`), so serving runs
exactly the code that built the training data, without labels.

Point-in-time rule (tested): the features of engine e at cycle t use only e's rows with cycle <= t. Windows are
`RANGE BETWEEN w-1 PRECEDING AND CURRENT ROW` over the cycle number within the engine, and every other constant
comes from the FeatureSpec fitted on training engines.
"""

from __future__ import annotations

import operator
from functools import reduce
from uuid import uuid4

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from cmapss_pm.features.spec import FeatureSpec

ENGINE = ["dataset", "split", "unit"]
KEY = [*ENGINE, "cycle"]
LABELS = ["rul", "rul_capped", "fail_within_n", "is_last_cycle"]


def feature_columns(spec: FeatureSpec) -> list[str]:
    """Model inputs, in a fixed order: age, normalised sensors, then rolling mean/std/slope per window."""
    cols = ["cycle", *[f"z_{s}" for s in spec.kept_sensors]]
    for w in spec.windows:
        for stat in ("mean", "std", "slope"):
            cols += [f"z_{s}_{stat}_{w}" for s in spec.kept_sensors]
    return cols


def assign_condition(df: DataFrame, spec: FeatureSpec) -> DataFrame:
    """Nearest centroid in standardised setting space -> `op_condition` (id) and `op_condition_dist`."""
    candidates = []
    for i, centre in enumerate(spec.centroids):
        sq = [
            ((F.col(c) - F.lit(v)) / F.lit(s)) ** 2
            for c, v, s in zip(spec.settings, centre, spec.settings_std, strict=True)
        ]
        candidates.append(F.struct(reduce(operator.add, sq).alias("d"), F.lit(i).alias("id")))
    best = F.array_min(F.array(*candidates))  # structs compare field by field: smallest distance, then lowest id
    return df.withColumn("_best", best).select(
        *df.columns,
        F.col("_best.id").alias("op_condition"),
        F.sqrt("_best.d").alias("op_condition_dist"),
    )


def normalise(df: DataFrame, spec: FeatureSpec) -> DataFrame:
    """z_<sensor> = (value - mean of its condition) / std of its condition, for the kept sensors."""

    def lookup(values: list[float]) -> Column:
        return F.element_at(F.array(*[F.lit(float(v)) for v in values]), F.col("op_condition") + 1)

    z = [((F.col(s) - lookup(spec.norm_mean[s])) / lookup(spec.norm_std[s])).alias(f"z_{s}") for s in spec.kept_sensors]
    return df.select("*", *z)


def rolling(df: DataFrame, spec: FeatureSpec) -> DataFrame:
    """Rolling mean and std with the DataFrame API; rolling slope (least squares on cycle) in Spark SQL."""
    by_engine = Window.partitionBy(*ENGINE).orderBy("cycle")
    stats = []
    for w in spec.windows:
        frame = by_engine.rangeBetween(-(w - 1), Window.currentRow)
        for s in spec.kept_sensors:
            stats.append(F.avg(f"z_{s}").over(frame).alias(f"z_{s}_mean_{w}"))
            stats.append(F.stddev_samp(f"z_{s}").over(frame).alias(f"z_{s}_std_{w}"))
    with_stats = df.select("*", *stats)
    view = f"features_{uuid4().hex}"
    with_stats.createOrReplaceTempView(view)
    slopes = ",\n  ".join(
        f"regr_slope(z_{s}, cycle) OVER (PARTITION BY dataset, split, unit ORDER BY cycle "
        f"RANGE BETWEEN {w - 1} PRECEDING AND CURRENT ROW) AS z_{s}_slope_{w}"
        for w in spec.windows
        for s in spec.kept_sensors
    )
    spark = df.sparkSession
    out = spark.sql(f"SELECT *,\n  {slopes}\nFROM {view}")  # analysed now: the view can go
    spark.catalog.dropTempView(view)
    return out


def build_features(df: DataFrame, spec: FeatureSpec) -> DataFrame:
    """Silver-shaped rows -> key, condition, feature columns (see `feature_columns`)."""
    staged = rolling(normalise(assign_condition(df, spec), spec), spec)
    return staged.select(*KEY, "op_condition", "op_condition_dist", *feature_columns(spec)[1:])


def add_labels(features: DataFrame, rul: DataFrame, spec: FeatureSpec) -> DataFrame:
    """RUL, capped RUL and failure within N cycles at every row (labels look ahead; features never do).

    Training engines run to failure, so RUL = last cycle - cycle. Test (and replayed test) engines stop early; their
    RUL at the last cycle comes from silver.rul, so RUL = that value + last cycle - cycle.
    """
    last = F.max("cycle").over(Window.partitionBy(*ENGINE))
    end = rul.select("dataset", "unit", F.col("rul").alias("_rul_at_end"))
    joined = features.join(end, ["dataset", "unit"], "left")
    rul_at_end = F.when(F.col("split") == "train", F.lit(0)).otherwise(F.col("_rul_at_end"))
    labelled = joined.withColumn("_last", last).withColumn("rul", (rul_at_end + F.col("_last") - F.col("cycle")))
    return labelled.select(
        *features.columns,
        F.col("rul").cast("int").alias("rul"),
        F.least(F.col("rul"), F.lit(spec.rul_cap)).cast("int").alias("rul_capped"),
        (F.col("rul") <= spec.horizon).cast("int").alias("fail_within_n"),
        (F.col("cycle") == F.col("_last")).alias("is_last_cycle"),
    )
