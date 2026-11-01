"""Silver layer: typed, validated, deduplicated tables built from bronze, with a quarantine for every rejected row.

Silver is rebuilt in full from bronze on each run (deterministic and idempotent; D-008). Row-level rules, in order;
a row is quarantined under the first rule it fails, and every bronze row ends up in exactly one of silver or
quarantine:

1. malformed_row        wrong token count in the delivered line (bronze kept the raw line)
2. non_numeric          a value that does not parse as a number
3. invalid_key          unit/cycle not a positive integer, or cycle above `validation.max_cycle`
4. out_of_range         a setting or sensor outside the contract range (`validation.ranges`, D-007)
5. superseded_duplicate same dataset/split/unit/cycle delivered with different values: the latest delivery wins
6. sequence_gap         per engine, cycles must run 1, 2, 3, ... (strictly increasing by one); rows from the
                        first missing cycle onwards are quarantined (rolling features would be wrong)

The pandera contract (quality/contract.py) then validates the silver table as a whole before it is trusted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from delta.tables import DeltaTable
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from cmapss_pm.config import Config
from cmapss_pm.ingest.bronze import bronze_path

KEY = ["dataset", "split", "unit", "cycle"]
ENGINE = ["dataset", "split", "unit"]


def silver_path(cfg: Config, table: str) -> str:
    return str(cfg.path("lakehouse_dir") / "silver" / table)


def _value_columns(cfg: Config) -> list[str]:
    return [*cfg.raw["schema"]["settings"], *cfg.sensor_columns]


def _first_failure(rules: list[tuple[str, Column, Column]]) -> tuple[Column, Column]:
    """(rule, reason) of the first failing rule; nulls when every rule passes."""
    rule = F.lit(None).cast("string")
    reason = F.lit(None).cast("string")
    for name, failed, detail in reversed(rules):
        rule = F.when(failed, F.lit(name)).otherwise(rule)
        reason = F.when(failed, detail).otherwise(reason)
    return rule, reason


def row_rules(df: DataFrame, cfg: Config) -> DataFrame:
    """Rules 1-4 (per row). Adds typed columns (*_t) and `rule` / `reason`."""
    vc = cfg.section("validation")
    values = _value_columns(cfg)
    typed = df.select(
        "*",
        F.col("unit").try_cast("double").alias("unit_d"),
        F.col("cycle").try_cast("double").alias("cycle_d"),
        *[F.col(c).try_cast("double").alias(f"{c}_t") for c in values],
    )
    # Each per-column test appears once; array_compact drops the nulls of passing columns (linear expression size).
    nn_flags = {
        c: F.col(c).isNotNull() & F.col(f"{c}_d" if c in ("unit", "cycle") else f"{c}_t").isNull()
        for c in ["unit", "cycle", *values]
    }
    nn_fail = F.greatest(*[f.cast("int") for f in nn_flags.values()]) == 1
    nn_detail = F.array_compact(F.array(*[F.when(f, F.lit(c)) for c, f in nn_flags.items()]))
    key_bad = (
        F.col("unit_d").isNull()
        | F.col("cycle_d").isNull()
        | (F.col("unit_d") <= 0)
        | (F.col("cycle_d") <= 0)
        | (F.col("unit_d") != F.floor("unit_d"))
        | (F.col("cycle_d") != F.floor("cycle_d"))
        | (F.col("cycle_d") > float(vc["max_cycle"]))
    )
    oor_flags = {}
    oor_text = []
    for c in values:
        lo, hi = (float(x) for x in vc["ranges"][c])
        flag = (F.col(f"{c}_t") < lo) | (F.col(f"{c}_t") > hi)
        oor_flags[c] = flag
        oor_text.append(F.when(flag, F.concat(F.lit(f"{c}="), F.col(c), F.lit(f" not in [{lo}, {hi}]"))))
    oor_fail = F.coalesce(F.greatest(*[f.cast("int") for f in oor_flags.values()]), F.lit(0)) == 1
    oor_detail = F.array_compact(F.array(*oor_text))
    rule, reason = _first_failure(
        [
            (
                "malformed_row",
                F.col("unit").isNull() & F.col("_raw_line").isNotNull(),
                F.concat(F.lit("wrong number of values in line: "), F.substring("_raw_line", 1, 120)),
            ),
            ("non_numeric", nn_fail, F.concat(F.lit("not numeric: "), F.array_join(nn_detail, ", "))),
            (
                "invalid_key",
                key_bad,
                F.concat(
                    F.lit("unit/cycle must be positive integers <= max_cycle: unit="),
                    F.coalesce("unit", F.lit("null")),
                    F.lit(" cycle="),
                    F.coalesce("cycle", F.lit("null")),
                ),
            ),
            ("out_of_range", oor_fail, F.array_join(oor_detail, "; ")),
        ]
    )
    return typed.withColumn("rule", rule).withColumn("reason", reason)


def resolve_duplicates(df: DataFrame) -> DataFrame:
    """Rule 5 on rows that passed 1-4: keep the latest delivery per key, flag the rest."""
    w = Window.partitionBy(*KEY).orderBy(
        F.col("_ingest_ts").desc(), F.col("_source_file").desc(), F.col("_record_hash").desc()
    )
    ranked = df.withColumn("_rank", F.row_number().over(w)).withColumn("_winner_file", F.first("_source_file").over(w))
    flagged = ranked.withColumn("rule", F.when(F.col("_rank") > 1, F.lit("superseded_duplicate"))).withColumn(
        "reason",
        F.when(
            F.col("_rank") > 1,
            F.concat(
                F.lit("same unit/cycle redelivered with different values; kept the later delivery from "),
                F.col("_winner_file"),
            ),
        ),
    )
    return flagged.drop("_rank", "_winner_file")


def sequence_rule(df: DataFrame) -> DataFrame:
    """Rule 6 on deduplicated rows: cycles per engine must be 1..n with no gaps."""
    w = Window.partitionBy(*ENGINE).orderBy("cycle_i")
    seq = df.withColumn("_expected", F.row_number().over(w)).withColumn(
        "_broken", F.col("cycle_i") != F.col("_expected")
    )
    first_break = (
        seq.where("_broken")
        .groupBy(*ENGINE)
        .agg(F.min("cycle_i").alias("_first_bad_cycle"), F.min("_expected").alias("_missing_cycle"))
    )
    out = seq.join(first_break, ENGINE, "left")
    bad = F.col("_first_bad_cycle").isNotNull() & (F.col("cycle_i") >= F.col("_first_bad_cycle"))
    return (
        out.withColumn("rule", F.when(bad, F.lit("sequence_gap")))
        .withColumn(
            "reason",
            F.when(
                bad,
                F.concat(
                    F.lit("cycle "),
                    F.col("_missing_cycle").cast("string"),
                    F.lit(" missing; rows from the gap onwards quarantined"),
                ),
            ),
        )
        .drop("_expected", "_broken", "_first_bad_cycle", "_missing_cycle")
    )


@dataclass
class SilverResult:
    run_id: str
    started: datetime
    counts: dict[str, int]
    quarantine_by_rule: dict[str, int]
    rul_checks: dict[str, int]
    engines: list[dict[str, Any]]
    extras: dict[str, Any] = field(default_factory=dict)


def silver_frames(bronze: DataFrame, cfg: Config) -> tuple[DataFrame, DataFrame, list[DataFrame]]:
    """Rules 1-6 on bronze cycle rows -> (silver rows, quarantine rows, frames to unpersist).

    Every rule is local to one engine (rules 1-4 to one row), so running this on all bronze rows of a set of
    engines gives exactly the rows a full rebuild gives for those engines. The full build and the incremental
    update (pipeline/incremental.py) both call it (D-019).
    """
    values = _value_columns(cfg)
    bronze_cols = bronze.columns
    checked = row_rules(bronze, cfg).cache()
    passed = checked.where(F.col("rule").isNull()).drop("rule", "reason")
    failed_rows = checked.where(F.col("rule").isNotNull())

    dedup = resolve_duplicates(
        passed.withColumn("cycle_i", F.col("cycle_d").cast("int"))
        .withColumn("unit_i", F.col("unit_d").cast("int"))
        .drop("unit", "cycle")
        .withColumnRenamed("unit_i", "unit")
        .withColumn("cycle", F.col("cycle_i"))
    ).cache()
    superseded = dedup.where(F.col("rule").isNotNull())
    kept = dedup.where(F.col("rule").isNull()).drop("rule", "reason")
    sequenced = sequence_rule(kept).cache()
    gaps = sequenced.where(F.col("rule").isNotNull())
    good = sequenced.where(F.col("rule").isNull())

    silver = good.select(
        "dataset",
        "split",
        F.col("unit").cast("int").alias("unit"),
        F.col("cycle").cast("int").alias("cycle"),
        *[F.col(f"{c}_t").alias(c) for c in values],
        "_record_hash",
        "_source_file",
        F.col("_ingest_ts").alias("_bronze_ingest_ts"),
    )

    def as_quarantine(df: DataFrame) -> DataFrame:
        cols = [F.col(c).cast("string").alias(c) if c in ("unit", "cycle") else F.col(c) for c in bronze_cols]
        return df.select(*cols, "rule", "reason")

    quarantine = (
        as_quarantine(failed_rows)
        .unionByName(as_quarantine(superseded))
        .unionByName(as_quarantine(gaps))
        .withColumn("source_table", F.lit("cycles"))
    )
    return silver, quarantine, [checked, dedup, sequenced]


def state_path(cfg: Config, name: str) -> Path:
    return cfg.path("lakehouse_dir") / "_state" / f"{name}.json"


def build_silver(spark: SparkSession, cfg: Config, run_id: str | None = None) -> SilverResult:
    started = datetime.now(UTC)
    run_id = run_id or f"silver-{started:%Y%m%dT%H%M%S}"
    # pin the bronze version read, so the incremental stream can start right after it (D-019)
    bronze_version = int(DeltaTable.forPath(spark, bronze_path(cfg, "cycles")).history(1).first()["version"])  # type: ignore[index]
    bronze = spark.read.format("delta").option("versionAsOf", bronze_version).load(bronze_path(cfg, "cycles"))

    good, quarantine, cached = silver_frames(bronze, cfg)
    silver = good.withColumn("_silver_run_id", F.lit(run_id))
    silver.write.format("delta").mode("overwrite").option("overwriteSchema", "true").partitionBy("dataset").save(
        silver_path(cfg, "cycles")
    )
    quarantine = quarantine.withColumn("_silver_run_id", F.lit(run_id)).withColumn(
        "quarantined_ts", F.current_timestamp()
    )

    rul_silver, rul_quarantine, rul_checks = build_silver_rul(spark, cfg, run_id, silver)
    all_q = quarantine.unionByName(rul_quarantine, allowMissingColumns=True)
    all_q.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(silver_path(cfg, "quarantine"))

    s = spark.read.format("delta").load(silver_path(cfg, "cycles"))
    q = spark.read.format("delta").load(silver_path(cfg, "quarantine"))
    counts = {
        "bronze_cycles": bronze.count(),
        "silver_cycles": s.count(),
        "quarantine_cycles": q.where(F.col("source_table") == "cycles").count(),
        "bronze_rul": spark.read.format("delta").load(bronze_path(cfg, "rul")).count(),
        "silver_rul": rul_silver,
        "quarantine_rul": q.where(F.col("source_table") == "rul").count(),
    }
    by_rule = {r["rule"]: int(r["count"]) for r in q.groupBy("rule").count().collect()}
    engines = [
        r.asDict()
        for r in s.groupBy("dataset", "split")
        .agg(F.countDistinct("unit").alias("engines"), F.count("*").alias("rows"), F.max("cycle").alias("max_cycle"))
        .orderBy("dataset", "split")
        .collect()
    ]
    for df in cached:
        df.unpersist()
    st = state_path(cfg, "silver_full_build")
    st.parent.mkdir(parents=True, exist_ok=True)
    st.write_text(json.dumps({"run_id": run_id, "bronze_version": bronze_version}), encoding="utf-8")
    return SilverResult(run_id, started, counts, by_rule, rul_checks, engines, {"bronze_version": bronze_version})


def build_silver_rul(
    spark: SparkSession, cfg: Config, run_id: str, silver_cycles: DataFrame
) -> tuple[int, DataFrame, dict[str, int]]:
    """Typed RUL table: one non-negative integer per test engine that exists in silver."""
    rul = spark.read.format("delta").load(bronze_path(cfg, "rul"))
    typed = rul.withColumn("unit_d", F.col("unit").try_cast("double")).withColumn(
        "rul_d", F.col("rul").try_cast("double")
    )
    test_engines = (
        silver_cycles.where(F.col("split") == "test")
        .select("dataset", "unit")
        .distinct()
        .withColumn("_has_engine", F.lit(True))
    )
    joined = (
        typed.join(
            test_engines,
            (typed.dataset == test_engines.dataset) & (typed.unit_d.cast("int") == test_engines.unit),
            "left",
        )
        .drop(test_engines.dataset)
        .drop(test_engines.unit)
    )
    w = Window.partitionBy("dataset", "unit_d").orderBy(F.col("_ingest_ts").desc(), F.col("_source_file").desc())
    ranked = joined.withColumn("_rank", F.row_number().over(w))
    rule, reason = _first_failure(
        [
            ("non_numeric", F.col("rul_d").isNull() | F.col("unit_d").isNull(), F.lit("RUL or unit not numeric")),
            (
                "invalid_value",
                (F.col("rul_d") < 0) | (F.col("rul_d") != F.floor("rul_d")),
                F.concat(F.lit("RUL must be a non-negative integer: "), F.col("rul")),
            ),
            (
                "superseded_duplicate",
                F.col("_rank") > 1,
                F.lit("RUL redelivered for the same engine; kept the later one"),
            ),
            (
                "rul_without_test_engine",
                F.col("_has_engine").isNull(),
                F.lit("no validated test engine with this unit in silver"),
            ),
        ]
    )
    flagged = ranked.withColumn("rule", rule).withColumn("reason", reason).cache()
    good = flagged.where(F.col("rule").isNull()).select(
        "dataset",
        F.col("unit_d").cast("int").alias("unit"),
        F.col("rul_d").cast("int").alias("rul"),
        "_source_file",
        F.col("_ingest_ts").alias("_bronze_ingest_ts"),
        F.lit(run_id).alias("_silver_run_id"),
    )
    good.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(silver_path(cfg, "rul"))
    n_good = spark.read.format("delta").load(silver_path(cfg, "rul")).count()
    quarantine = (
        flagged.where(F.col("rule").isNotNull())
        .select(*rul.columns, "rule", "reason")
        .withColumn("source_table", F.lit("rul"))
        .withColumn("_silver_run_id", F.lit(run_id))
        .withColumn("quarantined_ts", F.current_timestamp())
    )
    missing = (
        test_engines.join(
            good.select("dataset", "unit").withColumn("_has_rul", F.lit(True)), ["dataset", "unit"], "left"
        )
        .where(F.col("_has_rul").isNull())
        .count()
    )
    flagged.unpersist()
    return n_good, quarantine, {"test_engines_without_rul": missing}
