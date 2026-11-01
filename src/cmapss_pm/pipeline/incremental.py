"""Incremental bronze -> silver -> gold: a Structured Streaming query over the bronze Delta table.

Each micro-batch carries the bronze rows appended since the last one (the checkpoint remembers where it stopped;
`trigger(availableNow=True)` processes everything new, then stops, like a scheduled Databricks job). For every
engine touched by the batch, silver and gold are recomputed from that engine's *full* bronze history with the same
rule and feature code as the full rebuild, and swapped in atomically with a Delta `replaceWhere` (D-019):

- correctness: every rule and feature is local to one engine, so the result equals a full rebuild for those
  engines. A late row that arrives after its successors re-opens the engine, and the earlier sequence_gap
  quarantine is replaced by valid rows (verified by `verify_incremental` and the tests);
- cost: the data recomputed is only the touched engines' histories; wall time per batch is dominated by fixed
  overhead (measured in D-019).

The feature spec is loaded from artifacts/ (fitted by `cmapss gold` on training engines), never refitted here.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from delta.tables import DeltaTable
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from cmapss_pm.config import Config
from cmapss_pm.features.build import KEY, add_labels, build_features, end_of_life, feature_columns
from cmapss_pm.features.gold import gold_path
from cmapss_pm.features.spec import load_spec, spec_path
from cmapss_pm.ingest.bronze import bronze_path
from cmapss_pm.quality.contract import silver_cycles_schema, validate
from cmapss_pm.quality.silver import silver_frames, silver_path, state_path

Engine = tuple[str, str, float | None]  # dataset, split, unit as a number (None: unit missing or not numeric)


def unit_key() -> Column:
    """The unit as a number: engine identity across raw spellings ("7", "7.0"); null when missing or not numeric."""
    return F.col("unit").try_cast("double")


def checkpoint_dir(cfg: Config) -> Path:
    return cfg.path("lakehouse_dir") / "_checkpoints" / "bronze_cycles_to_silver_gold"


def affected_engines(batch: DataFrame) -> list[Engine]:
    rows = batch.select("dataset", "split", unit_key().alias("k")).distinct().collect()
    return [(r["dataset"], r["split"], None if r["k"] is None else float(r["k"])) for r in rows]


def _predicate(engines: list[Engine], unit_expr: str, integer_units_only: bool) -> str:
    """SQL predicate selecting the given engines (grouped per dataset and split)."""
    groups: dict[tuple[str, str], list[float | None]] = {}
    for d, s, k in engines:
        groups.setdefault((d, s), []).append(k)
    clauses = []
    for (d, s), keys in sorted(groups.items()):
        known = sorted(k for k in keys if k is not None and (not integer_units_only or k == int(k)))
        parts = []
        if known:
            values = ", ".join(str(int(k)) if integer_units_only else repr(k) for k in known)
            parts.append(f"{unit_expr} IN ({values})")
        if None in keys and not integer_units_only:
            parts.append(f"{unit_expr} IS NULL")
        if parts:
            clauses.append(f"(dataset = '{d}' AND split = '{s}' AND ({' OR '.join(parts)}))")
    return " OR ".join(clauses) if clauses else "false"


def _engine_rows(df: DataFrame, engines: list[Engine], key: Any) -> DataFrame:
    """Rows of `df` belonging to `engines` (null-safe on the unit key)."""
    spark = df.sparkSession
    sel = spark.createDataFrame(engines, "dataset string, split string, _k double")
    return df.join(
        F.broadcast(sel),
        (df["dataset"] == sel["dataset"]) & (df["split"] == sel["split"]) & key.eqNullSafe(sel["_k"]),
        "left_semi",
    )


def update_engines(spark: SparkSession, cfg: Config, engines: list[Engine], run_id: str) -> dict[str, Any]:
    """Recompute silver, quarantine and gold for `engines` from their full bronze history."""
    started = time.perf_counter()
    bronze = spark.read.format("delta").load(bronze_path(cfg, "cycles"))
    history = _engine_rows(bronze, engines, unit_key())
    good, quarantine, cached = silver_frames(history, cfg)
    good = good.withColumn("_silver_run_id", F.lit(run_id)).cache()
    now = datetime.now(UTC).replace(tzinfo=None)
    quarantine = quarantine.withColumn("_silver_run_id", F.lit(run_id)).withColumn("quarantined_ts", F.lit(now)).cache()
    contract_errors = validate(silver_cycles_schema(cfg), good)  # the silver gate (D-009), before anything is written
    if contract_errors:
        raise RuntimeError(f"silver contract failed for {len(engines)} engines: {contract_errors[:3]}")
    silver_pred = _predicate(engines, "unit", integer_units_only=True)
    q_pred = f"source_table = 'cycles' AND ({_predicate(engines, 'try_cast(unit AS DOUBLE)', False)})"
    good.write.format("delta").mode("overwrite").option("replaceWhere", silver_pred).save(silver_path(cfg, "cycles"))
    quarantine.write.format("delta").mode("overwrite").option("replaceWhere", q_pred).save(
        silver_path(cfg, "quarantine")
    )

    spec = load_spec(spec_path(cfg))
    silver_all = spark.read.format("delta").load(silver_path(cfg, "cycles"))
    rul = spark.read.format("delta").load(silver_path(cfg, "rul"))
    rows = silver_all.where(silver_pred)
    gold = add_labels(build_features(rows, spec), end_of_life(silver_all, rul), spec).select(
        "*", F.lit(spec.version).alias("_spec_version"), F.lit(run_id).alias("_gold_run_id")
    )
    gold.write.format("delta").mode("overwrite").option("replaceWhere", silver_pred).save(gold_path(cfg, "features"))
    stats = {
        "engines": len(engines),
        "silver_rows": good.count(),
        "quarantine_rows": quarantine.count(),
        "quarantine_by_rule": {r["rule"]: int(r["count"]) for r in quarantine.groupBy("rule").count().collect()},
        "seconds": round(time.perf_counter() - started, 1),
    }
    for df in [*cached, good, quarantine]:
        df.unpersist()
    return stats


def run_stream(spark: SparkSession, cfg: Config) -> list[dict[str, Any]]:
    """Process every bronze row appended since the last run; returns one stats dict per micro-batch."""
    if not spec_path(cfg).exists():
        raise SystemExit("no feature spec: run `cmapss silver` and `cmapss gold` once before streaming")
    source = bronze_path(cfg, "cycles")
    latest = int(DeltaTable.forPath(spark, source).history(1).first()["version"])  # type: ignore[index]
    ckpt = checkpoint_dir(cfg)
    reader = spark.readStream.format("delta")
    if not ckpt.exists():  # first start: begin right after the bronze version the full silver build read
        st = state_path(cfg, "silver_full_build")
        start = int(json.loads(st.read_text())["bronze_version"]) + 1 if st.exists() else 0
        if start > latest:
            return []  # nothing new yet; no checkpoint is created, so the next run applies the same rule
        reader = reader.option("startingVersion", start)
    results: list[dict[str, Any]] = []

    def process(batch: DataFrame, batch_id: int) -> None:
        engines = affected_engines(batch)
        if not engines:
            return
        run_id = f"stream-{datetime.now(UTC):%Y%m%dT%H%M%S}-b{batch_id}"
        stats = update_engines(batch.sparkSession, cfg, engines, run_id)
        results.append({"micro_batch": batch_id, "run_id": run_id, "bronze_rows": batch.count(), **stats})

    query = (
        reader.load(source)
        .writeStream.foreachBatch(process)
        .option("checkpointLocation", str(ckpt))
        .trigger(availableNow=True)
        .start()
    )
    query.awaitTermination()
    return results


def verify_incremental(spark: SparkSession, cfg: Config) -> dict[str, Any]:
    """Compare the incrementally maintained tables with a full recomputation from bronze (in memory)."""
    bronze = spark.read.format("delta").load(bronze_path(cfg, "cycles"))
    good, quarantine, cached = silver_frames(bronze, cfg)
    silver = spark.read.format("delta").load(silver_path(cfg, "cycles")).drop("_silver_run_id")
    q_cols = quarantine.columns
    q_table = spark.read.format("delta").load(silver_path(cfg, "quarantine")).where("source_table = 'cycles'")
    spec = load_spec(spec_path(cfg))
    rul = spark.read.format("delta").load(silver_path(cfg, "rul"))
    cols = [*KEY, "op_condition", "op_condition_dist", *feature_columns(spec)[1:], "rul", "rul_capped"]
    cols += ["fail_within_n", "is_last_cycle"]
    expected_gold = add_labels(build_features(good, spec), end_of_life(good, rul), spec).select(*cols)
    gold = spark.read.format("delta").load(gold_path(cfg, "features")).select(*cols)

    def diff(a: DataFrame, b: DataFrame) -> int:
        return a.exceptAll(b).count() + b.exceptAll(a).count()

    out = {
        "silver_rows": silver.count(),
        "silver_mismatches": diff(silver.select(*good.columns), good),
        "quarantine_rows": q_table.count(),
        "quarantine_mismatches": diff(q_table.select(*q_cols), quarantine.select(*q_cols)),
        "gold_rows": gold.count(),
        "gold_mismatches": diff(gold, expected_gold),
        "spec_versions_in_gold": [
            r[0]
            for r in spark.read.format("delta")
            .load(gold_path(cfg, "features"))
            .select("_spec_version")
            .distinct()
            .collect()
        ],
    }
    for df in cached:
        df.unpersist()
    return out
