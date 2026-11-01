"""Silver rules, quarantine, the pandera contract and the data-quality report (Spark; runs in the container / CI)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cmapss_pm.config import Config


def _valid_values(cfg: Config) -> list[str]:
    """Mid-range values for the 24 settings/sensors (inside the contract)."""
    return [f"{(lo + hi) / 2:.4f}" for lo, hi in cfg.section("validation")["ranges"].values()]


def _line(unit: object, cycle: object, values: list[str]) -> str:
    return " ".join([str(unit), str(cycle), *values]) + " "


def _issues_files(cfg: Config) -> list[Path]:
    """A second delivery that triggers every rule once (test data only)."""
    v = _valid_values(cfg)
    bad_range = list(v)
    bad_range[5] = "99999.0"  # sensor_03 far outside its range
    non_numeric = list(v)
    non_numeric[0] = "abc"
    corrected = list(v)  # unit 1 cycle 1 redelivered with different values
    lines = [
        _line(3, 1, v),
        _line(3, 2, v),
        _line(3, 4, v),
        _line(3, 5, v),  # engine 3: cycle 3 missing
        _line(4, 1, bad_range),
        _line(4, 2, non_numeric),
        _line(4, 0, v),  # invalid key: cycle 0
        "1 6 0.001",  # malformed: 3 tokens
        _line(1, 1, corrected),
    ]
    text_dir = cfg.path("landing_dir") / "text"
    train = text_dir / "train_FD001_issues.txt"
    train.write_text("\n".join(lines) + "\n")
    rul = text_dir / "RUL_FD001_issues.txt"
    rul.write_text("112 \n98 \n-5 \n")  # units 1-2 identical (deduplicated in bronze); unit 3: invalid value
    return [train, rul]


@pytest.fixture
def silver_run(spark: object, landing: Config) -> tuple[Config, object]:
    from cmapss_pm.ingest.bronze import ingest_landing, landing_files
    from cmapss_pm.quality.silver import build_silver

    ingest_landing(spark, landing, landing_files(landing, ("text",)))  # type: ignore[arg-type]
    ingest_landing(spark, landing, _issues_files(landing))  # type: ignore[arg-type]  # later delivery
    return landing, build_silver(spark, landing, run_id="silver-test")  # type: ignore[arg-type]


def test_every_bronze_row_lands_in_silver_or_quarantine(silver_run: tuple[Config, object]) -> None:
    _, res = silver_run
    c = res.counts  # type: ignore[attr-defined]
    # bronze cycles: 10 train + 6 test + 9 from the issues file (none deduplicated: all differ from the originals)
    assert c["bronze_cycles"] == 25
    assert c["bronze_cycles"] == c["silver_cycles"] + c["quarantine_cycles"]
    assert c["bronze_rul"] == c["silver_rul"] + c["quarantine_rul"] == 3


def test_each_rule_quarantines_the_expected_rows(spark: object, silver_run: tuple[Config, object]) -> None:
    from cmapss_pm.quality.silver import silver_path

    cfg, res = silver_run
    assert res.quarantine_by_rule == {  # type: ignore[attr-defined]
        "malformed_row": 1,
        "non_numeric": 1,
        "invalid_key": 1,
        "out_of_range": 1,
        "superseded_duplicate": 1,
        "sequence_gap": 2,
        "invalid_value": 1,
    }
    q = spark.read.format("delta").load(silver_path(cfg, "quarantine"))  # type: ignore[attr-defined]
    rows = {r["rule"]: r for r in q.collect()}
    assert "sensor_03=99999.0" in rows["out_of_range"]["reason"]
    assert "op_setting_1" in rows["non_numeric"]["reason"]
    assert "cycle 3 missing" in rows["sequence_gap"]["reason"]
    assert rows["superseded_duplicate"]["_source_file"] == "train_FD001.txt"  # the original lost to the correction
    assert "train_FD001_issues.txt" in rows["superseded_duplicate"]["reason"]


def test_silver_is_typed_deduplicated_and_sequenced(spark: object, silver_run: tuple[Config, object]) -> None:
    from pyspark.sql import functions as F

    from cmapss_pm.quality.silver import silver_path

    cfg, _ = silver_run
    s = spark.read.format("delta").load(silver_path(cfg, "cycles"))  # type: ignore[attr-defined]
    types = dict(s.dtypes)
    assert types["unit"] == "int" and types["cycle"] == "int" and types["sensor_01"] == "double"
    assert s.groupBy("dataset", "split", "unit", "cycle").count().where("count > 1").count() == 0
    e3 = [r["cycle"] for r in s.where("split = 'train' AND unit = 3").orderBy("cycle").collect()]
    assert e3 == [1, 2]  # prefix before the gap kept
    seq = s.groupBy("dataset", "split", "unit").agg(
        F.min("cycle").alias("lo"), F.max("cycle").alias("hi"), F.count("*").alias("n")
    )
    assert seq.where("lo != 1 OR hi != n").count() == 0  # every engine runs 1..n
    from_correction = s.where("split = 'train' AND unit = 1 AND cycle = 1").select("_source_file").first()[0]
    assert from_correction == "train_FD001_issues.txt"


def test_contract_passes_on_silver_and_catches_violations(spark: object, silver_run: tuple[Config, object]) -> None:
    from pyspark.sql import functions as F

    from cmapss_pm.quality.contract import silver_cycles_schema, silver_rul_schema, validate
    from cmapss_pm.quality.silver import silver_path

    cfg, _ = silver_run
    s = spark.read.format("delta").load(silver_path(cfg, "cycles"))  # type: ignore[attr-defined]
    r = spark.read.format("delta").load(silver_path(cfg, "rul"))  # type: ignore[attr-defined]
    assert validate(silver_cycles_schema(cfg), s) == []
    assert validate(silver_rul_schema(cfg), r) == []
    broken = s.withColumn("sensor_03", F.lit(99999.0)).withColumn("unit", F.lit(-1).cast("int"))
    errors = validate(silver_cycles_schema(cfg), broken)
    assert {e["column"] for e in errors} >= {"sensor_03", "unit"}


def test_rebuilding_silver_is_idempotent(spark: object, silver_run: tuple[Config, object]) -> None:
    from cmapss_pm.quality.silver import build_silver

    cfg, first = silver_run
    second = build_silver(spark, cfg, run_id="silver-test-2")  # type: ignore[arg-type]
    assert second.counts == first.counts  # type: ignore[attr-defined]
    assert second.quarantine_by_rule == first.quarantine_by_rule  # type: ignore[attr-defined]


def test_report_records_counts_and_status(spark: object, silver_run: tuple[Config, object]) -> None:
    from cmapss_pm.quality.report import write_report

    cfg, res = silver_run
    path = write_report(spark, cfg, res, [])  # type: ignore[arg-type]
    text = path.read_text(encoding="utf-8")
    assert "Overall status: **PASSED**" in text and "sequence_gap" in text
    assert (path.parent / "latest.md").exists() and (path.parent / "runs.csv").exists()
