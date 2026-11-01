"""SYNTHETIC REPLAY and the incremental bronze -> silver -> gold stream.

The fixture test file has two engines with three cycles each. With two cycles per batch and one late row, batch 1
delivers cycles 1-2 minus one held-back cycle-1 row (a gap), and batch 2 delivers cycle 3 plus the late row.
"""

from __future__ import annotations

from typing import Any

import pytest

from cmapss_pm.config import Config, with_overrides


@pytest.fixture
def rcfg(landing: Config) -> Config:
    return with_overrides(
        landing,
        landing.root,
        replay={"datasets": ["FD001"], "cycles_per_batch": 2, "late_rows": 1},
        features={"n_conditions": 1},
    )


def test_replay_delivers_every_real_row_once_with_one_late(rcfg: Config) -> None:
    from cmapss_pm.ingest.replay import emit_batch, load_state

    source = (rcfg.path("landing_dir") / "text" / "test_FD001.txt").read_text().splitlines()
    first = emit_batch(rcfg)
    second = emit_batch(rcfg)
    assert first is not None and second is not None and emit_batch(rcfg) is None
    assert [p.name for p in first.files] == ["replay_FD001_SYNTHETIC_b0001.txt"]
    b1 = first.files[0].read_text().splitlines()
    b2 = second.files[0].read_text().splitlines()
    assert (first.rows, first.late_held, second.late_released, second.finished) == (3, 1, 1, True)
    assert sorted(b1 + b2) == sorted(source)  # real lines, byte for byte, each exactly once
    late = load_state(rcfg).batches[0]
    assert late["late_held"] == 1 and late["cycles"] == [1, 2]
    held_line = b2[0]  # released rows lead the next batch
    assert held_line.split()[1] == "1"


def test_predicate_selects_engines_per_dataset_and_split() -> None:
    from cmapss_pm.pipeline.incremental import _predicate

    engines = [("FD001", "replay", 2.0), ("FD001", "replay", 1.0), ("FD002", "replay", None)]
    assert _predicate(engines, "unit", True) == "(dataset = 'FD001' AND split = 'replay' AND (unit IN (1, 2)))"
    loose = _predicate(engines, "u", False)
    assert "(dataset = 'FD002' AND split = 'replay' AND (u IS NULL))" in loose and "u IN (1.0, 2.0)" in loose


def _tables(spark: Any, cfg: Config) -> dict[str, Any]:
    from cmapss_pm.features.gold import gold_path
    from cmapss_pm.quality.silver import silver_path

    return {
        "silver": spark.read.format("delta").load(silver_path(cfg, "cycles")),
        "quarantine": spark.read.format("delta").load(silver_path(cfg, "quarantine")),
        "gold": spark.read.format("delta").load(gold_path(cfg, "features")),
    }


def test_incremental_stream_absorbs_late_rows_and_equals_a_full_rebuild(spark: Any, rcfg: Config) -> None:
    from cmapss_pm.features.gold import build_gold
    from cmapss_pm.ingest.bronze import ingest_landing, landing_files
    from cmapss_pm.ingest.replay import emit_batch
    from cmapss_pm.pipeline.incremental import run_stream, verify_incremental
    from cmapss_pm.quality.silver import build_silver

    ingest_landing(spark, rcfg, landing_files(rcfg, ("text",)))
    build_silver(spark, rcfg, run_id="silver-test")
    build_gold(spark, rcfg, run_id="gold-test")
    assert run_stream(spark, rcfg) == []  # nothing after the full build's bronze version

    # batch 1: one engine's cycle 1 is late, so its cycle 2 waits in quarantine
    b1 = emit_batch(rcfg)
    assert b1 is not None
    ingest_landing(spark, rcfg, b1.files)
    stats = run_stream(spark, rcfg)
    assert len(stats) == 1 and stats[0]["engines"] == 2
    t = _tables(spark, rcfg)
    gaps = t["quarantine"].where("split = 'replay'").collect()
    assert [(g["rule"], g["cycle"]) for g in gaps] == [("sequence_gap", "2")]
    assert t["silver"].where("split = 'replay'").count() == 2  # the on-time engine's cycles 1 and 2
    assert t["gold"].where("split = 'replay'").count() == t["silver"].where("split = 'replay'").count()

    # batch 2: the late row arrives; the engine is recomputed and the quarantine entry disappears
    b2 = emit_batch(rcfg)
    assert b2 is not None and b2.finished
    ingest_landing(spark, rcfg, b2.files)
    run_stream(spark, rcfg)
    t = _tables(spark, rcfg)
    assert t["quarantine"].where("split = 'replay'").count() == 0
    assert t["silver"].where("split = 'replay'").count() == 6
    replay = {(r["unit"], r["cycle"]): r for r in t["gold"].where("split = 'replay'").collect()}
    test = {(r["unit"], r["cycle"]): r for r in t["gold"].where("split = 'test'").collect()}
    assert replay.keys() == test.keys()
    for key, row in replay.items():  # same real values and same history -> same features and labels as the test rows
        assert row["rul"] == test[key]["rul"] and row["z_sensor_02_mean_5"] == test[key]["z_sensor_02_mean_5"]

    check = verify_incremental(spark, rcfg)
    assert check["silver_mismatches"] == check["quarantine_mismatches"] == check["gold_mismatches"] == 0
    assert len(check["spec_versions_in_gold"]) == 1
    assert run_stream(spark, rcfg) == []  # re-running with nothing new does nothing
    from cmapss_pm.pipeline.replay_run import write_replay_report

    path, report_check = write_replay_report(spark, rcfg)
    assert report_check["passed"] and "SYNTHETIC REPLAY" in path.read_text(encoding="utf-8")
