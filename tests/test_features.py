"""Feature spec, shared feature code, labels and point-in-time correctness (Spark; runs in the container / CI).

Most tests use a small generated frame (test data only): three operating conditions, two constant sensors, one
two-valued sensor and a wear trend, so every rule of the spec has something to find.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from cmapss_pm.config import Config, with_overrides

CONDITIONS = [(0.0, 0.0, 100.0), (25.0, 0.62, 60.0), (42.0, 0.84, 100.0)]  # three of the six real ones
TRAIN_LIVES = {1: 45, 2: 52, 3: 60, 4: 48}
TEST_CUTS = {1: (30, 7), 2: (22, 25)}  # unit: (cycles observed, NASA RUL at the last cycle)


def _engine_rows(rng: np.random.Generator, split: str, unit: int, life: int, observed: int) -> list[dict[str, Any]]:
    rows = []
    for cycle in range(1, observed + 1):
        c = int(rng.integers(len(CONDITIONS)))
        row: dict[str, Any] = {"dataset": "FD002", "split": split, "unit": unit, "cycle": cycle}
        for name, v in zip(["op_setting_1", "op_setting_2", "op_setting_3"], CONDITIONS[c], strict=True):
            row[name] = v + float(rng.normal(0, 0.001))
        wear = (cycle / life) ** 2
        for i in range(1, 22):
            base = 100.0 * (c + 1) + i
            if i in (1, 5):
                value = base  # constant within each condition
            elif i == 16:
                value = base + 0.01 * int(rng.integers(2))  # two values within each condition
            else:
                value = base + float(rng.normal(0, 1)) + 5.0 * wear * (1 if i % 2 else -1)
            row[f"sensor_{i:02d}"] = value
        rows.append(row)
    return rows


def _frames(spark: Any) -> tuple[Any, Any]:
    rng = np.random.default_rng(7)
    rows = []
    for unit, life in TRAIN_LIVES.items():
        rows += _engine_rows(rng, "train", unit, life, life)
    for unit, (observed, rul) in TEST_CUTS.items():
        rows += _engine_rows(rng, "test", unit, observed + rul, observed)
    cycles = spark.createDataFrame(pd.DataFrame(rows))
    rul = spark.createDataFrame(
        pd.DataFrame([{"dataset": "FD002", "unit": u, "rul": r} for u, (_, r) in TEST_CUTS.items()])
    )
    from pyspark.sql import functions as F

    return cycles.withColumn("unit", F.col("unit").cast("int")).withColumn("cycle", F.col("cycle").cast("int")), rul


@pytest.fixture
def fcfg(cfg: Config) -> Config:
    return with_overrides(
        cfg,
        cfg.root,
        features={"n_conditions": 3, "windows": [3, 5], "rul_cap": 20},
        business={"maintenance_lead_time_cycles": 10},
    )


@pytest.fixture
def fitted(spark: Any, fcfg: Config) -> tuple[Any, Any, Any]:
    from pyspark.sql import functions as F

    from cmapss_pm.features.spec import fit_spec

    cycles, rul = _frames(spark)
    return cycles, rul, fit_spec(cycles.where(F.col("split") == "train"), fcfg)


def test_spec_finds_conditions_and_drops_constant_sensors(fitted: tuple[Any, Any, Any]) -> None:
    _, _, spec = fitted
    assert len(spec.centroids) == 3
    for got, want in zip(spec.centroids, CONDITIONS, strict=True):  # sorted by setting, like CONDITIONS
        assert np.allclose(got, want, atol=0.01)
    assert set(spec.dropped_sensors) == {"sensor_01", "sensor_05", "sensor_16"}
    assert len(spec.kept_sensors) == 18
    assert spec.fitted_on["engines"] == {"FD002": 4}  # training engines only
    assert spec.horizon == 10 and spec.rul_cap == 20


def test_spec_json_round_trip(fitted: tuple[Any, Any, Any], tmp_path: Any) -> None:
    from cmapss_pm.features.spec import load_spec, save_spec

    _, _, spec = fitted
    again = load_spec(save_spec(spec, tmp_path / "spec.json"))
    assert again == spec and again.version == spec.version


def test_normalised_sensors_are_standard_within_each_condition(fitted: tuple[Any, Any, Any]) -> None:
    from cmapss_pm.features.build import build_features

    cycles, _, spec = fitted
    f = build_features(cycles.where("split = 'train'"), spec).toPandas()
    stats = f.groupby("op_condition")["z_sensor_02"].agg(["mean", "std"])
    assert np.allclose(stats["mean"], 0, atol=1e-9) and np.allclose(stats["std"], 1, atol=1e-9)


def test_rolling_features_match_a_pandas_reference(fitted: tuple[Any, Any, Any]) -> None:
    from cmapss_pm.features.build import build_features

    cycles, _, spec = fitted
    e = build_features(cycles, spec).where("split = 'train' AND unit = 3").orderBy("cycle").toPandas()
    z = e["z_sensor_02"]
    for w in spec.windows:
        assert np.allclose(e[f"z_sensor_02_mean_{w}"], z.rolling(w, min_periods=1).mean())
        assert np.allclose(e[f"z_sensor_02_std_{w}"], z.rolling(w, min_periods=2).std(), equal_nan=True)
        slope = [
            np.polyfit(e.cycle[max(0, i - w + 1) : i + 1], z[max(0, i - w + 1) : i + 1], 1)[0] if i > 0 else np.nan
            for i in range(len(e))
        ]
        assert np.allclose(e[f"z_sensor_02_slope_{w}"], slope, equal_nan=True)


def test_features_use_only_cycles_up_to_the_prediction_point(fitted: tuple[Any, Any, Any]) -> None:
    from pyspark.sql import functions as F

    from cmapss_pm.features.build import KEY, build_features, feature_columns
    from cmapss_pm.features.gold import point_in_time_check

    cycles, _, spec = fitted
    full = build_features(cycles, spec)
    t = 20
    # the future is replaced by garbage: features at cycle <= t must not change
    garbage = cycles.select(
        *[
            F.when(F.col("cycle") > t, F.lit(1e6)).otherwise(F.col(c)).alias(c) if c.startswith("sensor_") else F.col(c)
            for c in cycles.columns
        ]
    )
    cols = feature_columns(spec)[1:]
    a = full.where(F.col("cycle") <= t).orderBy(*KEY).select(*KEY, *cols).toPandas()
    b = build_features(garbage, spec).where(F.col("cycle") <= t).orderBy(*KEY).select(*KEY, *cols).toPandas()
    c = build_features(cycles.where(F.col("cycle") <= t), spec).orderBy(*KEY).select(*KEY, *cols).toPandas()
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a, c)
    pit = point_in_time_check(cycles, full, spec, seed=1)  # random cut per engine, the check run on real data
    assert pit["rows_compared"] == pit["rows_expected"] > 0
    assert pit["max_abs_diff"] == 0.0 and pit["null_pattern_mismatches"] == 0


def test_serving_one_engine_gives_the_training_rows(fitted: tuple[Any, Any, Any]) -> None:
    from cmapss_pm.features.build import KEY, build_features

    cycles, _, spec = fitted
    batch = build_features(cycles, spec).where("split = 'test' AND unit = 2").orderBy(*KEY).toPandas()
    alone = build_features(cycles.where("split = 'test' AND unit = 2"), spec).orderBy(*KEY).toPandas()
    pd.testing.assert_frame_equal(batch, alone)  # other engines never influence an engine's features


def test_labels(fitted: tuple[Any, Any, Any]) -> None:
    from cmapss_pm.features.build import add_labels, build_features, end_of_life

    cycles, rul, spec = fitted
    g = (
        add_labels(build_features(cycles, spec), end_of_life(cycles, rul), spec)
        .toPandas()
        .set_index(["split", "unit", "cycle"])
    )
    tr = g.loc[("train", 2)]
    assert (tr.rul == TRAIN_LIVES[2] - tr.index).all()
    assert tr.loc[TRAIN_LIVES[2], "rul"] == 0 and bool(tr.loc[TRAIN_LIVES[2], "is_last_cycle"])
    observed, end_rul = TEST_CUTS[1]
    te = g.loc[("test", 1)]
    assert te.loc[observed, "rul"] == end_rul and (te.rul == end_rul + observed - te.index).all()
    assert (g.rul_capped == g.rul.clip(upper=20)).all()
    assert (g.fail_within_n == (g.rul <= 10).astype(int)).all()
    assert tr.loc[TRAIN_LIVES[2] - 10, "fail_within_n"] == 1 and tr.loc[TRAIN_LIVES[2] - 11, "fail_within_n"] == 0
    assert g.is_last_cycle.sum() == len(TRAIN_LIVES) + len(TEST_CUTS)


def test_gold_end_to_end_on_the_fixture_files(spark: Any, landing: Config) -> None:
    from cmapss_pm.features.gold import build_gold
    from cmapss_pm.features.report import write_feature_report
    from cmapss_pm.features.spec import load_spec, spec_path
    from cmapss_pm.ingest.bronze import ingest_landing, landing_files
    from cmapss_pm.quality.silver import build_silver

    cfg = with_overrides(landing, landing.root, features={"n_conditions": 1})  # the fixtures run at sea level only
    ingest_landing(spark, cfg, landing_files(cfg, ("text",)))  # type: ignore[arg-type]
    build_silver(spark, cfg, run_id="silver-test")  # type: ignore[arg-type]
    res = build_gold(spark, cfg, run_id="gold-test")  # type: ignore[arg-type]
    assert res.counts["gold_features"] == res.counts["silver_cycles"] == 16
    assert load_spec(spec_path(cfg)) == res.spec
    assert res.point_in_time["max_abs_diff"] == 0.0
    test_rows = {r["split"]: r for r in res.labels}
    assert test_rows["test"]["rows_without_rul"] == 0
    report = write_feature_report(spark, cfg, res)  # type: ignore[arg-type]
    assert "Point-in-time correctness" in report.read_text(encoding="utf-8")
