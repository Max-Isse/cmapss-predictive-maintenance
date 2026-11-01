"""Metrics, grouped CV (leakage checks), conformal intervals, the pyfunc bundle and the MLflow champion alias.

No Spark: the models train on pandas frames. A small generated fleet (test data only) stands in for gold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cmapss_pm.config import Config, with_overrides
from cmapss_pm.models.data import ModelData

FEATURES = ["cycle", "z_sensor_02", "z_sensor_02_mean_5"]


def _fleet(n_engines: int, split: str, rng: np.random.Generator, truncate: bool) -> pd.DataFrame:
    frames = []
    for unit in range(1, n_engines + 1):
        life = int(rng.integers(80, 200))
        observed = int(rng.integers(30, life)) if truncate else life
        cycle = np.arange(1, observed + 1)
        rul = life - cycle
        z = np.exp(-rul / 40.0) * 3 + rng.normal(0, 0.3, len(cycle))
        frames.append(
            pd.DataFrame(
                {
                    "dataset": "FD001",
                    "split": split,
                    "unit": unit,
                    "cycle": cycle,
                    "z_sensor_02": z,
                    "z_sensor_02_mean_5": pd.Series(z).rolling(5, min_periods=1).mean(),
                    "rul": rul,
                    "is_last_cycle": cycle == observed,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df["rul_capped"] = df.rul.clip(upper=125)
    df["fail_within_n"] = (df.rul <= 30).astype(int)
    df[FEATURES] = df[FEATURES].astype("float32")
    return df


@pytest.fixture
def data() -> ModelData:
    rng = np.random.default_rng(3)
    train, test = _fleet(40, "train", rng, False), _fleet(15, "test", rng, True)
    meta = {
        "gold_delta_version": 0,
        "feature_spec_version": "test",
        "rul_cap": 125,
        "horizon": 30,
        "train_rows": len(train),
        "train_engines": 40,
        "test_rows": len(test),
        "test_engines": 15,
    }
    return ModelData(train, test, FEATURES, meta)


@pytest.fixture
def mcfg(cfg: Config) -> Config:
    models = dict(cfg.section("models"))
    models.update(
        cv_folds=3,
        n_jobs=1,
        cap_sensitivity=[100],
        lightgbm={**models["lightgbm"], "n_estimators": 40, "min_child_samples": 20},
        mlflow={**models["mlflow"], "tracking_uri": "sqlite:///mlruns/mlflow.db"},
    )
    return with_overrides(cfg, cfg.root, models=models)


# ------------------------------------------------------------------ metrics


def test_nasa_score_penalises_late_predictions_more() -> None:
    from cmapss_pm.models.metrics import nasa_score

    assert nasa_score(np.array([50.0]), np.array([60.0])) == pytest.approx(np.e - 1)  # 10 cycles late
    assert nasa_score(np.array([50.0]), np.array([37.0])) == pytest.approx(np.e - 1)  # 13 cycles early
    assert nasa_score(np.array([50.0]), np.array([60.0])) > nasa_score(np.array([50.0]), np.array([40.0]))


def test_threshold_reaches_the_precision_floor() -> None:
    from cmapss_pm.models.metrics import classification_metrics, threshold_at_precision

    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    thr = threshold_at_precision(y, p, 0.75)
    m = classification_metrics(y, p, thr)
    assert m["precision_at_threshold"] >= 0.75 and m["recall_at_threshold"] == 1.0 and thr == 0.4


def test_rul_bands() -> None:
    from cmapss_pm.models.metrics import rul_band

    got = rul_band(pd.Series([0, 30, 31, 125, 126]), [0, 30, 60, 125, 1000]).tolist()
    assert got == ["0-30", "0-30", "31-60", "61-125", ">125"]


def test_conformal_intervals_reach_nominal_coverage_on_new_data() -> None:
    from cmapss_pm.models.metrics import interval_metrics
    from cmapss_pm.models.train import conformal_q_hat

    rng = np.random.default_rng(0)
    y_cal, y_new = rng.normal(0, 2, 4000), rng.normal(0, 2, 4000)
    lo, hi = np.full(4000, -1.0), np.full(4000, 1.0)  # deliberately too narrow quantile models
    q = conformal_q_hat(y_cal, lo, hi, alpha=0.1)
    cover = interval_metrics(y_new, lo - q, hi + q)["coverage"]
    assert 0.88 <= cover <= 0.92


def test_group_corrections_hold_coverage_within_each_band() -> None:
    from cmapss_pm.models.metrics import interval_metrics
    from cmapss_pm.models.pyfunc import conformal_interval, prediction_group
    from cmapss_pm.models.train import conformal_by_group

    rng = np.random.default_rng(1)
    point = rng.uniform(0, 125, 20000)
    y = np.clip(point + rng.normal(0, 1 + point / 10, len(point)), 0, 125)  # noise grows with RUL
    lo, hi = point - 2, point + 2  # same width everywhere: too wide near 0, too narrow near the cap
    bounds = [30.0, 60.0, 90.0]
    cal, new = slice(0, 10000), slice(10000, None)
    groups = prediction_group(point[cal], bounds)
    q = conformal_by_group(y[cal], lo[cal], hi[cal], groups, len(bounds) + 1, alpha=0.1)
    assert q[0] < q[-1]  # narrowed where the model is precise, widened where it is not
    lower, upper = conformal_interval(point[new], lo[new], hi[new], bounds, q, 125.0)
    g_new = prediction_group(point[new], bounds)
    for g in range(len(bounds) + 1):
        m = g_new == g
        assert 0.86 <= interval_metrics(y[new][m], lower[m], upper[m])["coverage"] <= 0.95


# ------------------------------------------------------------------ leakage


def test_grouped_folds_never_split_an_engine(data: ModelData) -> None:
    from cmapss_pm.models.train import grouped_folds

    groups = ModelData.engine_id(data.train)
    folds = grouped_folds(groups, 5, seed=1)
    seen = np.zeros(len(groups), dtype=int)
    for tr, va in folds:
        assert not set(groups.iloc[tr]) & set(groups.iloc[va])
        seen[va] += 1
    assert (seen == 1).all()  # every row is predicted out-of-fold exactly once


def test_model_inputs_exclude_labels_and_identifiers() -> None:
    from cmapss_pm.features.build import LABELS, feature_columns
    from cmapss_pm.features.spec import FeatureSpec

    spec = FeatureSpec(["s1"], [0.0], [1.0], [[0.0]], ["sensor_02"], {}, {}, {}, [5], 125, 30)
    cols = feature_columns(spec)
    assert not set(cols) & {*LABELS, "dataset", "split", "unit", "op_condition", "_spec_version"}
    assert cols[0] == "cycle"  # age is the only key-like input, and it is known at prediction time


# ------------------------------------------------------------------ training, bundle, registry


@pytest.fixture
def trained(data: ModelData, mcfg: Config) -> Any:
    from cmapss_pm.models.train import train_all

    return train_all(data, mcfg)


def test_training_produces_oof_and_holdout_predictions(trained: Any, data: ModelData) -> None:
    res = trained
    assert len(res.oof) == len(data.train) and len(res.test) == len(data.test)
    for col in ("reg_mean_baseline", "reg_linear_ridge", "reg_lightgbm", "reg_lightgbm_cap100", "clf_lightgbm"):
        assert res.oof[col].notna().all() and res.test[col].notna().all()
    assert res.oof.reg_lightgbm.between(0, 125).all() and res.oof.reg_lightgbm_cap100.le(100).all()
    assert (res.test.rul_lower <= res.test.reg_lightgbm).all() and (res.test.reg_lightgbm <= res.test.rul_upper).all()
    out = res.bundle.predict_frame(data.test)  # the served bundle reproduces the evaluated predictions
    assert np.allclose(out.rul_pred, res.test.reg_lightgbm) and np.allclose(out.rul_upper, res.test.rul_upper)
    assert np.allclose(out.p_fail_within_n, res.test.clf_lightgbm_calibrated)
    rmse_lgbm = float(np.sqrt(np.mean((res.oof.reg_lightgbm - res.oof.rul_capped) ** 2)))
    rmse_mean = float(np.sqrt(np.mean((res.oof.reg_mean_baseline - res.oof.rul_capped) ** 2)))
    assert rmse_lgbm < rmse_mean  # the signal is learnable


def test_bundle_rejects_missing_features(trained: Any, data: ModelData) -> None:
    with pytest.raises(ValueError, match="missing feature columns"):
        trained.bundle.predict_frame(data.test.drop(columns=["z_sensor_02"]))


def test_report_and_champion_alias(trained: Any, data: ModelData, mcfg: Config, tmp_path: Path) -> None:
    import mlflow

    from cmapss_pm.models.registry import log_and_register
    from cmapss_pm.models.report import evaluate, write_model_report

    ev = evaluate(trained, mcfg)
    first = log_and_register(mcfg, trained.bundle, ev["flat"], data.test)
    assert first["promoted"] and first["previous_champion_value"] is None
    worse = {**ev["flat"], "cv_rmse_capped": ev["flat"]["cv_rmse_capped"] + 5}
    second = log_and_register(mcfg, trained.bundle, worse, data.test)
    assert not second["promoted"]
    better = {**ev["flat"], "cv_rmse_capped": ev["flat"]["cv_rmse_capped"] - 1}
    third = log_and_register(mcfg, trained.bundle, better, data.test)
    assert third["promoted"] and third["version"] == "3"

    champion = mlflow.pyfunc.load_model("models:/cmapss-maintenance@champion")
    pred = champion.predict(data.test[FEATURES].head(5))
    assert list(pred.columns) == ["rul_pred", "rul_lower", "rul_upper", "p_fail_within_n", "alert"]

    path = write_model_report(trained, ev, mcfg, third)
    text = path.read_text(encoding="utf-8")
    assert "Model comparison" in text and "moved to this version" in text
