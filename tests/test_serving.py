"""FastAPI service end to end (no Spark): champion from a temporary MLflow store, gold features in a small Delta
table written with delta-rs, the policy threshold, request IDs, JSON logs and the prediction log."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from deltalake import write_deltalake
from fastapi.testclient import TestClient

from cmapss_pm.config import Config, with_overrides
from cmapss_pm.models.data import ModelData
from tests.test_models import FEATURES, _fleet


def _model_cfg(cfg: Config) -> Config:
    models = dict(cfg.section("models"))
    models.update(
        cv_folds=3,
        n_jobs=1,
        cap_sensitivity=[100],
        lightgbm={**models["lightgbm"], "n_estimators": 30, "min_child_samples": 20},
    )
    return with_overrides(cfg, cfg.root, models=models, serving={"max_batch": 3}, monitoring={"min_rows": 20})


@pytest.fixture
def served(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A registered champion, gold features for 'replay' engines and a policy threshold, all under tmp_path."""
    from cmapss_pm.models.registry import log_and_register
    from cmapss_pm.models.report import evaluate
    from cmapss_pm.models.train import train_all
    from cmapss_pm.monitoring.baseline import build_baseline, save_baseline

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    mcfg = _model_cfg(cfg)
    rng = np.random.default_rng(5)
    train, test = _fleet(30, "train", rng, False), _fleet(8, "replay", rng, True)
    meta = {"gold_delta_version": 0, "feature_spec_version": "test", "rul_cap": 125, "horizon": 30}
    meta |= {"train_rows": len(train), "train_engines": 30, "test_rows": len(test), "test_engines": 8}
    data = ModelData(train, test, FEATURES, meta)
    res = train_all(data, mcfg)
    info = log_and_register(mcfg, res.bundle, evaluate(res, mcfg)["flat"], test)
    from mlflow.tracking import MlflowClient

    save_baseline(build_baseline(res, data, mcfg), mcfg, MlflowClient(), info["run_id"])
    gold = test[["dataset", "split", "unit", "cycle", *FEATURES[1:], "rul", "rul_capped", "fail_within_n"]].copy()
    gold["unit"] = gold["unit"].astype("int32")
    gold["cycle"] = gold["cycle"].astype("int32")
    write_deltalake(str(mcfg.path("lakehouse_dir") / "gold" / "features"), gold, partition_by=["dataset"])
    policy = {"champion_version": info["version"], "threshold": 0.2, "lead_time": 30}
    (mcfg.path("artifacts_dir")).mkdir(parents=True, exist_ok=True)
    (mcfg.path("artifacts_dir") / "policy.json").write_text(json.dumps(policy))
    return {"cfg": mcfg, "bundle": res.bundle, "test": test, "info": info}


@pytest.fixture
def client(served: dict[str, Any]) -> Iterator[TestClient]:
    from cmapss_pm.serving.app import create_app

    with TestClient(create_app(served["cfg"])) as c:
        yield c


def test_health_model_info_and_request_ids(client: TestClient, served: dict[str, Any]) -> None:
    r = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_loaded": True, "model_version": "1"}
    assert r.headers["X-Request-ID"] == "abc123"
    assert len(client.get("/health").headers["X-Request-ID"]) == 32  # generated when absent
    info = client.get("/model-info").json()
    assert info["threshold"] == 0.2 and info["threshold_source"].startswith("cost policy")
    assert info["n_features"] == 3 and info["lead_time_cycles"] == 30 and info["gold_features_version_now"] == 0


def test_predict_by_engine_and_by_features(client: TestClient, served: dict[str, Any]) -> None:
    test = served["test"]
    unit = 3
    last = int(test[test.unit == unit].cycle.max())
    r = client.post("/predict", json={"engine": {"dataset": "FD001", "split": "replay", "unit": unit}})
    assert r.status_code == 200
    body = r.json()["prediction"]
    assert body["engine"]["cycle"] == last  # latest cycle when none is given
    row = test[(test.unit == unit) & (test.cycle == last)]
    expected = served["bundle"].predict_frame(row).iloc[0]
    assert body["rul_pred"] == pytest.approx(expected.rul_pred, rel=1e-6)
    assert body["rul_lower"] <= body["rul_pred"] <= body["rul_upper"]
    assert body["decision"]["book_shop_visit"] == (body["p_fail_within_n"] >= 0.2)

    r = client.post("/predict", json={"engine": {"dataset": "FD001", "split": "replay", "unit": unit, "cycle": 5}})
    assert r.json()["prediction"]["engine"]["cycle"] == 5
    features = {f: float(row[f].iloc[0]) for f in FEATURES}
    r = client.post("/predict", json={"features": features})
    assert r.status_code == 200 and r.json()["prediction"]["engine"] is None
    assert r.json()["prediction"]["rul_pred"] == pytest.approx(body["rul_pred"], rel=1e-6)


def test_validation_and_not_found(client: TestClient) -> None:
    assert client.post("/predict", json={}).status_code == 422  # neither engine nor features
    both = {"engine": {"dataset": "FD001", "unit": 1}, "features": {"cycle": 1.0}}
    assert client.post("/predict", json=both).status_code == 422
    assert client.post("/predict", json={"engine": {"dataset": "FD009", "unit": 1}}).status_code == 422
    missing = client.post("/predict", json={"features": {"cycle": 10.0}})
    assert missing.status_code == 422 and "z_sensor_02" in missing.json()["detail"]
    r = client.post("/predict", json={"engine": {"dataset": "FD001", "split": "replay", "unit": 999}})
    assert r.status_code == 404


def test_batch_reports_errors_per_item_and_limits_size(client: TestClient) -> None:
    items = [
        {"engine": {"dataset": "FD001", "split": "replay", "unit": 1}},
        {"engine": {"dataset": "FD001", "split": "replay", "unit": 999}},
        {"engine": {"dataset": "FD001", "split": "replay", "unit": 2}},
    ]
    r = client.post("/predict/batch", json={"items": items})
    assert r.status_code == 200
    out = r.json()["items"]
    assert out[0]["prediction"] and out[2]["prediction"] and "unit 999" in out[1]["error"]
    assert client.post("/predict/batch", json={"items": items * 2}).status_code == 413


def test_predictions_are_logged_with_their_request_id(client: TestClient, served: dict[str, Any]) -> None:
    from cmapss_pm.serving.store import PredictionLog, prediction_log_path

    client.post(
        "/predict", json={"engine": {"dataset": "FD001", "split": "replay", "unit": 1}}, headers={"X-Request-ID": "r-1"}
    )
    row = served["test"].iloc[[10]]
    client.post(
        "/predict", json={"features": {f: float(row[f].iloc[0]) for f in FEATURES}}, headers={"X-Request-ID": "r-2"}
    )
    logged = PredictionLog(prediction_log_path(served["cfg"])).read().set_index("request_id")
    assert set(logged.index) == {"r-1", "r-2"}
    assert logged.loc["r-1", "source"] == "api-lookup" and logged.loc["r-1", "features_json"] is None
    assert logged.loc["r-2", "source"] == "api-features" and "z_sensor_02" in logged.loc["r-2", "features_json"]
    assert (logged.model_version == "1").all()


def test_json_log_lines_carry_the_request_id() -> None:
    from cmapss_pm.serving.app import REQUEST_ID, JsonFormatter

    token = REQUEST_ID.set("rid-9")
    try:
        record = logging.makeLogRecord({"name": "cmapss.api", "levelname": "INFO", "msg": "request", "status": 200})
        line = json.loads(JsonFormatter().format(record))
    finally:
        REQUEST_ID.reset(token)
    assert line["request_id"] == "rid-9" and line["status"] == 200 and line["message"] == "request"


def test_health_is_503_without_a_model(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    from cmapss_pm.serving.app import create_app

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with TestClient(create_app(cfg)) as c:  # empty MLflow store: no champion
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json={"engine": {"dataset": "FD001", "unit": 1}}).status_code == 503


def test_batch_scoring_and_monitoring(served: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    from cmapss_pm.monitoring import run as mon
    from cmapss_pm.monitoring.report import write_monitoring_report

    cfg = served["cfg"]
    scored = mon.score_split(cfg, "replay")
    assert scored["rows"] == len(served["test"])
    calls: list[str] = []
    fake = {
        "reason": "trigger",
        "champion_version_before": "1",
        "challenger_version": "2",
        "primary_metric": "cv_rmse_capped",
        "challenger_value": 1.0,
        "champion_value": 1.0,
        "promoted": False,
    }
    monkeypatch.setattr(mon, "retrain_challenger", lambda *a, **k: calls.append("retrain") or fake)
    res = mon.monitor(cfg, retrain=True)
    assert res.meta["rows"] == len(served["test"])
    assert {"window", "max_feature_psi", "psi_p_fail_within_n", "performance_degraded"} <= set(res.windows.columns)
    assert calls == (["retrain"] if res.signals["retrain_triggered"] else [])
    assert write_monitoring_report(res, cfg).exists()

    # a broken sensor feed: shifted inputs -> drift, and worse RUL predictions -> performance signal and a retrain
    gold_path = cfg.path("lakehouse_dir") / "gold" / "features"
    broken = served["test"][
        ["dataset", "split", "unit", "cycle", *FEATURES[1:], "rul", "rul_capped", "fail_within_n"]
    ].copy()
    broken[["z_sensor_02", "z_sensor_02_mean_5"]] += 3.0
    broken["unit"], broken["cycle"] = broken["unit"].astype("int32"), broken["cycle"].astype("int32")
    write_deltalake(str(gold_path), broken, mode="overwrite", partition_by=["dataset"])
    mon.score_split(cfg, "replay")
    calls.clear()
    res = mon.monitor(cfg, retrain=True)
    assert res.signals["drift"] and res.signals["performance"] and res.signals["retrain_triggered"]
    assert calls == ["retrain"]
    assert res.windows.max_feature_psi.max() > 0.25
