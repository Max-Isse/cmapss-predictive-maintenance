"""Batch scoring into the prediction log, monitoring per window, and the retraining trigger.

`score_split` scores every gold row of a split (the SYNTHETIC REPLAY by default) with the champion and appends them
to the batch predictions table, as the API would one engine at a time into the prediction log; monitoring reads
both. `monitor` joins the logged predictions with the labels
(for the replay these are known in advance from the NASA RUL; a live fleet learns them later) and judges each window
of the replay against the model's training baseline (D-028):

- drift: PSI of every input feature and of both outputs against the baseline deciles;
- performance: RMSE and interval coverage per true-RUL band against the out-of-fold baseline of the same band, so a
  fleet that is simply older (more rows near failure) is not mistaken for a worse model.

Retraining is triggered by the signals in `monitoring.retrain_on`; a retrain is a normal training run, registered as
a challenger, and the champion alias moves only if it improves the primary CV metric (D-023, D-029).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.compute as pc
from deltalake import DeltaTable
from mlflow.tracking import MlflowClient

from cmapss_pm.config import Config
from cmapss_pm.models.metrics import interval_metrics, regression_metrics, rul_band
from cmapss_pm.monitoring.baseline import load_baseline
from cmapss_pm.monitoring.drift import proportions, psi
from cmapss_pm.serving.store import (
    KEY,
    PredictionLog,
    batch_predictions_path,
    gold_features_path,
    load_champion_model,
    prediction_log_path,
)

log = logging.getLogger(__name__)
LABELS = ["rul", "rul_capped", "fail_within_n"]


def read_gold(cfg: Config, split: str, columns: list[str]) -> pd.DataFrame:
    dt = DeltaTable(str(gold_features_path(cfg)))
    return dt.to_pyarrow_dataset().to_table(columns=columns, filter=pc.field("split") == split).to_pandas()


def score_split(cfg: Config, split: str = "replay") -> dict[str, Any]:
    model = load_champion_model(cfg)
    feats = [f for f in model.bundle.features if f not in KEY]
    gold = read_gold(cfg, split, [*KEY, *feats])
    out = model.bundle.predict_frame(gold)
    now = datetime.now(UTC)
    run_id = f"batch-{now:%Y%m%dT%H%M%S}"
    rows = pd.DataFrame(
        {
            "request_id": run_id,
            "logged_at": now,
            "source": "batch",
            "model_name": model.name,
            "model_version": model.version,
            "threshold": model.threshold,
            **{k: gold[k] for k in KEY},
            "rul_pred": out["rul_pred"],
            "rul_lower": out["rul_lower"],
            "rul_upper": out["rul_upper"],
            "p_fail_within_n": out["p_fail_within_n"],
            "book_shop_visit": out["p_fail_within_n"] >= model.threshold,
            "latency_ms": np.nan,
            "features_json": None,
        }
    )
    rows["unit"] = rows["unit"].astype("int64")
    rows["cycle"] = rows["cycle"].astype("int64")
    PredictionLog(batch_predictions_path(cfg)).append_frame(rows)
    return {"run_id": run_id, "rows": len(rows), "model_version": model.version, "split": split}


@dataclass
class MonitorResult:
    model_version: str
    windows: pd.DataFrame
    performance: pd.DataFrame  # per window and band: RMSE and coverage vs baseline
    overall: pd.DataFrame  # whole replay, per band
    top_drift: pd.DataFrame
    signals: dict[str, Any]
    retrain: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _judge_band(g: pd.DataFrame, base: dict[str, float], mc: dict[str, Any], nominal: float) -> dict[str, Any]:
    reg = regression_metrics(g.rul_capped, g.rul_pred)
    iv = interval_metrics(g.rul_capped, g.rul_lower, g.rul_upper)
    enough = len(g) >= int(mc["min_rows"])
    ratio = reg["rmse"] / base["rmse"] if base.get("rmse") else float("nan")
    return {
        "rows": len(g),
        "rmse": reg["rmse"],
        "baseline_rmse": base.get("rmse"),
        "rmse_ratio": ratio,
        "coverage": iv["coverage"],
        "rmse_degraded": bool(enough and ratio > 1 + float(mc["performance_tolerance"])),
        "coverage_degraded": bool(enough and iv["coverage"] < nominal - float(mc["coverage_drop"])),
        "judged": enough,
    }


def monitor(cfg: Config, retrain: bool = False, force_retrain: bool = False) -> MonitorResult:
    mc = cfg.section("monitoring")
    k = int(cfg.section("replay")["cycles_per_batch"])
    edges_rul = [int(e) for e in cfg.section("models")["rul_bands"]]
    model = load_champion_model(cfg)
    baseline = load_baseline(MlflowClient(), model.run_id)
    logs = [PredictionLog(p).read() for p in (batch_predictions_path(cfg), prediction_log_path(cfg))]
    preds = pd.concat([x for x in logs if len(x)], ignore_index=True) if any(len(x) for x in logs) else pd.DataFrame()
    if preds.empty:
        raise SystemExit("prediction log is empty: run `cmapss score` (or call the API) first")
    preds = preds[(preds.model_version == model.version) & (preds.split == "replay")]
    preds = preds.sort_values("logged_at").drop_duplicates(KEY, keep="last")
    feats = [f for f in baseline["features"] if f not in KEY]
    gold = read_gold(cfg, "replay", [*KEY, *feats, *LABELS])
    df = preds.drop(columns=["features_json"]).merge(gold, on=KEY, how="inner")
    df["window"] = ((df.cycle - 1) // k + 1).astype(int)
    nominal = float(baseline["interval_nominal_coverage"])
    alert, share_alert = float(mc["psi_alert"]), float(mc["drift_feature_share_alert"])

    windows, perf, drift_rows = [], [], []
    for w in sorted(int(v) for v in df["window"].unique()):
        g = df[df["window"] == w]
        row: dict[str, Any] = {
            "window": int(w),
            "cycles": f"{(w - 1) * k + 1}-{w * k}",
            "rows": len(g),
            "engines": g[["dataset", "unit"]].drop_duplicates().shape[0],
        }
        enough = len(g) >= int(mc["min_rows"])
        values = {
            f: psi(baseline["features"][f]["proportions"], proportions(g[f], baseline["features"][f]["edges"]))
            for f in [*feats, *(["cycle"] if "cycle" in baseline["features"] else [])]
        }
        drift_rows += [{"window": w, "feature": f, "psi": v} for f, v in values.items()]
        no_age = {f: v for f, v in values.items() if f != "cycle"}
        row["psi_cycle"] = values.get("cycle", float("nan"))
        row["max_feature_psi"] = max(no_age.values())
        row["share_features_psi_alert"] = float(np.mean([v > alert for v in no_age.values()]))
        for name in ("rul_pred", "p_fail_within_n"):
            b = baseline["outputs"][name]
            row[f"psi_{name}"] = psi(b["proportions"], proportions(g[name], b["edges"]))
        row["drift_alert"] = bool(enough and row["share_features_psi_alert"] >= share_alert)
        bands = rul_band(g.rul, edges_rul)
        degraded = False
        for band in sorted(bands.unique()):
            j = _judge_band(g[bands == band], baseline["performance_by_band"].get(band, {}), mc, nominal)
            perf.append({"window": w, "band": band, **j})
            degraded |= j["rmse_degraded"] or j["coverage_degraded"]
        row["performance_degraded"] = degraded
        row["judged"] = enough
        windows.append(row)

    overall = []
    bands = rul_band(df.rul, edges_rul)
    for band in sorted(bands.unique()):
        overall.append(
            {"band": band, **_judge_band(df[bands == band], baseline["performance_by_band"].get(band, {}), mc, nominal)}
        )
    win = pd.DataFrame(windows)
    drift = pd.DataFrame(drift_rows)
    top = (
        drift[drift.feature != "cycle"]
        .sort_values("psi", ascending=False)
        .groupby("window")
        .head(3)
        .sort_values(["window", "psi"], ascending=[True, False])
    )
    overall_df = pd.DataFrame(overall)
    window_flags = int(win.performance_degraded.sum())
    period_degraded = bool((overall_df.rmse_degraded | overall_df.coverage_degraded).any())
    scope = str(mc.get("performance_scope", "period"))
    signals: dict[str, Any] = {
        # D-028: judged over the whole period by default; per-window band cells are diagnostics, because a replay that
        # starts every engine together makes each window one age group, and age shifts errors within a band
        "performance": period_degraded if scope == "period" else bool(window_flags),
        "performance_scope": scope,
        "windows_with_degraded_band_cells": window_flags,
        "drift": bool(win.drift_alert.any()),
    }
    retrain_on = [str(s) for s in mc["retrain_on"]]
    signals["retrain_triggered"] = any(signals[s] for s in retrain_on)
    signals["retrain_on"] = retrain_on
    result = MonitorResult(
        model_version=model.version,
        windows=win,
        performance=pd.DataFrame(perf),
        overall=overall_df,
        top_drift=top,
        signals=signals,
        meta={"rows": len(df), "champion_run_id": model.run_id, "baseline_rows": baseline["data"].get("train_rows")},
    )
    if (retrain and signals["retrain_triggered"]) or force_retrain:
        result.retrain = retrain_challenger(
            cfg, model.version, reason="forced" if force_retrain and not signals["retrain_triggered"] else "trigger"
        )
    return result


def retrain_challenger(cfg: Config, champion_version: str, reason: str) -> dict[str, Any]:
    """Train a challenger on the current gold version; the registry promotes it only if its CV metric is better."""
    from cmapss_pm.models.pipeline import train_and_register  # needs Spark: runs in the pipeline container
    from cmapss_pm.spark import get_spark

    info = train_and_register(get_spark(cfg), cfg, write_report=False)
    return {
        "reason": reason,
        "champion_version_before": champion_version,
        "challenger_version": info["version"],
        "primary_metric": info["primary_metric"],
        "challenger_value": info["value"],
        "champion_value": info["previous_champion_value"],
        "promoted": info["promoted"],
    }
