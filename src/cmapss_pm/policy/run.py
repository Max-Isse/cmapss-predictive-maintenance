"""Evaluate the maintenance policy: out-of-fold probabilities, cross-fitted thresholds, baselines, sensitivity.

Data (D-024): the training engines, which run to failure, so every policy's full outcome is known. Their decision
signal is the out-of-fold P(fail within N) of a LightGBM classifier with the champion's settings (one per horizon N
in the sensitivity analysis), calibrated with an isotonic fit on the other folds. The served threshold is chosen on
all training engines with the champion's own calibrator. The official test engines stop before failure, so they
give a partial holdout check (`test_check`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from cmapss_pm.config import Config
from cmapss_pm.models.data import ModelData
from cmapss_pm.models.pyfunc import MaintenanceBundle
from cmapss_pm.models.registry import setup_mlflow
from cmapss_pm.models.train import classifiers, grouped_folds
from cmapss_pm.policy.simulate import Costs, Fleet, PolicyResult, cross_fitted, final_threshold

log = logging.getLogger(__name__)
POLICIES = ["model", "fixed_interval", "run_to_failure", "perfect_information"]


def load_champion(cfg: Config) -> tuple[MaintenanceBundle, Any]:
    mc = cfg.section("models")["mlflow"]
    client, _ = setup_mlflow(cfg)
    version = client.get_model_version_by_alias(mc["registered_model"], mc["champion_alias"])
    model = mlflow.pyfunc.load_model(f"models:/{mc['registered_model']}@{mc['champion_alias']}")
    return model.unwrap_python_model().bundle, version


def oof_probabilities(data: ModelData, cfg: Config, horizons: list[int]) -> pd.DataFrame:
    """Rows of the training engines with fold, raw OOF probability and cross-fitted calibrated probability per N."""
    train = data.train
    rows = train[["dataset", "unit", "cycle", "rul"]].copy()
    rows["engine"] = ModelData.engine_id(train)
    folds = grouped_folds(rows["engine"], int(cfg.section("models")["cv_folds"]), cfg.seed)
    fold = np.full(len(rows), -1)
    for k, (_, va) in enumerate(folds):
        fold[va] = k
    rows["fold"] = fold
    x = train[data.features]
    make = classifiers(cfg)["lightgbm"]
    for n in horizons:
        y = (train["rul"] <= n).astype(int).to_numpy()
        raw = np.zeros(len(train))
        for tr, va in folds:
            raw[va] = make().fit(x.iloc[tr], y[tr]).predict_proba(x.iloc[va])[:, 1]
        cal = np.zeros(len(train))
        for k in range(len(folds)):
            held = (rows["fold"] == k).to_numpy()
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw[~held], y[~held])
            cal[held] = iso.predict(raw[held])
        rows[f"raw_{n}"], rows[f"p_{n}"] = raw, cal
        log.info("horizon %s: OOF probabilities ready", n)
    return rows


def summaries(res: PolicyResult, fleet: Fleet, by_dataset: bool = False) -> pd.DataFrame:
    out = []
    for name in POLICIES:
        o = res.outcomes[name]
        if by_dataset:
            for d in sorted(fleet.engines["dataset"].unique()):
                m = (fleet.engines["dataset"] == d).to_numpy()
                part = type(o)(o.cost[m], o.failed[m], o.lost[m], o.flown[m])
                out.append({"policy": name, "dataset": d, **part.summary()})
        else:
            out.append({"policy": name, **o.summary()})
    return pd.DataFrame(out)


def scenario_row(label: str, res: PolicyResult) -> dict[str, Any]:
    c = {k: float(v.cost.mean()) for k, v in res.outcomes.items()}
    best_baseline = min(c["fixed_interval"], c["run_to_failure"])
    return {
        "scenario": label,
        **{f"{k}_cost_per_engine": c[k] for k in POLICIES},
        "model_failure_rate": float(res.outcomes["model"].failed.mean()),
        "fixed_interval_failure_rate": float(res.outcomes["fixed_interval"].failed.mean()),
        "saving_vs_best_baseline": best_baseline - c["model"],
        "saving_vs_best_baseline_pct": (best_baseline - c["model"]) / best_baseline,
        "share_of_attainable_saving": (best_baseline - c["model"]) / (best_baseline - c["perfect_information"]),
        "median_threshold": float(np.median(list(res.thresholds_by_fold.values()))),
    }


def test_check(bundle: MaintenanceBundle, data: ModelData, threshold: float, lead: int, costs: Costs) -> pd.DataFrame:
    """Official test engines (stopped before failure): what the served policy would have done, where knowable.

    decided in window: outcome known (in time if RUL at the decision > lead time);
    not decided, RUL at the last cycle <= lead: a failure is certain (any later booking is too late);
    not decided otherwise: censored, the outcome lies after the observed window.
    """
    test = data.test
    p = bundle.predict_frame(test)["p_fail_within_n"].to_numpy()
    df = test[["dataset", "unit", "cycle", "rul"]].assign(p=p, engine=ModelData.engine_id(test))
    rows = []
    for engine, g in df.sort_values(["engine", "cycle"]).groupby("engine"):
        hit = g[g.p >= threshold]
        last = g.iloc[-1]
        if len(hit):
            rul_at = float(hit.iloc[0].rul)
            status = "decided: in time" if rul_at > lead else "decided: too late"
            lost = rul_at - lead if rul_at > lead else float("nan")
        else:
            rul_at, lost = float("nan"), float("nan")
            status = "not decided: failure certain" if last.rul <= lead else "not decided: censored"
        rows.append(
            {"engine": engine, "dataset": last.dataset, "status": status, "rul_at_decision": rul_at, "lost": lost}
        )
    per_engine = pd.DataFrame(rows)
    return (
        per_engine.groupby("status")
        .agg(engines=("engine", "size"), mean_rul_at_decision=("rul_at_decision", "mean"), mean_lost=("lost", "mean"))
        .reset_index()
    )


@dataclass
class PolicyRun:
    base: PolicyResult
    fleet: Fleet
    threshold: float
    curve: pd.DataFrame
    summary: pd.DataFrame
    by_dataset: pd.DataFrame
    scenarios: pd.DataFrame
    test: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)


def run_policy(data: ModelData, cfg: Config) -> PolicyRun:
    pc = cfg.section("policy")
    lead = int(cfg.section("business")["maintenance_lead_time_cycles"])
    bundle, version = load_champion(cfg)
    for key in ("gold_delta_version", "feature_spec_version"):
        if bundle.meta[key] != data.meta[key]:
            raise SystemExit(f"champion was trained on {key}={bundle.meta[key]}, data is {data.meta[key]}: retrain")
    n_base = bundle.horizon
    horizons = sorted({int(h) for h in pc["horizons"]} | {int(n) for n in pc["lead_time_scenarios"]} | {n_base})
    rows = oof_probabilities(data, cfg, horizons)
    costs = Costs.from_config(cfg)
    grid, interval = int(pc["threshold_grid_points"]), tuple(int(v) for v in pc["fixed_interval_candidates"])

    fleet = Fleet.from_rows(rows, f"p_{n_base}")
    base = cross_fitted(fleet, lead, costs, grid, interval)  # type: ignore[arg-type]
    # served threshold: all training engines, the champion's calibrator (fitted on the same OOF probabilities)
    served = rows.assign(p_served=bundle.calibrator.predict(rows[f"raw_{n_base}"]))
    threshold, curve = final_threshold(Fleet.from_rows(served, "p_served"), lead, costs, grid)

    scen = [scenario_row(f"base: N={n_base}, L={lead}, costs as configured", base)]
    for param, values in pc["cost_scenarios"].items():
        key = {"unplanned_failure_cost": "unplanned", "planned_maintenance_cost": "planned"}.get(param, "lost_life")
        for v in values:
            res = cross_fitted(fleet, lead, Costs.from_config(cfg, **{key: float(v)}), grid, interval)  # type: ignore[arg-type]
            scen.append(scenario_row(f"{param} = {v}", res))
    for n in pc["horizons"]:
        if int(n) != n_base:
            res = cross_fitted(Fleet.from_rows(rows, f"p_{n}"), lead, costs, grid, interval)  # type: ignore[arg-type]
            scen.append(scenario_row(f"horizon N = {n} (L = {lead})", res))
    for lt in pc["lead_time_scenarios"]:
        if int(lt) != lead:
            res = cross_fitted(Fleet.from_rows(rows, f"p_{lt}"), int(lt), costs, grid, interval)  # type: ignore[arg-type]
            scen.append(scenario_row(f"lead time L = {lt} (N = {lt})", res))

    meta = {
        "champion_version": version.version,
        "champion_run_id": version.run_id,
        "gold_delta_version": data.meta["gold_delta_version"],
        "feature_spec_version": data.meta["feature_spec_version"],
        "horizon": n_base,
        "lead_time": lead,
        "costs": costs.__dict__,
        "threshold": threshold,
        "threshold_by_fold": base.thresholds_by_fold,
        "fixed_interval_by_fold": base.intervals_by_fold,
    }
    return PolicyRun(
        base=base,
        fleet=fleet,
        threshold=threshold,
        curve=curve,
        summary=summaries(base, fleet),
        by_dataset=summaries(base, fleet, by_dataset=True),
        scenarios=pd.DataFrame(scen),
        test=test_check(bundle, data, threshold, lead, costs),
        meta=meta,
    )


def save_policy(cfg: Config, run: PolicyRun) -> dict[str, Any]:
    """artifacts/policy.json (read by serving), a copy in reports/tables, and tags on the champion model version."""
    cost = dict(zip(run.summary["policy"], run.summary["cost_per_engine"], strict=True))
    doc = {
        **run.meta,
        "rule": "book a shop visit when p_fail_within_n >= threshold",
        "expected_cost_per_engine_cross_fitted": float(cost["model"]),
        "note": "costs are ASSUMPTIONS (configs/config.yaml, business); NASA C-MAPSS is a simulation",
    }
    text = json.dumps(doc, indent=2, default=str)
    for path in (cfg.path("artifacts_dir") / "policy.json", cfg.path("tables_dir") / "policy.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    client, _ = setup_mlflow(cfg)
    name = cfg.section("models")["mlflow"]["registered_model"]
    client.set_model_version_tag(name, str(run.meta["champion_version"]), "policy_threshold", f"{run.threshold:.6f}")
    client.set_model_version_tag(
        name, str(run.meta["champion_version"]), "policy_lead_time", str(run.meta["lead_time"])
    )
    return doc
