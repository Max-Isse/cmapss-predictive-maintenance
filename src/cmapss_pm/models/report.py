"""Model comparison: cross-validated and official-holdout metrics, segments, calibration, intervals, cap sensitivity.

Writes reports/model_comparison.md, reports/tables/model_*.csv and reports/figures/model_*.png (generated).
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cmapss_pm.config import Config
from cmapss_pm.models.metrics import (
    calibration_table,
    classification_metrics,
    expected_calibration_error,
    interval_metrics,
    nasa_score,
    regression_metrics,
    rul_band,
    threshold_at_precision,
)
from cmapss_pm.models.pyfunc import prediction_group
from cmapss_pm.models.train import TrainResult

REG = ["reg_mean_baseline", "reg_linear_ridge", "reg_lightgbm"]
CLF = ["clf_prevalence_baseline", "clf_logistic", "clf_lightgbm", "clf_lightgbm_calibrated"]


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return str(df.astype(object).to_markdown(index=False, floatfmt=floatfmt)) if len(df) else "_none_"


def _name(col: str) -> str:
    return col.removeprefix("reg_").removeprefix("clf_")


def evaluate(res: TrainResult, cfg: Config) -> dict[str, Any]:
    mc = cfg.section("models")
    edges = [int(e) for e in mc["rul_bands"]]
    oof, test = res.oof, res.test
    last = test[test.is_last_cycle]
    cap = res.bundle.rul_cap

    cv_reg = pd.DataFrame([{"model": _name(c), **regression_metrics(oof.rul_capped, oof[c])} for c in REG])
    rows = []
    for c in REG:
        rows.append(
            {
                "model": _name(c),
                "rmse_true_rul": regression_metrics(last.rul, last[c])["rmse"],
                "mae_true_rul": regression_metrics(last.rul, last[c])["mae"],
                "nasa_score": nasa_score(last.rul, last[c]),
                "rmse_capped_rul": regression_metrics(last.rul_capped, last[c])["rmse"],
                "rmse_all_cycles_capped": regression_metrics(test.rul_capped, test[c])["rmse"],
            }
        )
    test_reg = pd.DataFrame(rows)
    per_dataset = pd.DataFrame(
        [
            {
                "dataset": d,
                "engines": len(g),
                "rmse_true_rul": regression_metrics(g.rul, g.reg_lightgbm)["rmse"],
                "mae_true_rul": regression_metrics(g.rul, g.reg_lightgbm)["mae"],
                "nasa_score": nasa_score(g.rul, g.reg_lightgbm),
                "rmse_capped_rul": regression_metrics(g.rul_capped, g.reg_lightgbm)["rmse"],
                "share_true_rul_above_cap": float((g.rul > cap).mean()),
                "ridge_rmse_true_rul": regression_metrics(g.rul, g.reg_linear_ridge)["rmse"],
            }
            for d, g in last.groupby("dataset")
        ]
    )
    test = test.assign(band=rul_band(test.rul, edges))
    band_rows = []
    for band, g in test.groupby("band"):
        band_rows.append(
            {
                "band": band,
                "rows": len(g),
                "lightgbm_rmse_capped": regression_metrics(g.rul_capped, g.reg_lightgbm)["rmse"],
                "lightgbm_bias": regression_metrics(g.rul_capped, g.reg_lightgbm)["bias"],
                "ridge_rmse_capped": regression_metrics(g.rul_capped, g.reg_linear_ridge)["rmse"],
                **{f"interval_{k}": v for k, v in interval_metrics(g.rul_capped, g.rul_lower, g.rul_upper).items()},
            }
        )
    bands = pd.DataFrame(band_rows)
    order = {b: i for i, b in enumerate(rul_band(pd.Series(edges[:-1]) + 1, edges))}
    bands = bands.sort_values("band", key=lambda s: s.map(order)).reset_index(drop=True)

    target = float(mc["fixed_precision"])
    clf_rows = []
    for c in CLF:
        thr = threshold_at_precision(oof.fail_within_n.to_numpy(), oof[c].to_numpy(), target)
        cv = classification_metrics(oof.fail_within_n, oof[c], thr)
        te = classification_metrics(test.fail_within_n, test[c], thr)
        tl = classification_metrics(last.fail_within_n, last[c], thr)
        clf_rows.append(
            {
                "model": _name(c),
                "cv_pr_auc": cv["pr_auc"],
                "cv_brier": cv["brier"],
                "threshold_from_cv": thr,
                "test_pr_auc": te["pr_auc"],
                "test_roc_auc": te["roc_auc"],
                "test_brier": te["brier"],
                "test_ece": expected_calibration_error(test.fail_within_n, test[c]),
                "test_precision_at_thr": te["precision_at_threshold"],
                "test_recall_at_thr": te["recall_at_threshold"],
                "last_cycle_pr_auc": tl["pr_auc"],
                "last_cycle_recall_at_thr": tl["recall_at_threshold"],
            }
        )
    clf = pd.DataFrame(clf_rows)
    calib = pd.concat(
        [
            calibration_table(test.fail_within_n, test.clf_lightgbm).assign(model="lightgbm (raw)"),
            calibration_table(test.fail_within_n, test.clf_lightgbm_calibrated).assign(model="lightgbm (isotonic)"),
        ]
    )
    intervals = pd.DataFrame(
        [
            {
                "set": "CV out-of-fold, training engines",
                **interval_metrics(oof.rul_capped, oof.rul_lower, oof.rul_upper),
            },
            {"set": "test, all cycles", **interval_metrics(test.rul_capped, test.rul_lower, test.rul_upper)},
            {"set": "test, last cycle", **interval_metrics(last.rul_capped, last.rul_lower, last.rul_upper)},
            *[
                {"set": f"test, {d}", **interval_metrics(g.rul_capped, g.rul_lower, g.rul_upper)}
                for d, g in test.groupby("dataset")
            ],
        ]
    )
    b = res.bundle
    gb = b.group_bounds
    group_names = [f"<= {gb[0]:.0f}", *[f"{lo:.0f}-{hi:.0f}" for lo, hi in itertools.pairwise(gb)], f"> {gb[-1]:.0f}"]
    g_rows = []
    for i, name in enumerate(group_names):
        o = oof[prediction_group(oof.reg_lightgbm.to_numpy(), b.group_bounds) == i]
        te = test[prediction_group(test.reg_lightgbm.to_numpy(), b.group_bounds) == i]
        g_rows.append(
            {
                "predicted_rul": name,
                "correction_q": b.q_hat[i],
                "cv_rows": len(o),
                "cv_coverage": interval_metrics(o.rul_capped, o.rul_lower, o.rul_upper)["coverage"],
                "test_rows": len(te),
                "test_coverage": interval_metrics(te.rul_capped, te.rul_lower, te.rul_upper)["coverage"],
                "test_mean_width": interval_metrics(te.rul_capped, te.rul_lower, te.rul_upper)["mean_width"],
            }
        )
    interval_groups = pd.DataFrame(g_rows)
    caps = [*[int(c) for c in mc["cap_sensitivity"]], cap]
    cap_rows = []
    for cap_value in sorted(caps):
        col = "reg_lightgbm" if cap_value == cap else f"reg_lightgbm_cap{cap_value}"
        low = oof.rul <= 100
        cap_rows.append(
            {
                "cap": cap_value,
                "cv_rmse_true_rul_le_100": regression_metrics(oof.rul[low], oof[col][low])["rmse"],
                "test_last_rmse_true_rul": regression_metrics(last.rul, last[col])["rmse"],
                "test_last_nasa_score": nasa_score(last.rul, last[col]),
                "test_last_rmse_true_rul_le_125": regression_metrics(
                    last.rul[last.rul <= 125], last[col][last.rul <= 125]
                )["rmse"],
            }
        )
    cap_sens = pd.DataFrame(cap_rows)
    lg = clf[clf.model == "lightgbm_calibrated"].iloc[0].to_dict()
    flat = {
        "cv_rmse_capped": float(cv_reg.loc[cv_reg.model == "lightgbm", "rmse"].iloc[0]),
        "cv_mae_capped": float(cv_reg.loc[cv_reg.model == "lightgbm", "mae"].iloc[0]),
        "cv_pr_auc": float(lg["cv_pr_auc"]),
        "test_last_rmse": float(test_reg.loc[test_reg.model == "lightgbm", "rmse_true_rul"].iloc[0]),
        "test_last_rmse_capped": float(test_reg.loc[test_reg.model == "lightgbm", "rmse_capped_rul"].iloc[0]),
        "test_last_mae": float(test_reg.loc[test_reg.model == "lightgbm", "mae_true_rul"].iloc[0]),
        "test_last_nasa_score": float(test_reg.loc[test_reg.model == "lightgbm", "nasa_score"].iloc[0]),
        "test_pr_auc": float(lg["test_pr_auc"]),
        "test_brier": float(lg["test_brier"]),
        "test_ece": float(lg["test_ece"]),
        "test_precision_at_threshold": float(lg["test_precision_at_thr"]),
        "test_recall_at_threshold": float(lg["test_recall_at_thr"]),
        "test_interval_coverage": float(intervals["coverage"].iloc[1]),
        "test_interval_width": float(intervals["mean_width"].iloc[1]),
    }
    return {
        "cv_reg": cv_reg,
        "test_reg": test_reg,
        "per_dataset": per_dataset,
        "bands": bands,
        "clf": clf,
        "calibration": calib,
        "intervals": intervals,
        "interval_groups": interval_groups,
        "cap_sensitivity": cap_sens,
        "flat": flat,
    }


def _figures(res: TrainResult, ev: dict[str, Any], cfg: Config) -> list[Path]:
    out = cfg.path("figures_dir")
    out.mkdir(parents=True, exist_ok=True)
    last = res.test[res.test.is_last_cycle].sort_values("rul")
    cap = res.bundle.rul_cap
    paths = []
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(last))
    ax.fill_between(x, last.rul_lower, last.rul_upper, color="C0", alpha=0.2, label="90% interval (conformal)")
    ax.plot(x, last.reg_lightgbm, ".", ms=3, color="C0", label="LightGBM prediction")
    ax.plot(x, last.rul, "-", color="k", lw=1, label="true RUL (NASA)")
    ax.axhline(cap, color="0.5", ls="--", lw=0.8, label=f"cap {cap}")
    ax.set_xlabel("test engines, sorted by true RUL")
    ax.set_ylabel("RUL at last observed cycle")
    ax.set_title("Official test set (707 engines): predictions are capped at the target cap by design")
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths.append(out / "model_test_predictions.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for model, g in ev["calibration"].groupby("model"):
        ax.plot(g.mean_predicted, g.observed_rate, marker="o", label=model)
    ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1)
    ax.set_xlabel("mean predicted P(fail within N)")
    ax.set_ylabel("observed rate")
    ax.set_title("Calibration on the test engines (all cycles)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths.append(out / "model_calibration.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)
    return paths


def write_model_report(res: TrainResult, ev: dict[str, Any], cfg: Config, mlflow_info: dict[str, Any]) -> Path:
    tables = cfg.path("tables_dir")
    tables.mkdir(parents=True, exist_ok=True)
    keys = ("cv_reg", "test_reg", "per_dataset", "bands", "clf", "calibration", "intervals", "interval_groups")
    for key in (*keys, "cap_sensitivity"):
        ev[key].to_csv(tables / f"model_{key}.csv", index=False)
    top = res.importances.head(15)
    top.to_csv(tables / "model_feature_importance.csv", index=False)
    figs = _figures(res, ev, cfg)
    b = res.bundle
    mc = cfg.section("models")
    reports = cfg.path("reports_dir")
    fig_list = ", ".join(f"`{p.relative_to(reports).as_posix()}`" for p in figs)
    cal = ev["calibration"][["model", "bin", "rows", "mean_predicted", "observed_rate"]]
    text = f"""<!-- GENERATED by `cmapss train` - do not edit by hand. -->
# Model comparison

Data: gold Delta version {b.meta["gold_delta_version"]}, feature spec `{b.meta["feature_spec_version"]}`;
{b.meta["train_engines"]} training engines ({b.meta["train_rows"]:,} cycles) for cross-validation and fitting;
{b.meta["test_engines"]} official NASA test engines ({b.meta["test_rows"]:,} cycles) as the final holdout, used once
for this report. Protocol and choices: D-020 to D-023; methodology sections 3-4. NASA C-MAPSS is a simulation.

MLflow: run `{mlflow_info["run_id"]}`, registered `{mlflow_info["model"]}` version {mlflow_info["version"]};
alias `{mlflow_info["alias"]}` {"moved to this version" if mlflow_info["promoted"] else "unchanged"}
({mlflow_info["primary_metric"]} = {mlflow_info["value"]:.3f}; previous champion:
{mlflow_info["previous_champion_value"] if mlflow_info["previous_champion_value"] is not None else "none"}).

## 1. RUL regression

Target: RUL capped at {b.rul_cap} (D-016). Cross-validation: {mc["cv_folds"]}-fold GroupKFold by engine on the training
engines, out-of-fold predictions, all cycles.

{_md(ev["cv_reg"])}

Official test set, each engine's last cycle (the standard C-MAPSS evaluation), against the NASA true RUL; NASA
score is the PHM08 asymmetric score summed over engines (late predictions cost more). `rmse_capped_rul` compares
with min(true RUL, {b.rul_cap}); `rmse_all_cycles_capped` uses every test cycle.

{_md(ev["test_reg"])}

LightGBM by dataset (last cycle):

{_md(ev["per_dataset"])}

By true-RUL band, all test cycles (capped target; intervals: 90% nominal):

{_md(ev["bands"])}

## 2. Failure within {b.horizon} cycles

Each model's alert threshold is the lowest probability that reaches precision {mc["fixed_precision"]:.0%} on the
cross-validated predictions (ASSUMPTION: a planner accepts 1 false alert in 5); the test set then shows the
precision and recall that threshold actually achieves. `lightgbm_calibrated` = LightGBM with an isotonic
calibrator fitted on the out-of-fold probabilities.

{_md(ev["clf"])}

Calibration on the test engines (equal-width bins; ECE in the table above):

{_md(cal)}

## 3. Prediction intervals (conformalized quantile regression)

LightGBM quantile models (5% and 95% of the uncapped RUL, clipped to the cap) plus a conformal correction q per band
of *predicted* RUL, computed on the out-of-fold predictions (q < 0 narrows the interval). Nominal coverage
{1 - b.alpha:.0%} of the capped RUL within each band; on out-of-fold rows it holds by construction, so the test rows
are the real check. Above the cap band the true capped RUL is mostly the cap itself, so coverage there is easy;
the decision-relevant bands are the low ones.

{_md(ev["interval_groups"])}

Coverage overall and by dataset (the true-RUL bands are in section 1):

{_md(ev["intervals"])}

## 4. RUL cap sensitivity (D-016)

LightGBM refitted with each cap. `cv_rmse_true_rul_le_100`: out-of-fold RMSE where true RUL <= 100, a range all three
caps can represent.

{_md(ev["cap_sensitivity"])}

## 5. Most important features (share of total gain)

{_md(top)}

Figures: {fig_list}.
"""
    path = reports / "model_comparison.md"
    path.write_text(text, encoding="utf-8")
    return path
