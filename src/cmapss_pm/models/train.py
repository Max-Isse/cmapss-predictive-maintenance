"""Grouped cross-validation, baselines and LightGBM models, calibration and conformal intervals.

Protocol (D-020):
1. Training engines only. GroupKFold by engine (`models.cv_folds`), so every out-of-fold (OOF) prediction comes
   from a model that never saw that engine. Model choices, the probability calibrator, the alert threshold and the
   conformal correction are all fitted on OOF predictions.
2. Final models are refitted on all training engines and predict the official test engines once, for the report.
   Nothing is chosen on the test set.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cmapss_pm.config import Config
from cmapss_pm.models.data import ID, ModelData
from cmapss_pm.models.metrics import threshold_at_precision
from cmapss_pm.models.pyfunc import MaintenanceBundle, conformal_interval, prediction_group

log = logging.getLogger(__name__)
Factory = Callable[[], Any]


def lgbm_params(cfg: Config) -> dict[str, Any]:
    mc = cfg.section("models")
    return {
        **mc["lightgbm"],
        "random_state": cfg.seed,
        "n_jobs": int(mc["n_jobs"]),
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }


def regressors(cfg: Config) -> dict[str, Factory]:
    p = lgbm_params(cfg)
    return {
        "mean_baseline": lambda: DummyRegressor(strategy="mean"),
        "linear_ridge": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)),
        "lightgbm": lambda: lgb.LGBMRegressor(objective="regression", **p),
    }


def classifiers(cfg: Config) -> dict[str, Factory]:
    p = lgbm_params(cfg)
    return {
        "prevalence_baseline": lambda: DummyClassifier(strategy="prior"),
        "logistic": lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)
        ),
        "lightgbm": lambda: lgb.LGBMClassifier(objective="binary", **p),
    }


def quantile_models(cfg: Config) -> dict[str, Factory]:
    alpha = float(cfg.section("models")["interval_alpha"])
    p = lgbm_params(cfg)
    return {
        "q_lower": lambda: lgb.LGBMRegressor(objective="quantile", alpha=alpha / 2, **p),
        "q_upper": lambda: lgb.LGBMRegressor(objective="quantile", alpha=1 - alpha / 2, **p),
    }


def grouped_folds(groups: pd.Series, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(groups)), groups=groups.to_numpy()))


def conformal_q_hat(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
    """CQR correction: the (1 - alpha)(1 + 1/n) empirical quantile of max(lower - y, y - upper)."""
    scores = np.maximum(lower - y, y - upper)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def conformal_by_group(
    y: np.ndarray, lower: np.ndarray, upper: np.ndarray, groups: np.ndarray, n_groups: int, alpha: float
) -> list[float]:
    """One CQR correction per group (Mondrian conformal): coverage holds within each band of predicted RUL.

    A group with too few rows for a stable quantile (< 20 / alpha) falls back to the pooled correction.
    """
    pooled = conformal_q_hat(y, lower, upper, alpha)
    out = []
    for g in range(n_groups):
        m = groups == g
        out.append(conformal_q_hat(y[m], lower[m], upper[m], alpha) if m.sum() >= 20 / alpha else pooled)
    return out


@dataclass
class TrainResult:
    oof: pd.DataFrame  # training rows: ids, labels, OOF predictions (reg_*, clf_*, q_*)
    test: pd.DataFrame  # test rows: ids, labels, predictions of the final models
    bundle: MaintenanceBundle
    importances: pd.DataFrame
    seconds: dict[str, float]


def _predict(model: Any, x: pd.DataFrame, kind: str) -> np.ndarray:
    return model.predict_proba(x)[:, 1] if kind == "clf" else model.predict(x)


def train_all(data: ModelData, cfg: Config) -> TrainResult:
    mc = cfg.section("models")
    cap = int(data.meta["rul_cap"])
    alpha = float(mc["interval_alpha"])
    train, test, feats = data.train, data.test, data.features
    x, xt = train[feats], test[feats]
    folds = grouped_folds(ModelData.engine_id(train), int(mc["cv_folds"]), cfg.seed)

    targets = {"reg": train["rul_capped"].to_numpy(float), "clf": train["fail_within_n"].to_numpy(int)}
    jobs: list[tuple[str, str, Factory, np.ndarray, float]] = []  # (column, kind, factory, target, clip cap)
    jobs += [(f"reg_{n}", "reg", f, targets["reg"], cap) for n, f in regressors(cfg).items()]
    # quantile models learn the UNCAPPED RUL: on the capped target the 95% quantile is the cap itself for most rows,
    # the loss has no gradient to split on and the model stays constant (D-022); clipping maps them to the cap
    jobs += [(n, "reg", f, train["rul"].to_numpy(float), cap) for n, f in quantile_models(cfg).items()]
    for c in mc["cap_sensitivity"]:
        y_c = np.minimum(train["rul"].to_numpy(float), c)
        jobs.append((f"reg_lightgbm_cap{c}", "reg", regressors(cfg)["lightgbm"], y_c, float(c)))
    jobs += [(f"clf_{n}", "clf", f, targets["clf"], 1.0) for n, f in classifiers(cfg).items()]

    oof = train[[*ID, "rul", "rul_capped", "fail_within_n", "is_last_cycle"]].copy()
    out = test[[*ID, "rul", "rul_capped", "fail_within_n", "is_last_cycle"]].copy()
    finals: dict[str, Any] = {}
    seconds: dict[str, float] = {}
    for col, kind, make, y, clip in jobs:
        started = time.perf_counter()
        pred = np.zeros(len(train))
        for tr, va in folds:
            model = make().fit(x.iloc[tr], y[tr])
            pred[va] = _predict(model, x.iloc[va], kind)
        final = make().fit(x, y)
        test_pred = _predict(final, xt, kind)
        if kind == "reg":
            pred, test_pred = np.clip(pred, 0, clip), np.clip(test_pred, 0, clip)
        oof[col], out[col] = pred, test_pred
        finals[col] = final
        seconds[col] = round(time.perf_counter() - started, 1)
        log.info("%-24s %5.1fs", col, seconds[col])

    # calibration and alert threshold from OOF probabilities
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof["clf_lightgbm"], targets["clf"])
    oof["clf_lightgbm_calibrated"] = calibrator.predict(oof["clf_lightgbm"])
    out["clf_lightgbm_calibrated"] = calibrator.predict(out["clf_lightgbm"])
    threshold = threshold_at_precision(targets["clf"], oof["clf_lightgbm_calibrated"].to_numpy(), mc["fixed_precision"])

    # conformalized quantile regression from OOF quantiles
    bounds = [float(b) for b in mc["interval_group_bounds"]]
    groups = prediction_group(oof["reg_lightgbm"].to_numpy(), bounds)
    q_hat = conformal_by_group(
        targets["reg"], oof["q_lower"].to_numpy(), oof["q_upper"].to_numpy(), groups, len(bounds) + 1, alpha
    )
    for frame in (oof, out):
        frame["rul_lower"], frame["rul_upper"] = conformal_interval(
            frame["reg_lightgbm"].to_numpy(),
            frame["q_lower"].to_numpy(),
            frame["q_upper"].to_numpy(),
            bounds,
            q_hat,
            cap,
        )

    bundle = MaintenanceBundle(
        features=feats,
        regressor=finals["reg_lightgbm"],
        q_lower=finals["q_lower"],
        q_upper=finals["q_upper"],
        q_hat=q_hat,
        group_bounds=bounds,
        classifier=finals["clf_lightgbm"],
        calibrator=calibrator,
        threshold=threshold,
        rul_cap=cap,
        horizon=int(data.meta["horizon"]),
        alpha=alpha,
        meta=dict(data.meta),
    )
    importances = pd.DataFrame(
        {
            "feature": feats,
            "gain_rul": finals["reg_lightgbm"].booster_.feature_importance("gain"),
            "gain_fail": finals["clf_lightgbm"].booster_.feature_importance("gain"),
        }
    )
    for c in ("gain_rul", "gain_fail"):
        importances[c] = importances[c] / importances[c].sum()
    return TrainResult(oof, out, bundle, importances.sort_values("gain_rul", ascending=False), seconds)
