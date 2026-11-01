"""The served model: one MLflow pyfunc bundling RUL point and interval, and the calibrated failure probability.

Input: gold feature columns (from `features/build.py::build_features`). Output per row:
- rul_pred           LightGBM point prediction of the capped RUL, clipped to [0, cap];
- rul_lower/upper    conformalized quantile interval, corrected per band of predicted RUL (1 - alpha nominal
                     coverage of the capped RUL within each band; D-022);
- p_fail_within_n    isotonic-calibrated probability that the engine fails within N cycles;
- alert              p_fail_within_n >= the threshold chosen on cross-validation (fixed precision).
The phase 6 policy replaces `alert` with a cost-based threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModel


def prediction_group(point: np.ndarray, bounds: list[float]) -> np.ndarray:
    """Band index of each point prediction: <= bounds[0] -> 0, (bounds[0], bounds[1]] -> 1, ..., > bounds[-1]."""
    return np.searchsorted(np.asarray(bounds, float), np.asarray(point, float), side="left")


def conformal_interval(
    point: np.ndarray, lower_raw: np.ndarray, upper_raw: np.ndarray, bounds: list[float], q_hat: list[float], cap: float
) -> tuple[np.ndarray, np.ndarray]:
    """Quantile-model bounds widened (or narrowed) by the correction of the point's band, clipped to [0, cap].

    The quantile models predict the uncapped RUL; min(., cap) turns their bounds into bounds for the capped RUL.
    The interval always contains the point prediction.
    """
    q = np.asarray(q_hat, float)[prediction_group(point, bounds)]
    # bounds of the uncapped RUL -> bounds of the capped RUL, then the correction (same order as in calibration)
    lower = np.clip(np.clip(np.asarray(lower_raw, float), 0.0, cap) - q, 0.0, cap)
    upper = np.clip(np.clip(np.asarray(upper_raw, float), 0.0, cap) + q, 0.0, cap)
    return np.minimum(lower, point), np.maximum(upper, point)


@dataclass
class MaintenanceBundle:
    features: list[str]
    regressor: Any
    q_lower: Any
    q_upper: Any
    q_hat: list[float]  # conformal correction per band of predicted RUL
    group_bounds: list[float]
    classifier: Any
    calibrator: Any
    threshold: float
    rul_cap: int
    horizon: int
    alpha: float
    meta: dict[str, Any] = field(default_factory=dict)

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"missing feature columns: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        x = df[self.features].astype("float32")
        cap = float(self.rul_cap)
        point = np.clip(self.regressor.predict(x), 0.0, cap)
        lower, upper = conformal_interval(
            point, self.q_lower.predict(x), self.q_upper.predict(x), self.group_bounds, self.q_hat, cap
        )
        p = self.calibrator.predict(self.classifier.predict_proba(x)[:, 1])
        return pd.DataFrame(
            {
                "rul_pred": point,
                "rul_lower": lower,
                "rul_upper": upper,
                "p_fail_within_n": p,
                "alert": p >= self.threshold,
            },
            index=df.index,
        )


class MaintenancePyfunc(PythonModel):  # type: ignore[misc]
    def __init__(self, bundle: MaintenanceBundle) -> None:
        self.bundle = bundle

    def predict(self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
        return self.bundle.predict_frame(model_input)
