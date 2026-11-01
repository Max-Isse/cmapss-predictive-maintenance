"""PSI and the band-stratified performance judgement (no Spark, no MLflow)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cmapss_pm.monitoring.drift import bin_edges, proportions, psi


def test_psi_is_zero_for_the_same_distribution_and_large_for_a_shift() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 20000)
    edges = bin_edges(ref, 10)
    base = proportions(ref, edges)
    assert len(edges) == 9 and abs(sum(base) - 1) < 1e-9
    assert psi(base, proportions(rng.normal(0, 1, 20000), edges)) < 0.01
    assert psi(base, proportions(rng.normal(0.2, 1, 20000), edges)) < 0.1
    assert psi(base, proportions(rng.normal(1.0, 1, 20000), edges)) > 0.25


def test_proportions_track_missing_values_and_discrete_features() -> None:
    edges = bin_edges(np.array([1.0, 1.0, 1.0, 2.0, 2.0, np.nan]), 10)
    assert edges == [1.0, 2.0]  # duplicate quantiles collapse into fewer bins
    p = proportions(np.array([0.5, 1.0, 2.0, np.nan]), edges)
    assert p == [0.25, 0.25, 0.25, 0.25]  # below 1 | [1, 2) | >= 2 | missing


def test_band_judgement_separates_mix_from_degradation() -> None:
    from cmapss_pm.monitoring.run import _judge_band

    mc = {"min_rows": 10, "performance_tolerance": 0.25, "coverage_drop": 0.1}
    rng = np.random.default_rng(1)
    y = rng.uniform(0, 30, 200)
    ok = pd.DataFrame({"rul_capped": y, "rul_pred": y + rng.normal(0, 5, 200)})
    ok["rul_lower"], ok["rul_upper"] = ok.rul_pred - 10, ok.rul_pred + 10
    good = _judge_band(ok, {"rmse": 5.0}, mc, 0.9)
    assert not good["rmse_degraded"] and not good["coverage_degraded"] and good["judged"]
    bad = ok.assign(rul_pred=ok.rul_pred + 15, rul_lower=ok.rul_lower + 15, rul_upper=ok.rul_upper + 15)
    j = _judge_band(bad, {"rmse": 5.0}, mc, 0.9)
    assert j["rmse_degraded"] and j["coverage_degraded"]
    few = _judge_band(bad.head(5), {"rmse": 5.0}, mc, 0.9)
    assert not few["judged"] and not few["rmse_degraded"]  # too few rows to judge
