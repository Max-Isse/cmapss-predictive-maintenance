"""Policy simulation: timeline and costs, decision cycles, cross-fitting without the held-out fold, holdout check."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cmapss_pm.policy.simulate import (
    Costs,
    Fleet,
    best_fixed_interval,
    cross_fitted,
    final_threshold,
    outcome_from_decision,
    subset,
)

COSTS = Costs(unplanned=1000.0, planned=200.0, lost_life=1.0)


def test_timeline_and_costs() -> None:
    eol = np.array([100.0, 100.0, 100.0, 100.0])
    decision = np.array([60.0, 69.0, 70.0, np.inf])  # RUL at decision 40, 31, 30, never
    o = outcome_from_decision(eol, decision, lead=30, costs=COSTS)
    assert o.failed.tolist() == [False, False, True, True]  # RUL must exceed the lead time
    assert o.lost.tolist() == [10.0, 1.0, 0.0, 0.0]
    assert o.cost.tolist() == [210.0, 201.0, 1000.0, 1000.0]
    assert o.flown.tolist() == [90.0, 99.0, 100.0, 100.0]


def _rows(n_engines: int, seed: int, noise: float = 0.05) -> pd.DataFrame:
    """Engines whose probability rises as failure nears (test data only), 5 folds, two datasets."""
    rng = np.random.default_rng(seed)
    frames = []
    for e in range(n_engines):
        life = int(rng.integers(120, 300))
        cycle = np.arange(1, life + 1)
        rul = life - cycle
        p = np.clip(1 / (1 + np.exp((rul - 30) / 6)) + rng.normal(0, noise, life), 0, 1)
        frames.append(
            pd.DataFrame(
                {
                    "engine": f"FD00{1 + e % 2}/{e}",
                    "dataset": f"FD00{1 + e % 2}",
                    "cycle": cycle,
                    "rul": rul,
                    "p": p,
                    "fold": e % 5,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_decision_cycle_is_the_first_crossing_of_the_running_maximum() -> None:
    rows = pd.DataFrame(
        {
            "engine": "e",
            "dataset": "FD001",
            "cycle": [1, 2, 3, 4],
            "rul": [3, 2, 1, 0],
            "p": [0.1, 0.6, 0.2, 0.9],
            "fold": 0,
        }
    )
    fleet = Fleet.from_rows(rows, "p")
    assert fleet.engines.eol.tolist() == [4.0]
    got = fleet.decision_cycles(np.array([0.05, 0.5, 0.7, 0.95]))[0]
    assert got[:3].tolist() == [1.0, 2.0, 4.0] and np.isinf(got[3])


def test_fixed_interval_optimum() -> None:
    eol = np.array([100.0, 200.0])
    # removing at 99 keeps both planned: cost (200+1)+(200+101); at 199 the first fails: 1000+(200+1)
    t = best_fixed_interval(eol, np.arange(50.0, 251.0), lead=30, costs=COSTS)
    assert t == 99.0


def test_model_policy_beats_baselines_when_the_signal_is_informative() -> None:
    fleet = Fleet.from_rows(_rows(100, 0), "p")
    res = cross_fitted(fleet, lead=30, costs=COSTS, grid_points=200, interval_range=(50, 400))
    cost = {k: v.cost.mean() for k, v in res.outcomes.items()}
    assert cost["perfect_information"] == pytest.approx(201.0)  # planned + 1 lost cycle per engine
    assert cost["run_to_failure"] == 1000.0
    assert cost["perfect_information"] < cost["model"] < cost["fixed_interval"] < cost["run_to_failure"]
    assert set(res.thresholds_by_fold) == {0, 1, 2, 3, 4}


def test_cross_fitting_never_uses_the_held_out_fold() -> None:
    from cmapss_pm.policy.simulate import model_costs, threshold_grid

    rows = _rows(60, 1)
    rows.loc[rows.fold == 0, "p"] = 1.0  # fold 0 would pull a threshold choice anywhere if it were used
    fleet = Fleet.from_rows(rows, "p")
    res = cross_fitted(fleet, lead=30, costs=COSTS, grid_points=200, interval_range=(50, 400))
    others = subset(fleet, (fleet.engines.fold != 0).to_numpy())
    grid = threshold_grid(np.concatenate(others.running_max), 200)
    assert res.thresholds_by_fold[0] == float(grid[int(np.argmin(model_costs(others, grid, 30, COSTS)))])
    for d in ("FD001", "FD002"):
        fit = others.engines[others.engines.dataset == d].eol.to_numpy()
        assert res.intervals_by_fold[0][d] == best_fixed_interval(fit, np.arange(50.0, 401.0), 30, COSTS)


def test_served_threshold_minimises_the_cost_curve() -> None:
    fleet = Fleet.from_rows(_rows(50, 2), "p")
    tau, curve = final_threshold(fleet, lead=30, costs=COSTS, grid_points=100)
    assert curve.cost_per_engine.min() == pytest.approx(curve.loc[curve.threshold == tau, "cost_per_engine"].iloc[0])


def test_holdout_check_statuses() -> None:
    from cmapss_pm.models.data import ModelData
    from cmapss_pm.policy.run import test_check

    test = pd.DataFrame(
        {
            "dataset": "FD001",
            "split": "test",
            "unit": [1, 1, 2, 2, 3, 3, 4, 4],
            "cycle": [1, 2] * 4,
            "rul": [41, 40, 21, 20, 61, 60, 25, 24],
            "p": [0.1, 0.9, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1],
        }
    )

    class Fake:
        def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"p_fail_within_n": df["p"].to_numpy()}, index=df.index)

    data = ModelData(test.iloc[:0], test, [], {})
    out = test_check(Fake(), data, threshold=0.5, lead=30, costs=COSTS).set_index("status")  # type: ignore[arg-type]
    assert out.engines.to_dict() == {
        "decided: in time": 1,  # unit 1: decided at RUL 40
        "decided: too late": 1,  # unit 2: decided at RUL 20
        "not decided: censored": 1,  # unit 3: RUL 60 at the last cycle, outcome after the window
        "not decided: failure certain": 1,  # unit 4: RUL 24 <= lead time, no decision yet
    }
    assert out.loc["decided: in time", "mean_lost"] == 10.0
