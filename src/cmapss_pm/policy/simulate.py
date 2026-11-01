"""Engine-life simulation of maintenance policies under the cost assumptions (pure numpy/pandas, no Spark).

Timeline (D-024): an engine fails after its last cycle, `eol`. A decision at the end of cycle t books a shop visit
after cycle t + L (L = lead time). The visit is in time when the engine still had more than L cycles left at the
decision (RUL at decision > L); the cycles it could still have flown, RUL - L, are lost life. Otherwise the engine
fails in service first.

Cost per engine (assumptions in `business`):
- in-service failure: unplanned_failure_cost (no planned visit is charged: the failure replaces it);
- planned visit: planned_maintenance_cost + lost_life_cost_per_cycle x lost cycles.

Policies:
- model: decide at the first cycle with P(fail within N) >= threshold;
- fixed interval: remove at cycle T (booked at T - L), T chosen per dataset;
- run to failure: never decide; every engine fails;
- perfect information (lower bound): decide exactly when RUL = L + 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cmapss_pm.config import Config


@dataclass(frozen=True)
class Costs:
    unplanned: float
    planned: float
    lost_life: float

    @classmethod
    def from_config(cls, cfg: Config, **overrides: float) -> Costs:
        b = cfg.section("business")
        values = {
            "unplanned": float(b["unplanned_failure_cost"]),
            "planned": float(b["planned_maintenance_cost"]),
            "lost_life": float(b["lost_life_cost_per_cycle"]),
        }
        return cls(**{**values, **overrides})


@dataclass
class Outcome:
    """Per-engine result of one policy: cost, whether it failed in service, lost cycles, cycles flown."""

    cost: np.ndarray
    failed: np.ndarray
    lost: np.ndarray
    flown: np.ndarray

    def summary(self) -> dict[str, float]:
        planned = ~self.failed
        return {
            "engines": len(self.cost),
            "failures": int(self.failed.sum()),
            "failure_rate": float(self.failed.mean()),
            "planned": int(planned.sum()),
            "mean_lost_cycles_when_planned": float(self.lost[planned].mean()) if planned.any() else float("nan"),
            "mean_cycles_flown": float(self.flown.mean()),
            "cost_per_engine": float(self.cost.mean()),
            "total_cost": float(self.cost.sum()),
        }


def outcome_from_decision(eol: np.ndarray, decision: np.ndarray, lead: int, costs: Costs) -> Outcome:
    """`decision`: cycle of the booking decision per engine (np.inf = never)."""
    eol = np.asarray(eol, float)
    decision = np.asarray(decision, float)
    rul_at = eol - decision  # -inf when never decided
    planned = np.isfinite(decision) & (rul_at > lead)
    lost = np.where(planned, rul_at - lead, 0.0)
    cost = np.where(planned, costs.planned + costs.lost_life * lost, costs.unplanned)
    flown = np.where(planned, decision + lead, eol)
    return Outcome(cost=cost, failed=~planned, lost=lost, flown=flown)


@dataclass
class Fleet:
    """Per-engine trajectories of the decision signal: cycles and running maximum of the probability."""

    engines: pd.DataFrame  # one row per engine: engine, dataset, eol, fold
    cycles: list[np.ndarray]
    running_max: list[np.ndarray]

    @classmethod
    def from_rows(cls, rows: pd.DataFrame, prob: str) -> Fleet:
        """`rows`: one row per engine-cycle with columns engine, dataset, cycle, rul, fold and `prob`."""
        cycles, running_max, meta = [], [], []
        for engine, g in rows.sort_values(["engine", "cycle"]).groupby("engine", sort=True):
            c = g["cycle"].to_numpy(float)
            cycles.append(c)
            running_max.append(np.maximum.accumulate(g[prob].to_numpy(float)))
            meta.append(
                {
                    "engine": engine,
                    "dataset": g["dataset"].iloc[0],
                    "eol": float(c[-1] + g["rul"].iloc[-1]),
                    "last_cycle": float(c[-1]),
                    "fold": int(g["fold"].iloc[0]),
                }
            )
        return cls(pd.DataFrame(meta), cycles, running_max)

    def decision_cycles(self, thresholds: np.ndarray) -> np.ndarray:
        """First cycle at which the probability reaches each threshold: shape (engines, thresholds); inf = never."""
        out = np.full((len(self.cycles), len(thresholds)), np.inf)
        for i, (c, m) in enumerate(zip(self.cycles, self.running_max, strict=True)):
            idx = np.searchsorted(m, thresholds, side="left")  # m is non-decreasing
            hit = idx < len(c)
            out[i, hit] = c[idx[hit]]
        return out


def threshold_grid(p: np.ndarray, points: int) -> np.ndarray:
    return np.unique(np.quantile(np.asarray(p, float), np.linspace(0.0, 1.0, points)))


def model_costs(fleet: Fleet, thresholds: np.ndarray, lead: int, costs: Costs) -> np.ndarray:
    """Mean cost per engine for each threshold (rows of `fleet`)."""
    eol = fleet.engines["eol"].to_numpy()
    decisions = fleet.decision_cycles(thresholds)
    return np.array(
        [outcome_from_decision(eol, decisions[:, j], lead, costs).cost.mean() for j in range(len(thresholds))]
    )


def fixed_interval_decisions(eol: np.ndarray, removal: float, lead: int) -> np.ndarray:
    return np.full(len(eol), removal - lead, dtype=float)


def best_fixed_interval(eol: np.ndarray, candidates: np.ndarray, lead: int, costs: Costs) -> float:
    means = [
        outcome_from_decision(eol, fixed_interval_decisions(eol, t, lead), lead, costs).cost.mean() for t in candidates
    ]
    return float(candidates[int(np.argmin(means))])


def perfect_information_decisions(eol: np.ndarray, lead: int) -> np.ndarray:
    return np.asarray(eol, float) - (lead + 1)


def subset(fleet: Fleet, mask: np.ndarray) -> Fleet:
    idx = np.where(mask)[0]
    return Fleet(
        fleet.engines.iloc[idx].reset_index(drop=True),
        [fleet.cycles[i] for i in idx],
        [fleet.running_max[i] for i in idx],
    )


@dataclass
class PolicyResult:
    outcomes: dict[str, Outcome]  # policy -> per-engine outcomes, engines in `fleet.engines` order
    thresholds_by_fold: dict[int, float]
    intervals_by_fold: dict[int, dict[str, float]]


def cross_fitted(
    fleet: Fleet, lead: int, costs: Costs, grid_points: int, interval_range: tuple[int, int]
) -> PolicyResult:
    """Choose the threshold and the fixed intervals on the other folds' engines, evaluate on each fold's engines."""
    eng = fleet.engines
    n = len(eng)
    decisions = {k: np.full(n, np.inf) for k in ("model", "fixed_interval")}
    candidates = np.arange(interval_range[0], interval_range[1] + 1, dtype=float)
    thresholds: dict[int, float] = {}
    intervals: dict[int, dict[str, float]] = {}
    for k in sorted(eng["fold"].unique()):
        held = (eng["fold"] == k).to_numpy()
        train = subset(fleet, ~held)
        grid = threshold_grid(np.concatenate(train.running_max), grid_points)
        tau = float(grid[int(np.argmin(model_costs(train, grid, lead, costs)))])
        thresholds[int(k)] = tau
        decisions["model"][held] = subset(fleet, held).decision_cycles(np.array([tau]))[:, 0]
        intervals[int(k)] = {}
        for d in sorted(eng["dataset"].unique()):
            fit = (~held) & (eng["dataset"] == d).to_numpy()
            t = best_fixed_interval(eng.loc[fit, "eol"].to_numpy(), candidates, lead, costs)
            intervals[int(k)][d] = t
            target = held & (eng["dataset"] == d).to_numpy()
            decisions["fixed_interval"][target] = t - lead
    eol = eng["eol"].to_numpy()
    outcomes = {
        "model": outcome_from_decision(eol, decisions["model"], lead, costs),
        "fixed_interval": outcome_from_decision(eol, decisions["fixed_interval"], lead, costs),
        "run_to_failure": outcome_from_decision(eol, np.full(n, np.inf), lead, costs),
        "perfect_information": outcome_from_decision(eol, perfect_information_decisions(eol, lead), lead, costs),
    }
    return PolicyResult(outcomes, thresholds, intervals)


def final_threshold(fleet: Fleet, lead: int, costs: Costs, grid_points: int) -> tuple[float, pd.DataFrame]:
    """Threshold for serving, chosen on all training engines; returns it with the cost curve."""
    grid = threshold_grid(np.concatenate(fleet.running_max), grid_points)
    curve = model_costs(fleet, grid, lead, costs)
    return float(grid[int(np.argmin(curve))]), pd.DataFrame({"threshold": grid, "cost_per_engine": curve})
