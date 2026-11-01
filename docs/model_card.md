# Model card: `cmapss-maintenance`

| | |
|---|---|
| Registered model | `cmapss-maintenance` (MLflow; Unity Catalog name on Databricks: `<catalog>.cmapss.cmapss_maintenance`), alias `champion` = version 1 |
| Type | MLflow pyfunc bundle: LightGBM RUL regressor, two LightGBM quantile models with conformal correction, LightGBM classifier with isotonic calibration |
| Outputs | `rul_pred` (capped at 125 cycles), `rul_lower` / `rul_upper` (90% interval), `p_fail_within_n` (N = 30), `alert` |
| Decision rule | book a shop visit when `p_fail_within_n` ≥ 0.0094 (cost-based, `artifacts/policy.json`; D-024, D-025) |
| Training data | NASA C-MAPSS FD001–FD004 (simulation): 709 training engines, 160,359 cycles; gold Delta version and feature spec `1035ca0e0605` recorded with the run |
| Inputs | 161 point-in-time features: cycle, 16 per-condition-normalised sensors, rolling mean/std/slope over 5/15/30 cycles (`docs/methodology.md` §2) |
| Use | portfolio project; decision support for maintenance planning, not an autonomous removal decision |

## Intended use

Rank and flag engines for a shop-visit booking. A planner sees the remaining-life estimate, its interval and the
failure probability, and decides. The threshold encodes **assumed** costs (`business` in the config).

## Not intended for

- Real aircraft engines without validation on real operating and maintenance data. C-MAPSS is simulated, with one
  clean degradation path per engine and no repairs, sensor faults or censoring.
- Safety-critical decisions on its own, or use where costs and lead times differ from the stated assumptions without
  re-running the policy step.
- Engines in operating conditions outside the six seen in training. `op_condition_dist` shows how far a row is from
  the nearest condition.

## Performance (official NASA test engines, used once; `reports/model_comparison.md`)

| Metric | Value | Baseline |
|---|---|---|
| RMSE, last cycle vs NASA RUL (707 engines) | 22.7 cycles | ridge 27.0, mean 51.8 |
| RMSE, last cycle vs RUL capped at 125 | 12.5 cycles | ridge 17.2 |
| RMSE by dataset vs capped RUL | FD001 12.1, FD002 12.5, FD003 11.6, FD004 12.9 | |
| PHM08 score, last cycle (sum) | 12,864 | ridge 23,343 |
| PR-AUC, failure within 30 cycles (all test cycles, 2.5% positive) | 0.93 | logistic 0.86 |
| Interval coverage (nominal 90%), true RUL 0–30 | 0.90 at mean width 24 cycles | |
| Cross-validated RMSE (capped, grouped by engine, 5 folds) | 12.3 | the metric used for promotion (D-023) |

Cost policy on full training-engine lives, cross-fitted: 231 per engine against 306 for the best fixed interval and
201 for perfect information. That is about 24% less than fixed intervals under the assumed costs
(`reports/maintenance_policy.md`).

## Known weaknesses

- **Late bias near failure:** +6.4 cycles mean error for true RUL 31–60, the costly direction.
- **Interval coverage below nominal** for true RUL 31–60 (0.87) and 61–125 (0.83); coverage is guaranteed only within
  bands of *predicted* RUL (D-022).
- **Capped target:** healthy engines are predicted near 125 by design. The headline last-cycle RMSE against the NASA
  RUL partly measures the cap (FD002/FD004 have more test engines above it).
- **Age reliance:** long-lived engines that are old but far from failure are predicted conservatively (seen in
  monitoring, D-028).
- **Thresholds and base rates:** a fixed-precision threshold did not transfer from training to test (80% → 66%),
  which is why the served threshold is cost-based.
- **Horizon:** with N equal to the lead time, the decision sits on the lowest, coarsest probabilities; N = 45 is
  recommended (D-025).

## Ethical and operational considerations

No personal data is involved. The main risk is misplaced trust: the numbers come from a simulation and assumed
costs. Keep a human in the loop, show the interval with the point estimate, and re-run the policy step when costs
change.

## Lineage and reproducibility

Every training run records in MLflow the gold Delta version, the feature spec version, all metrics, the config and
a monitoring baseline. `make train` reproduces the run (fixed seeds, deterministic LightGBM).
