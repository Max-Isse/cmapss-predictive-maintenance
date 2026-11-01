# Monitoring plan

What is watched, how often, against what, and what happens next. Implementation: `src/cmapss_pm/monitoring/`;
report: `reports/monitoring.md`; decisions D-027 to D-029.

| What | How | Cadence | Threshold | Action |
|---|---|---|---|---|
| Data quality | silver rules, quarantine, pandera contract, per-run report | every pipeline run | any contract failure; quarantine growth | a contract failure stops the run; investigate the quarantine by rule |
| Pipeline health | replay/stream batch log: rows, seconds, quarantine | every batch | rows ≠ delivered; rising time per batch | check the files; compact the tables (D-019) |
| Input drift | PSI of the 161 features against training deciles | daily / per window | ≥ 30% of features with PSI > 0.25 | warning: check whether the fleet mix changed (ageing, new conditions) |
| Output drift | PSI of `rul_pred` and `p_fail_within_n` | daily / per window | PSI > 0.25 | warning; review together with input drift |
| Operating envelope | `op_condition_dist` in gold | every batch | far above the training maximum (0.005) | flag the rows: an operating condition outside the training set |
| Performance | RMSE and interval coverage per true-RUL band against the out-of-fold baseline, as labels mature | weekly / when labels arrive | over the period: RMSE > 1.25 × baseline, or coverage < nominal − 0.10 | **retrain** (challenger) |
| Decision quality | realised cost per engine and failures caught, as engines are removed or fail | monthly | cost materially above the policy estimate (231 per engine) | re-run the policy step; review the cost assumptions |
| Serving | JSON request logs (status, latency), `/health` | continuous | 5xx rate, latency, health 503 | alert the owner; restart; check the model store |

**Retraining:** `cmapss monitor --retrain` trains a challenger on the current gold version. The champion alias moves
only if the cross-validated RMSE improves (D-023, D-029). With a real fleet, engines that have failed or been removed
since the last training run join the training set, because their full lives are then known.

**Labels lag:** RUL is known only after failure or removal, and failure-within-30 labels mature 30 cycles after the
prediction. Performance monitoring therefore lags drift monitoring by design.
