# Recommendations

For maintenance planning and engineering leads. **Every figure is an estimate on NASA's simulated engines under
assumed costs** (in-service failure 1,000, planned visit 200, 1 per cycle of life given up, lead time 30 cycles;
`configs/config.yaml`). The evidence is in the linked reports; the judgement calls are in `docs/decisions.md`.

## 1. Book shop visits from the cost-based failure probability, not from fixed intervals

- **What:** book a visit when the calibrated probability of failure within 30 cycles reaches 0.0094; the visit
  follows the 30-cycle lead time.
- **Why:** on 709 run-to-failure engines (cross-fitted), the expected cost is 231 per engine, against 306 for the best
  fixed interval and 1,000 for run to failure. That is about **24% less than fixed intervals** and 71% of what perfect
  foresight would save. Most of the gain comes from giving up far less useful life (24 cycles per visit instead of 95),
  with slightly fewer in-service failures (7 vs 11 of 709). Source: `reports/maintenance_policy.md`.
- **Robustness:** the rule beats the best baseline under every cost scenario tested. Its edge shrinks to 13% when a
  failure costs barely more than a planned visit (250 vs 200).
- **Condition:** re-run `cmapss policy` whenever the cost or lead-time assumptions change. The threshold belongs to a
  model version and a set of costs (D-025).

## 2. Train the served classifier for a horizon longer than the lead time

- **What:** retrain the failure classifier for "fails within 45 cycles" (lead time + 15), re-choose its threshold, and
  promote it if the policy cost improves.
- **Why:** with the horizon equal to the 30-cycle lead time, the decision must fire while the probability is still tiny,
  where calibration is coarsest. N = 45 moves the decision to p ≈ 0.5 and lowers the cross-fitted cost from 231 to
  223 per engine (−3.7%). A horizon shorter than the lead time is worse than fixed intervals (395).
- **Caveat:** the gain is small, and its variation across folds has not been measured; treat it as a candidate for a
  challenger run.

## 3. Use the RUL output for planning, with its interval

- Remaining life is predicted with an RMSE of 12.5 cycles against RUL capped at 125 (official test engines), and it is
  about equally good on all four datasets (11.6–12.9). Show the 90% interval alongside it: near failure it is
  24 cycles wide and covers 90% of outcomes.
- Expect late predictions in the 31–60 cycle band (+6.4 cycles on average), and treat any RUL above about 100 as
  "healthy": the target is capped at 125 by design.
- Do not compare the headline RMSE against the NASA RUL (22.7) across datasets. It mostly reflects how many engines are
  above the cap.

## 4. Operate with the monitoring plan; retrain on performance, not on drift alone

- Input drift will fire whenever the fleet's age mix changes; on the replay it fired in most windows while performance
  stayed within tolerance (RMSE 0.95–1.14× baseline in every true-RUL band). Retrain when band performance
  degrades over a period; treat drift as a prompt to look. See `docs/monitoring_plan.md`.
- Add engines that have failed or been removed to the training data at each retrain; the promotion rule (better
  cross-validated RMSE) keeps a worse challenger out.

## 5. Before any real-world use

1. Validate on real operating and maintenance records (repairs, sensor faults, censored engines), not on C-MAPSS.
2. Replace the cost assumptions with actual figures for unplanned removals, shop visits and lost life, then re-run the
   policy step.
3. Deploy the Databricks bundle to a dev workspace and work through `databricks/README.md` (managed Unity Catalog
   tables, Auto Loader, serving choice).
4. Schedule compaction (`OPTIMIZE`) for the incrementally written tables: time per replay batch rose over the replay,
   within 84–163 s (D-019).

## What this project does not show

- Savings on a real fleet, or in currency: the costs are assumptions in arbitrary units.
- Performance beyond the six operating conditions and two fault modes simulated in C-MAPSS.
- The effect of shop capacity, spare-engine availability or fleet scheduling on the policy (each engine is treated
  independently).
