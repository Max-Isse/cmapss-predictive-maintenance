# Limitations

## Data

- C-MAPSS is a NASA simulation: one clean degradation path per engine, injected noise, no repairs, no sensor faults
  and no censoring of training engines. Nothing here transfers to a real fleet without validation on real data.
- Costs and the lead time are assumptions (`business`); every saving is an estimate under them.
- The SYNTHETIC REPLAY re-delivers the official test engines with simulated timing (D-018). Its labels are known in
  advance, which a live fleet does not have.

## Modelling

- The RUL target is capped at 125 (D-016), so predictions above the cap are impossible by design.
- Intervals cover 90% within bands of predicted RUL, not of true RUL; coverage is 0.83–0.87 at true RUL 31–125 (D-022).
- No hyperparameter search (D-021), and XGBoost is not compared.
- Two fault modes in FD003/FD004 move some sensors in opposite directions, and one model serves both.

## Decision layer

- One engine is one life: no partial repairs, shop capacity limits or fleet scheduling constraints.
- The holdout check on test engines is partial: 458 of 707 are censored (D-024).
- The served classifier uses N equal to the lead time; N = 45 would lower the cost by about 3.7% (recommended, not
  applied).

## Engineering

- Per-batch incremental processing time grows with table history (84–163 s per batch locally; D-019), and compaction
  is not scheduled.
- The API reads gold with delta-rs on every request (about 250 ms) and picks up a new champion only on restart.
- Everything runs locally: the Databricks bundle is not deployed or validated (D-030), and Delta access is
  path-based.
- Monitoring windows in the replay are age groups, so per-window performance is diagnostic only (D-028).
