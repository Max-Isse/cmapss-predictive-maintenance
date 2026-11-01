# Decision log

Judgement calls with the alternatives considered.

## D-001 Data source and version pin
**Choice:** NASA PCoE data repository, data set 6 (Turbofan Engine Degradation Simulation, C-MAPSS), from the link
on the official repository page (`source` in `configs/config.yaml`). The archive's SHA-256 is pinned in config. A
download with a different checksum stops the pipeline instead of silently using changed data.
**Alternatives:** Kaggle mirrors (provenance unclear, sometimes re-formatted).

## D-002 Spark runs on Linux (project container / CI), not native Windows
**Choice:** every Spark step runs in `docker/Dockerfile.dev`: Python 3.11, OpenJDK 17, uv, the locked
dependencies, and Delta jars resolved at build time. CI runs the same tests natively on Ubuntu. Non-Spark code and
tests also run on Windows; Spark tests skip there with a pointer to the container.
**Why:** Spark 4 on Windows needs Hadoop's `winutils.exe`, which is only distributed as unofficial third-party
binaries. Linux is also what Databricks runs, so the container is the closest local stand-in.
**Alternatives:** WSL2 Ubuntu (equally valid; needs a JDK installed with sudo); unofficial winutils (rejected).

## D-003 Bronze keeps values as delivered (strings) plus the raw line for text files
**Choice:** bronze stores the 26 values as strings exactly as delivered, and keeps the raw text line for text
files. Typing and validation happen in silver. A text line with the wrong number of tokens is kept with its raw
line and null values, so silver can quarantine it with a reason instead of the loader failing or dropping it.
**Alternatives:** typed bronze (a malformed value would fail the load or be silently cast to null).

## D-004 Idempotency: insert-only MERGE on record identity, plus a file-level skip
**Choice:**
- *Record identity:* each file is deduplicated and MERGEd into bronze, inserting only records whose
  `(dataset, split, unit, cycle, _record_hash)` is not already present. `_record_hash` is xxhash64 of the values
  cast to double, so it does not depend on formatting: the same record delivered as text and as Parquet has the
  same hash.
- *File-level skip:* a file whose checksum is already logged as loaded is skipped. `--force` bypasses the skip;
  the MERGE still prevents duplicates.
- *Changed values:* a changed value for the same engine-cycle is a new record. Bronze is append-only, so both
  versions are kept, and silver decides which is valid.

**Tested:** re-running a file inserts 0 rows; a Parquet delivery of the same records inserts 0; a corrected value
inserts exactly one new row (`tests/test_ingest.py`). On the full data, a forced re-run of all 24 files inserted
0 of 531,926 rows read (`docs/data_lineage.md`).
**Implementation note:** an insert-only MERGE that inserts nothing does not commit a new Delta version, so the
inserted-row count is read from the table history only if the MERGE created a new version. The first version of
the loader read the previous commit's metrics and over-reported; the tests caught it.
**Alternatives:** overwrite partitions (not append-only, loses history); key on (unit, cycle) only (would silently
drop corrections).

## D-005 Two delivery formats in the landing zone
**Choice:** landing holds the NASA text files and Parquet conversions of them (`landing/text`, `landing/parquet`).
The bronze loader reads both. Parquet is produced from the same parsed text, so its values are identical.
**Why:** the brief asks the pipeline to handle multiple formats. Delivering the same records in two formats also
tests cross-format idempotency (D-004).

## D-006 Silver rules: fixed order, first failure is the recorded reason
**Choice:** six row rules run in a fixed order: malformed_row, non_numeric, invalid_key, out_of_range,
superseded_duplicate, sequence_gap (RUL rows: non_numeric, invalid_value, superseded_duplicate,
rul_without_test_engine). A failing row goes to `silver/quarantine` once, under its first failing rule, with a
human-readable reason (for example `sensor_03=99999.0 not in [1205.246, 1654.334]`). Reconciliation is checked on
every run: bronze rows = silver rows + quarantined rows.
**Alternatives:** record every failing rule per row (more detail, but each row should have one actionable cause);
drop bad rows silently (rejected: no audit trail).

## D-007 Sensor range contract fixed in config from the training files
**Choice:** each setting and sensor has an inclusive range in `validation.ranges`. The range is the training
minimum and maximum widened by 10% of the range on each side, computed once and written into config.
**Why:** a range learned on every run would accept whatever arrives (circular). A fixed contract catches unit
changes and sensor faults in replayed data. All test values fall inside the unwidened training ranges, so the
contract does not reject legitimate test data.
**Alternatives:** physical limits from engine documentation (not available for the simulation); per-operating-
condition ranges (tighter, but they need the condition clustering of phase 3).

## D-008 Silver is rebuilt in full from bronze on each run
**Choice:** silver tables and the quarantine are overwritten on each run from the complete bronze history.
**Why:** deterministic and idempotent: the same bronze gives the same silver (tested). At about 265k rows a full
rebuild takes under a minute. Phase 4's replay will revisit this for incremental processing.

## D-009 Pandera contract as an independent gate on silver
**Choice:** `quality/contract.py` builds a pandera (PySpark backend) schema from config: types, nullability,
dataset and split domains, positive unit and cycle, the same value ranges. It validates the finished silver
tables, and a violation fails the run after the report is written.
**Why:** the row rules decide what is quarantined; the contract checks the result independently, so a bug in a
rule cannot pass bad data downstream silently.
**Compatibility notes:** pandera's PySpark backend imports `pyspark.pandas`, which in Spark 4.0 needs pandas < 3,
so pandas is pinned. Its `.pandera` accessor is registered only when `pandera.pyspark` is imported.

## D-010 Redelivered records: the latest delivery wins
**Choice:** when the same dataset/split/unit/cycle arrives with different values, silver keeps the latest by
bronze ingest time and quarantines the older versions as `superseded_duplicate`, naming the file that replaced
them.
**Why:** a redelivery is how a source corrects data. Bronze keeps every version (D-004), so the choice can be
revisited.
**Alternative:** quarantine all versions until resolved by hand (safer for safety-critical feeds; noted as an option
for the monitoring phase).

## D-011 Sequence rule: engine cycles must run 1, 2, 3, … ; keep the prefix before a gap
**Choice:** "strictly increasing" is enforced as follows: after deduplication, each engine's cycles, sorted, must
be exactly 1..n. Rows from the first missing cycle onwards are quarantined; the clean prefix is kept.
**Why:** rolling-window features (phase 3) assume consecutive cycles. The prefix is still valid history.
**Alternative:** quarantine the whole engine (loses valid data). Ordering within a delivered file is not checked:
bronze does not store line positions.

## D-012 Bronze hashing must not fail on bad input
**Choice:** the record hash uses `try_cast`. Rows with non-numeric text are hashed on their delivered strings, and
malformed lines on their raw text.
**Why:** Spark 4 runs in ANSI mode, so a plain cast of `abc` to double raises and aborted the whole bronze load
(found by the silver tests). Bronze must accept anything that arrives so silver can quarantine it. Numeric rows
hash exactly as before: a forced re-run of all 24 files inserted 0 rows.

## D-013 One feature spec, fitted on training engines only, shared by training and serving
**Choice:** every constant the features need (operating-condition centroids, kept sensors, per-condition mean and
standard deviation, windows, RUL cap, horizon) is fitted on the training engines of all four datasets pooled and
saved as `artifacts/feature_spec.json` (copy: `reports/tables/feature_spec.json`), with a content-hash version
stamped on every gold row. `features/build.py::build_features` is the only feature code: the gold build, the
replay and the API call it with that spec.
**Why:** test and replayed engines never influence a constant they are scored with, and serving cannot drift
from training because there is no second implementation. Pooling is possible because all four datasets simulate
the same engine model and share the sea-level condition; per-condition statistics already absorb the conditions.
**Cost:** serving needs a local Spark session (a JVM in the API container); acceptable for a batch-scoring
portfolio service, revisited in phase 7 if latency matters.
**Near-constant rule:** a sensor is dropped when it has at most 2 distinct values within every operating condition
(`features.constant_max_distinct`). That drops sensors 01, 05, 18, 19 (one value per condition) and 16 (two
values, a quantisation step); sensor 10 (3–6 values, weak signal) and 06 are kept and left to the models. Counting
within conditions matters: across the fleet every sensor varies because the conditions shift them.

## D-014 Operating conditions by k-means (k = 6) on standardised settings, best of 10 seeds
**Choice:** Spark ML k-means on the three settings standardised by their training mean and standard deviation;
the lowest-cost of 10 seeded runs; centroids stored in original units, sorted, so condition ids are stable.
Each row gets the nearest centroid and its distance (`op_condition_dist`, kept for monitoring).
**Why:** the readme states six conditions; the data confirm them: every point of every split lies within 0.005
standardised units of a centroid. Rounding the settings instead was fragile (signed zeros split the sea-level
condition into three keys). A single k-means run merged two conditions in the test data, hence the restarts.
**Caveat:** standardising amplifies a setting that is constant up to jitter; on pooled training data all three
settings vary, and k = 1 (a single-condition fit) skips k-means.

## D-015 Per-condition normalisation, then rolling windows on the normalised values
**Choice:** z = (value − condition mean) / condition standard deviation for each kept sensor; rolling mean and
standard deviation (DataFrame API) and least-squares slope on cycle (Spark SQL `regr_slope`) over 5, 15 and 30
cycles, framed `RANGE BETWEEN w−1 PRECEDING AND CURRENT ROW` over the cycle number within each engine.
**Why:** in FD002 and FD004 the raw correlation of every sensor with RUL is about 0 (the condition dominates);
after normalisation it is up to −0.69, as in FD001 (`reports/features.md` §3). Rolling over raw values would mix
conditions. A RANGE frame over cycle means "the last w cycles", which stays correct even if a gap ever passed
silver. Windows shorter than w at the start of life use the cycles available; std and slope are null on the first
cycle (left for the models, which handle missing values).

## D-016 RUL target capped at 125 cycles (piecewise linear)
**Choice:** `rul_capped = min(rul, 125)` (`features.rul_cap`); the uncapped `rul` is kept for evaluation.
**Why:** a health index built from five sensors, relative to each engine's own early life, is flat within noise
above RUL ≈ 170 and rises over the last 100–150 cycles. The least-squares hinge optimum is 95 (biased low because
wear accelerates); 125 sits between the two, costs 7.6% more hinge SSE than the optimum, is below the 5% quantile
of training engine life (149), and is the usual C-MAPSS choice, keeping RMSE comparable with published work.
**Sensitivity (phase 5, `reports/model_comparison.md` §4):** a cap of 100 is more accurate where true RUL is at most
100 (out-of-fold RMSE 10.8 vs 14.0 at 125) and on test engines with true RUL at most 125 (12.1 vs 12.6), but it is
much worse on all test engines (RMSE 29.6 vs 22.7, NASA score 38,867 vs 12,864), because every healthy engine is
predicted at 100. A cap of 150 is the reverse (20.9 overall, 16.3 for true RUL at most 125). 125 is kept: the
maintenance decision uses P(fail within 30) from the classifier (phase 6), so the cap shapes only the RUL output,
where 125 balances the two regions.

## D-017 Failure-within-N label with N = the maintenance lead time (30 cycles)
**Choice:** `fail_within_n = 1` when `rul <= 30`, N read from `business.maintenance_lead_time_cycles`; labels are
added by `add_labels`, separately from the features. Test-engine RUL at every cycle = NASA RUL at the last cycle +
cycles to the end; `is_last_cycle` marks the official evaluation point.
**Why:** the decision the model supports is "schedule a shop visit now", which needs at least the lead time to act.
Base rates: 12.5–15% of training cycles, about 2–3% of test cycles, 20–25% of test engines at their last cycle.

## D-018 SYNTHETIC REPLAY: real test-engine rows, simulated arrival
**Choice:** `cmapss replay --batches N` re-delivers the NASA test engines into `landing/replay/` as
`replay_<dataset>_SYNTHETIC_b<batch>.txt`. Each line is the original test-file line, byte for byte; batch b
carries cycles (b−1)·25+1 … b·25 of every test engine (`replay.cycles_per_batch`), all engines start together,
and an engine stops after its last test cycle. Five rows per batch (`replay.late_rows`, seeded) are held back and
delivered with the next batch, as a deliberate late-arrival disturbance. Progress lives in
`landing/replay/_state.json`; bronze stores the rows under split `replay`, so the SYNTHETIC label travels with
every row in `_source_file`.
**Why:** C-MAPSS has no live feed. Replaying the official test engines exercises the incremental pipeline,
serving and monitoring on real sensor values, and their ground truth (NASA RUL) is known for evaluation.
**Labels:** a replayed engine's RUL comes from its test engine's end of life (last test cycle + NASA RUL), known
before the replay ends. A live fleet would learn this only after failure; labels are used for evaluation and
monitoring only, never as features.
**Not simulated:** staggered engine start times, sensor faults, schema changes. Duplicates and out-of-range
values are covered by the silver tests (D-010, D-007).

## D-019 Incremental processing: a Structured Streaming query over bronze, engine-scoped recompute
**Choice:** `cmapss stream` (and each replay batch) runs a Structured Streaming query on the bronze Delta table
with `trigger(availableNow=True)` and a checkpoint under `lakehouse/_checkpoints/`. Each micro-batch lists the
engines it touches and recomputes silver, quarantine and gold for those engines from their **full bronze
history**, with the same functions as the full rebuild (`silver_frames`, `build_features`, `add_labels`),
then swaps them in with Delta `replaceWhere`, one atomic commit per table. The pandera contract gates the
recomputed silver rows before anything is written. The first run starts right after the bronze version that the
last full `cmapss silver` read (pinned with `versionAsOf` and recorded in `lakehouse/_state/`).
**Why:** every rule and feature is local to one engine, so recomputing touched engines equals a full rebuild for
them; this is checked after every replay (`verify_incremental`: silver, quarantine and gold compared row for row
with a full recomputation). A late row re-opens its engine, so the `sequence_gap` quarantine of the rows after it
is replaced by valid rows one batch later (D-011 keeps the prefix; this restores the rest). Appending only new
rows to gold would be cheaper but would freeze such errors in place.
**Cost, as measured (full replay, 20 batches):** the data work per batch is small (one engine's history is at
most a few hundred rows), but the wall time is not proportional to it. Each batch took 84–163 s on a 4-core local
Spark: bronze ingestion of the 2–4 batch files 33–77 s, the stream step 45–93 s, and both grew over the replay while
the engines recomputed fell from 707 to 2. The time is per-batch overhead (a new streaming query, Delta log and
file listing, contract checks, rewriting the gold files that hold the touched engines) that grows with table
history and small files. Not tuned in this phase; candidates: periodic `OPTIMIZE` compaction or auto-compaction,
deletion vectors so `replaceWhere` marks rows instead of rewriting files, and fewer, larger triggers.
**Alternatives:** stream silver → gold with Delta change data feed (two queries; more state to keep consistent);
`foreachBatch` MERGE of new rows only (cannot repair late arrivals). On Databricks, the same code runs as a job with
an Auto Loader source in place of the file-level bronze loader (phase 8 notes).
**Limits:** new RUL deliveries are not streamed; `cmapss silver` rebuilds the RUL table. The feature spec is
read, never refitted, by the stream: refitting is a deliberate `cmapss gold` run.

## D-020 Evaluation protocol: grouped CV on training engines, official test engines as a one-time holdout
**Choice:** all fitting and every choice (calibrator, alert threshold, conformal correction, cap comparison,
champion promotion) uses 5-fold GroupKFold by engine on the 709 training engines. Final models are refitted on all
training engines and score the 707 official test engines once, for the report. The official C-MAPSS figure is each
test engine's last cycle against the NASA RUL; all test cycles are also reported (more rows, all RUL bands).
**Why:** rows of one engine are strongly autocorrelated, so a row-level split would put near-copies of validation
rows in training and flatter every metric. Choosing anything on the test set would make it a validation set.
**Leakage tests:** folds never share an engine and cover every row once; model inputs exclude labels and
identifiers (`tests/test_models.py`); feature constants come from training engines only (D-013); features are
point-in-time (D-015).

## D-021 Models: baselines, then LightGBM with fixed settings; no hyperparameter search
**Choice:** RUL: mean of the capped target, ridge regression on the same 161 features ("linear degradation"
baseline), LightGBM. Failure within N: prevalence, logistic regression, LightGBM, and LightGBM with an isotonic
calibrator fitted on out-of-fold probabilities. LightGBM settings are fixed in config (500 trees, learning rate
0.03, 31 leaves, at least 200 rows per leaf, row and column subsampling): conservative values for ~160k
autocorrelated rows.
**Why:** the gain over baselines is the evidence that the model learns; a search over hyperparameters would add
little information on a benchmark this size and costs a nested CV to report honestly. XGBoost would be a like-for-
like alternative to LightGBM and is not run.
**Alert threshold:** the lowest probability that reaches 80% precision on out-of-fold predictions
(`models.fixed_precision`, an ASSUMPTION). Phase 6 replaces it with a cost-based threshold.

## D-022 Prediction intervals: conformalized quantile regression, corrected per band of predicted RUL
**Choice:** LightGBM quantile models (5% and 95%) of the **uncapped** RUL, clipped to the cap, give a per-row
interval for the capped RUL. A conformal correction q, the ⌈(n+1)(1−α)⌉/n quantile of max(lower − y, y − upper)
over out-of-fold rows, is computed separately for each band of *predicted* RUL (≤30, 30–60, 60–90, 90–115, >115;
`models.interval_group_bounds`; Mondrian conformal), for 90% nominal coverage within each band. The interval is
clipped to [0, cap] and always contains the point prediction. Bands with fewer than 20/α rows use the pooled q.
**What went wrong first (kept as a lesson):** with quantile models of the *capped* RUL, the 95% model predicted the
constant 125 for every row: about 40% of training rows sit exactly at the cap, so the upper quantile of the target is
the cap, the quantile loss has no gradient to split on, and the model never leaves its initial value. Intervals
near failure were then 108 cycles wide, and a single pooled correction was pinned at 0 by the many rows exactly at
the cap, so coverage fell to 0.84–0.87 in the bands that matter. Training on the uncapped RUL and correcting per
predicted band fixed both: in the 0–30 true-RUL band the test coverage is 0.90 at a mean width of 24 cycles.
**Caveats:** conformal guarantees are marginal within each *predicted* band and assume exchangeable rows; rows of one
engine are not independent. Coverage by *true*-RUL band is not guaranteed and is reported: on the test engines it is
below nominal at 31–60 and 61–125 (`reports/model_comparison.md`). In the two upper bands q is 0: most of their rows
are at the cap and score exactly 0, so the interval is the quantile models' own (over-covering above 115).

## D-023 MLflow: SQLite tracking, one pyfunc bundle, "champion" alias moved only on improvement
**Choice:** tracking in `mlruns/mlflow.db` (git-ignored) with artifacts under `mlruns/artifacts`. Each run logs the
LightGBM settings, the gold Delta version and feature spec version (tags), CV and test metrics, the config, and the
report. One registered model, `cmapss-maintenance`, holds a pyfunc that returns RUL, its interval, the calibrated
failure probability and the alert. Alias `champion` moves to a new version only when `cv_rmse_capped` (a CV metric,
never a test metric) improves.
**Why:** serving loads `models:/cmapss-maintenance@champion` and gets every output from one versioned artifact,
traceable to the exact data version. Phase 7 reuses the same promotion rule for challenger runs.

## D-024 Policy evaluation: simulate full engine lives with out-of-fold probabilities, cross-fitted choices
**Choice:** the decision rule is "book a shop visit at the first cycle where the calibrated P(fail within N) reaches
the threshold"; the visit happens after the lead time L. It is in time if the engine still had more than L cycles
left at the decision, and the cycles it could still have flown (RUL − L) count as lost life; otherwise the engine
fails in service and costs `unplanned_failure_cost` (no planned visit is charged in that case). Policies:
model rule; fixed interval (remove at cycle T, chosen per dataset, a deliberately strong baseline); run to failure;
perfect information (decide at RUL = L + 1, the lower bound).
**Data:** the 709 training engines, because they run to failure and every policy's outcome is known. Their signal
is the out-of-fold probability of a classifier with the champion's settings, calibrated by an isotonic fit on the
other folds. The threshold and the fixed intervals are chosen on four folds and applied to the fifth, so the
reported costs are not scored on the engines that chose them. The official test engines stop before failure; they
give a partial check (decisions inside the observed window, certain failures, censored engines).
**Why:** a threshold chosen by expected cost uses the asymmetry of the costs directly, unlike the fixed-precision
alert of phase 5, whose precision did not transfer to the test fleet (D-021).
**Caveats:** costs are assumptions (no cost data come with C-MAPSS); one engine is one life, with no repair,
partial maintenance or shop capacity limits; the simulated fleet's failure physics is cleaner than a real fleet's.

## D-025 Served threshold and its storage; sensitivity scenarios
**Choice:** the served threshold is chosen on all training engines with the champion's own calibrator and stored
in `artifacts/policy.json` (copy in `reports/tables/policy.json`) and as tag `policy_threshold` on the champion
model version; the API (phase 7) reads it. Sensitivity re-chooses thresholds and intervals (cross-fitted) for each
scenario: one cost assumption at a time, the label horizon N with L fixed (classifier retrained for each N), and
the lead time L with N = L.
**Why:** a threshold is only meaningful together with the model version and costs it was chosen for, so it lives
next to both; the scenarios show which conclusions survive other cost assumptions.
**Results (2026-11-01, `reports/maintenance_policy.md`):** served threshold 0.0094 on the calibrated P(fail within
30); cross-fitted cost per engine 231 for the model rule, 306 for the best fixed interval, 1,000 for run to failure
and 201 for perfect information: a saving of about 75 per engine (24%) against fixed intervals, 71% of what perfect
information would save. Estimates under the assumed costs, on simulated engines.
**Finding on N:** with N equal to the lead time, the rule must fire while the engine is still outside the positive
class, so the chosen threshold sits on the lowest steps of the calibrated probability (0.008–0.012 across folds)
and the cost curve is flat from 0.001 to 0.03: the classifier's resolution in that region is coarse. N = 45 with
L = 30 moves the decision to p ≈ 0.5 and lowers the cross-fitted cost to 223 (−3.7%); N = 15 < L is worse than
fixed intervals (395), because a 15-cycle horizon carries no signal when a 30-cycle lead time requires deciding.
**Recommendation (not yet applied):** train the served classifier with a horizon N > L (for example L + 15) and
re-choose the threshold; the gold label `fail_within_n` stays tied to the configured lead time for evaluation.


## D-026 Serving scores gold features; the API image has no Spark or Java (revises D-013's serving note)
**Choice:** the API does not recompute features. A request names an engine, and the API reads that engine's latest
(or a given) gold row with delta-rs (`deltalake`), or the client posts a full feature row. Gold rows come only from
`features/build.py`, the single feature implementation, which the stream keeps current (D-019). Spark, Delta-Spark,
pandera and the report libraries moved to the `pipeline` extra; the API image installs the core dependencies only.
**Why:** one feature implementation still serves training and serving, with no skew, while the API needs no JVM:
the image is smaller, starts in seconds and scores a row in milliseconds. D-013 had accepted a local Spark session in
the API; with the stream maintaining gold, that is unnecessary.
**Limit:** an engine's prediction is as fresh as the last stream run; scoring a history that never went through the
pipeline requires the client to post the features. Measured in compose: a lookup prediction takes about 250 ms,
mostly the delta-rs scan of the dataset partition; caching recent rows would cut it if latency mattered.
**Found in the end-to-end run:** MLflow writes a metadata file next to a model it loads, which failed on the
read-only artifact mount; the API now loads into a private temporary copy.

## D-027 API conventions: request IDs, JSON logs, prediction log, served threshold
**Choice:** every request gets an ID (the `X-Request-ID` header, or a generated one returned in the response) and
one JSON log line (method, path, status, latency, request ID). Every prediction is appended to the Delta table
`lakehouse/serving/prediction_log` with request ID, model version, threshold, engine key, outputs and decision;
posted feature rows are stored as JSON (looked-up rows are in gold by key). The decision uses the cost-based
threshold from `artifacts/policy.json` when it was chosen for the loaded model version, and otherwise the model's
fixed-precision threshold; `/model-info` says which. The image runs as a non-root user with a health check;
compose starts it after the MLflow server is healthy, and `/health` returns 503 when no model is loaded.
**Why:** a prediction must be traceable to its request, model version and threshold, and the log is the input of
monitoring. A model version and its threshold stay paired, so a newly promoted model never runs with another
model's threshold.
**Ownership (found in the end-to-end run):** the batch scorer runs as root in the pipeline container, so a table it
created was not writable by the API's non-root user. Each writer now has its own table: `serving/batch_predictions`
(`cmapss score`) and `serving/prediction_log` (API, on a named volume the image creates owned by that user);
monitoring reads both.
**Limit:** one append per request makes many small Delta files; a scheduled `OPTIMIZE` would compact them.

## D-028 Monitoring: PSI drift per window, performance by true-RUL band against an out-of-fold baseline
**Choice:** at training, a baseline is stored with the MLflow run: training deciles of every feature, out-of-fold
deciles of both outputs, and out-of-fold RMSE and interval coverage per true-RUL band. `cmapss score` writes the
champion's predictions for the replay into the prediction log; `cmapss monitor` joins them with their labels and,
per replay window, computes PSI per feature and output, and RMSE and coverage per band. Performance is degraded
when a band with at least 500 rows has RMSE above 1.25 × baseline or coverage 0.10 below nominal; drift is
alerted when at least 30% of features have PSI above 0.25.
**Why out-of-fold:** the replayed engines are the official test engines, so a test-set baseline would compare the
data with itself. **Why bands:** an ageing fleet has more rows near failure, where errors differ; comparing within
bands keeps a change of mix from reading as a worse model.
**Labels:** for the replay they are known in advance (NASA RUL); a live fleet learns them only after failure, so
performance monitoring always lags.
**Scope of the performance judgement (changed after the first run):** the first run judged each window and fired:
8 windows had a band beyond tolerance, mostly the >125 band of windows 4–9 (RMSE up to 1.75 × baseline). Those rows
are engines already 76–225 cycles old yet far from failure, which the model, using age, predicts conservatively;
because the replay starts every engine together, each window is one age group. Over the whole replay every band is
within tolerance (RMSE 0.95–1.14 × baseline). The retraining signal is therefore judged over the whole period
(`monitoring.performance_scope: period`); per-window cells stay in the report as diagnostics. Drift fires in most
windows (fleet ageing) and only warns.

## D-029 Retraining trigger and challenger vs champion
**Choice:** `monitoring.retrain_on` lists the signals that trigger retraining; by default performance only. Drift
alone warns, because an ageing fleet shifts every input without the model being wrong. `cmapss monitor --retrain`
trains a challenger when triggered (`--force-retrain` regardless) with the same training pipeline on the current
gold version; it is registered as a new version, and the champion alias moves only if the challenger improves
`cv_rmse_capped` (D-023). The API picks up a new champion on restart.
**Why:** retraining is an ordinary, logged, reproducible run; the promotion rule cannot be gamed by the data that
triggered it, because it uses a cross-validated metric on training engines.
**Limit:** in this benchmark no new run-to-failure engines arrive, so a challenger trains on the same data as the
champion and is not promoted; with a real fleet, matured engines would join the training set.
**Exercised (2026-11-01):** the per-window run above triggered a retrain; challenger version 2 reached the same
`cv_rmse_capped` (12.306) and the alias stayed on version 1, as the rule requires.
