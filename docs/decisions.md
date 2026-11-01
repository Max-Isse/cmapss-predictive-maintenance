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
**Sensitivity:** to be checked in phase 5 (models trained with caps 100, 125, 150).

## D-017 Failure-within-N label with N = the maintenance lead time (30 cycles)
**Choice:** `fail_within_n = 1` when `rul <= 30`, N read from `business.maintenance_lead_time_cycles`; labels are
added by `add_labels`, separately from the features. Test-engine RUL at every cycle = NASA RUL at the last cycle +
cycles to the end; `is_last_cycle` marks the official evaluation point.
**Why:** the decision the model supports is "schedule a shop visit now", which needs at least the lead time to act.
Base rates: 12.5–15% of training cycles, about 2–3% of test cycles, 20–25% of test engines at their last cycle.
