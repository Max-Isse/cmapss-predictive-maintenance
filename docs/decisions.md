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
