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
