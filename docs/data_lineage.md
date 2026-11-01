# Data lineage

Row counts per layer are written by each run (phase 2 adds the per-run data-quality report); this page describes
the flow.

```
NASA PCoE archive (SHA-256 pinned)
  └─ data/raw/cmapss/{train,test,RUL}_FD00x.txt               cmapss fetch (extract, manifest)
       ├─ data/landing/text/*.txt                              copied as delivered
       └─ data/landing/parquet/*.parquet                       parsed to typed columns (same values)
            └─ bronze.cycles  (append-only, strings + metadata) cmapss bronze: insert-only MERGE on
               bronze.rul                                       (dataset, split, unit[, cycle], _record_hash)
               bronze.ingest_log (one row per file per run)
                 └─ silver.* (typed, validated, deduplicated) + silver.quarantine          phase 2
                      └─ gold.features (engine × cycle) + labels                          phase 3
data/landing/replay/  ← SYNTHETIC REPLAY of test engines, cycle by cycle (real values, simulated arrival)  phase 4
```

| Step | Code | Rule | Decision |
|---|---|---|---|
| Download | `ingest/fetch.py` | Official NASA URL; checksum must match the pinned value | D-001 |
| Landing | `ingest/convert.py` | Text copied; Parquet parsed from the same text | D-005 |
| Bronze | `ingest/bronze.py` | Values as delivered + raw line; metadata; insert-only MERGE; file-level skip | D-003, D-004 |

## Silver (`cmapss silver`; report: `reports/data_quality/latest.md`)

- **Result:** all 265,256 bronze cycles and 707 RUL rows pass the six row rules and the pandera contract, and
  none are quarantined. Reconciliation (bronze = silver + quarantine) holds.
- **What that confirms:** the NASA benchmark is clean. There are no gaps in any engine's cycles, no conflicting
  duplicates and no out-of-range values. The rules exist for the replayed and future data, and each one is
  demonstrated on deliberately broken deliveries in `tests/test_silver.py`.

## Bronze row counts (`reports/tables/bronze_counts.csv`, written by `cmapss bronze`)

| Table | Dataset | Split | Rows | Engines | Malformed |
|---|---|---|---|---|---|
| cycles | FD001 | train / test | 20,631 / 13,096 | 100 / 100 | 0 |
| cycles | FD002 | train / test | 53,759 / 33,991 | 260 / 259 | 0 |
| cycles | FD003 | train / test | 24,720 / 16,596 | 100 / 100 | 0 |
| cycles | FD004 | train / test | 61,249 / 41,214 | 249 / 248 | 0 |
| rul | FD001–FD004 | — | 100 / 259 / 100 / 248 | same | 0 |

Every count equals the row count of the NASA source file in `data/raw_manifest.json`.

**Idempotency on the full data (bronze ingest log, 2026-11-01):**
- *First load:* 24 files (12 text + 12 Parquet), 531,926 rows read, 265,963 inserted. All text rows were
  inserted; all 265,963 Parquet rows matched existing records and were not inserted.
- *Re-run:* all 24 files were skipped by checksum.
- *`--force` re-run:* all 24 files went through the MERGE again, and 0 rows were inserted.

## Gold (`cmapss gold`; evidence: `reports/features.md`)

| Step | Code | Rule | Decision |
|---|---|---|---|
| Feature spec | `features/spec.py` | Fitted on silver training engines only; saved with a content-hash version | D-013, D-014 |
| Features | `features/build.py` | Condition, per-condition z-scores, rolling mean/std/slope over 5/15/30 cycles | D-015 |
| Labels | `features/build.py::add_labels` | RUL, RUL capped at 125, failure within 30 cycles, last-cycle flag | D-016, D-017 |
| Table | `features/gold.py` | `gold/features`: rebuilt in full, partitioned by dataset, `_spec_version` and `_gold_run_id` on every row | D-008 |

- **Result (2026-11-01):** 265,256 silver cycles in, 265,256 gold rows out (one per engine × cycle), 161 feature
  columns, every test row labelled.
- **Point-in-time check on the full data:** 1,416 engines cut at a random cycle, 134,007 rows recomputed from the
  cut history; all identical to gold (maximum difference 0, no null-pattern differences).

## SYNTHETIC REPLAY and incremental processing (`cmapss replay`; report: `reports/replay.md`)

| Step | Code | Rule | Decision |
|---|---|---|---|
| Replay | `ingest/replay.py` | Real test-file lines, 25 cycles per engine per batch, 5 late rows per batch; files `replay_<dataset>_SYNTHETIC_b<batch>.txt` | D-018 |
| Bronze | `ingest/bronze.py` | Same loader: new files only, insert-only MERGE, split `replay` | D-004 |
| Stream | `pipeline/incremental.py` | Structured Streaming on bronze (`availableNow`, checkpoint); touched engines recomputed from full bronze history; contract gate; `replaceWhere` into silver, quarantine, gold | D-019 |

- **Result (2026-11-01, full replay):** 20 batches delivered all 104,897 test rows of 707 engines (FD001–FD004).
  Bronze inserted exactly the delivered rows. Each batch's late rows left 24–88 rows of their engines in
  quarantine as `sequence_gap`; the next batch released them, and after the last batch the quarantine is empty.
- **Incremental = full rebuild:** silver (370,153 rows), quarantine (0) and gold (370,153 rows) maintained batch by
  batch match a full recomputation from bronze row for row (0 mismatches). Gold carries one feature-spec version.
- **Timing:** 84–163 s per batch on local Spark; split and causes in D-019 (per-batch overhead that grows with the
  tables, not with the rows delivered).
