# C-MAPSS predictive maintenance

A production-style predictive-maintenance pipeline on the NASA C-MAPSS turbofan simulation data:
- a PySpark + Delta Lake medallion lakehouse (bronze → silver → gold);
- remaining-useful-life and failure-risk models;
- a cost-based maintenance policy;
- FastAPI serving, monitoring and retraining.

It runs locally and maps onto Databricks Jobs, Unity Catalog and MLflow.

**Status:** phases 0–5 of 8 complete (ingestion to bronze, validated silver, gold features and labels, SYNTHETIC
replay with incremental streaming updates, RUL and failure-risk models in MLflow). See [docs/phase_plan.md](docs/phase_plan.md).

## Decision supported

When to pull each engine for maintenance. Too late risks an in-service failure; too early wastes remaining useful
life and shop capacity. Costs, lead time and the comparison baselines are in
[docs/business_case.md](docs/business_case.md). The data is a NASA **simulation**; see
[data/README.md](data/README.md) for what that means for any real-world claim.

## How to run

Spark needs Linux, so all Spark steps run in the project container (D-002). Prerequisites: Docker Desktop and
[uv](https://docs.astral.sh/uv/).

```bash
docker compose build pipeline
```

```bash
uv run cmapss fetch
```

```bash
docker compose run --rm pipeline uv run cmapss bronze
```

```bash
docker compose run --rm pipeline uv run cmapss silver
```

```bash
docker compose run --rm pipeline uv run cmapss gold
```

```bash
docker compose run --rm pipeline uv run cmapss replay --batches 20
```

```bash
docker compose run --rm pipeline uv run cmapss train
```

```bash
docker compose run --rm pipeline uv run pytest
```

- `fetch` downloads the archive from NASA, verifies the pinned checksum, extracts it and stages text and Parquet
  files in `data/landing/`. It runs on the host or in the container.
- `bronze` loads every landing file into Delta with an insert-only MERGE. Re-running it inserts nothing (D-004).
- `silver` types, validates and deduplicates bronze into silver. Rejected rows go to `silver/quarantine` with a
  reason. A pandera contract gates the result, and the step writes a data-quality report to
  [reports/data_quality/latest.md](reports/data_quality/latest.md).
- `gold` fits the feature spec on training engines (operating conditions, near-constant sensors, per-condition
  normalisation), builds 161 point-in-time features and the labels for every engine × cycle, checks
  point-in-time correctness on the full data and writes the evidence to [reports/features.md](reports/features.md).
- `replay` is a **SYNTHETIC REPLAY**: it re-delivers the real NASA test engines into `data/landing/replay/`, 25
  cycles per engine per batch, with 5 deliberately late rows per batch. Each batch goes through bronze, then a
  Structured Streaming step (`cmapss stream`) recomputes silver and gold for the engines it touched. The run
  checks that the result equals a full rebuild ([reports/replay.md](reports/replay.md)).
- `train` cross-validates the baselines and LightGBM models by engine, fits the calibrator and conformal intervals,
  scores the official NASA test engines once, logs everything to MLflow (`mlruns/`) and registers the bundle as
  `cmapss-maintenance` with alias `champion` when it improves. Results:
  [reports/model_comparison.md](reports/model_comparison.md).
- Every threshold, path and cost assumption is in [configs/config.yaml](configs/config.yaml).

## Layout

| Path | Contents |
|---|---|
| `src/cmapss_pm/ingest/` | download and verification, landing staging, bronze Delta loader |
| `src/cmapss_pm/quality/` | silver rules and quarantine, pandera contract, data-quality report |
| `src/cmapss_pm/pipeline/` | incremental bronze → silver → gold stream, replay driver and report |
| `src/cmapss_pm/models/` | training data, grouped CV, baselines and LightGBM, conformal intervals, MLflow registry, report |
| `src/cmapss_pm/features/` | feature spec (fit on training engines), shared feature code, labels, gold build, evidence report |
| `docker/` | Linux pipeline image (Java 17 + Spark 4 + Delta) |
| `tests/` | pytest; Spark tests run in the container and on CI |
| `docs/` | business case, methodology, decisions, phase plan, lineage |
