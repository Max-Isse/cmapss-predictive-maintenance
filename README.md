# C-MAPSS predictive maintenance

A production-style predictive-maintenance pipeline on the NASA C-MAPSS turbofan simulation data:
- a PySpark + Delta Lake medallion lakehouse (bronze → silver → gold);
- remaining-useful-life and failure-risk models;
- a cost-based maintenance policy;
- FastAPI serving, monitoring and retraining.

It runs locally and maps onto Databricks Jobs, Unity Catalog and MLflow.

**Status:** phases 0–2 of 8 complete (ingestion to bronze, validated silver). See [docs/phase_plan.md](docs/phase_plan.md).

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
docker compose run --rm pipeline uv run pytest
```

- `fetch` downloads the archive from NASA, verifies the pinned checksum, extracts it and stages text and Parquet
  files in `data/landing/`. It runs on the host or in the container.
- `bronze` loads every landing file into Delta with an insert-only MERGE. Re-running it inserts nothing (D-004).
- `silver` types, validates and deduplicates bronze into silver. Rejected rows go to `silver/quarantine` with a
  reason. A pandera contract gates the result, and the step writes a data-quality report to
  [reports/data_quality/latest.md](reports/data_quality/latest.md).
- Every threshold, path and cost assumption is in [configs/config.yaml](configs/config.yaml).

## Layout

| Path | Contents |
|---|---|
| `src/cmapss_pm/ingest/` | download and verification, landing staging, bronze Delta loader |
| `docker/` | Linux pipeline image (Java 17 + Spark 4 + Delta) |
| `tests/` | pytest; Spark tests run in the container and on CI |
| `docs/` | business case, decisions, phase plan, lineage |
