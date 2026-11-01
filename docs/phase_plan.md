# Phase plan

Each phase ends with tests green, lint clean, and a summary of what was built, key results, assumptions and open
questions. Standards: `docs/project_standards.md`.

| Phase | Scope | Main outputs | Status |
|---|---|---|---|
| 0 | Repo structure, standards, phase plan, business case and cost assumptions, Spark/Delta runtime decision | this file, `docs/business_case.md`, `docs/decisions.md`, `configs/config.yaml`, `docker/` | done |
| 1 | Download and verify NASA data; landing zone with text and Parquet; **bronze** Delta tables (append-only, ingest metadata, insert-only MERGE); idempotency tests | `ingest/`, `data/raw_manifest.json`, bronze `cycles` / `rul` / `ingest_log` | done |
| 2 | **Silver**: typing, validation (schema, sensor ranges, strictly increasing cycles, no gaps, duplicates) with pandera or Great Expectations; quarantine table with reasons; per-run data-quality report with row counts per layer | `quality/silver.py`, `quality/contract.py`, `quality/report.py`, silver + quarantine tables, `reports/data_quality/`, D-006–D-012 | done |
| 3 | **Gold** features at engine × cycle grain: near-constant sensor removal (justified), operating-condition clustering and per-condition normalisation for FD002/FD004, rolling mean/std/slope windows in PySpark (one in Spark SQL); RUL with piecewise-linear cap, failure-within-N labels; point-in-time correctness test; shared train/serve feature code | `features/`, gold tables, `reports/features.md`, methodology §1–2, D-013–D-017 | done |
| 4 | **SYNTHETIC replay** of test engines cycle by cycle into the landing zone; streaming-style incremental bronze → silver → gold | `ingest/replay.py`, `pipeline/`, `reports/replay.md`, lineage, D-018–D-019 | done |
| 5 | Models: grouped split by engine, official test set as final holdout; baselines (mean RUL, linear degradation); LightGBM/XGBoost for RUL and failure-within-N; RMSE, MAE, NASA score, PR-AUC, recall at fixed precision, calibration; conformal or quantile intervals with coverage; segments by FD00x and true-RUL band; MLflow with Delta versions, model registry alias "champion" | `models/`, `reports/model_comparison.md`, methodology §3–4, D-020–D-023 | done |
| 6 | Decision layer: maintenance rule from P(failure within N), threshold by expected cost; compare with fixed-interval and run-to-failure; sensitivity on costs and N | `policy/`, `reports/maintenance_policy.md`, methodology §5, D-024–D-025 | done |
| 7 | Serving and operations: FastAPI (/health, /model-info, /predict, /predict/batch), structured JSON logs with request IDs, prediction log table, Dockerfile (slim, non-root), docker-compose with API + MLflow; drift (PSI) and performance monitoring; retraining trigger with challenger vs champion | `serving/`, `monitoring/`, `docker/Dockerfile.api`, `docker-compose.yml`, `reports/monitoring.md`, D-026–D-029 | done |
| 8 | `databricks/` asset bundle and mapping notes (Jobs, Unity Catalog, MLflow); documentation pack: recommendations, model card, lineage, limitations, monitoring plan, how-to-run | `databricks/`, `configs/databricks.yaml`, `reports/recommendations.md`, `docs/model_card.md`, `docs/limitations.md`, `docs/monitoring_plan.md`, D-030 | done |

## Repository layout

```
configs/config.yaml          paths, seeds, schema, cost assumptions, Spark settings
data/raw/                    NASA archive + extracted text (git-ignored); data/raw_manifest.json pins the version
data/landing/                files waiting for ingestion: text/, parquet/, later replay/ (git-ignored)
data/lakehouse/              Delta tables bronze/, silver/, gold/ (git-ignored)
docker/Dockerfile.dev        Linux image with Java 17 + uv + locked deps; runs every Spark step
docker-compose.yml           pipeline service (API and MLflow added in phase 7)
databricks/                  bundle config and mapping notes (phase 8)
src/cmapss_pm/               ingest/, quality/, features/, pipeline/, models/, policy/, serving/, monitoring/
tests/                       pytest; Spark tests run in the container and on Linux CI
docs/                        business case, methodology, decisions, lineage, model card
reports/                     generated reports, tables, figures
```
