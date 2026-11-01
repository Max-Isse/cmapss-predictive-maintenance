# Databricks mapping

How this repository maps onto Databricks Jobs, Unity Catalog and MLflow, what already carries over unchanged, and
the porting steps that remain. **Not deployed:** no Databricks workspace was available, so the bundle has not been
validated against one (`databricks bundle validate -t dev` is the first step). Decision: D-030.

## What carries over unchanged

| Local | Databricks | Notes |
|---|---|---|
| `cmapss <step>` CLI | Jobs `python_wheel_task`, entry point `cmapss` | the bundle builds the wheel with `uv build` |
| `docker compose run pipeline …` | job cluster on an ML runtime with Spark 4 / Delta 4 | `get_spark` returns the cluster's active session |
| Delta tables (bronze, silver, gold) | Delta tables | same formats, MERGE, `replaceWhere`, time travel (`versionAsOf`) |
| Structured Streaming, `trigger(availableNow)` + checkpoint | the same, as a scheduled job | the stream's semantics do not change |
| MLflow tracking (SQLite) | Databricks-managed MLflow | `tracking_uri: databricks` in `configs/databricks.yaml` |
| Registry alias `champion` | Unity Catalog registry alias `champion` | `registry_uri: databricks-uc`; three-level model name |
| pandera contract, policy, monitoring | unchanged Python | run as job tasks |

## Bundle (`databricks/databricks.yml`, `databricks/resources/jobs.yml`)

- **`cmapss-pipeline`** (hourly, paused by default): `bronze` (new landing files only) → `stream` (incremental
  silver and gold for the touched engines). This is the production path for replay or live deliveries.
- **`cmapss-model-lifecycle`** (on demand): `train` → `policy` → `score` → `monitor --retrain`.
- Targets `dev` and `prod` differ by catalog (`ml_dev`, `ml_prod`); `CMAPSS_CONFIG` points tasks at
  `configs/databricks.yaml`, which `extends` the local config.
- The one-off `fetch`, full `silver` and `gold` rebuilds and `replay` stay manual (run them as single-task runs).

## Porting steps that remain

1. **Managed tables instead of paths.** The code reads and writes Delta by path (`load(path)` / `save(path)`). On Unity
   Catalog, path-based Delta in a volume works for files but loses table governance; the port is to replace the path
   helpers (`bronze_path`, `silver_path`, `gold_path`) with table names (`catalog.schema.bronze_cycles`, …) and use
   `spark.table` / `saveAsTable`. All table access goes through those three helpers, so the change is local.
2. **Landing with Auto Loader.** `ingest/bronze.py` keeps its own file log (checksum, skip, insert-only MERGE). On
   Databricks, `cloudFiles` (Auto Loader) with a checkpoint gives the same exactly-once file semantics; keep the MERGE
   on `_record_hash` for idempotency across formats.
3. **Serving.** The FastAPI container can run unchanged on any container host reading gold through delta-rs, or the
   registered pyfunc can be served by Databricks Model Serving. That endpoint takes feature rows, so either the client
   sends them, or a feature lookup (Feature Serving / online tables on `gold.features`) replaces the delta-rs lookup.
4. **Secrets and identity.** Jobs run as a service principal with `USE CATALOG`, `USE SCHEMA`, `MODIFY` on the schema
   and `CREATE MODEL`; no credentials live in the repository.
5. **Compaction.** Schedule `OPTIMIZE` on bronze, gold and the prediction log (small files from per-batch MERGE and
   per-request appends; D-019, D-027), or enable predictive optimisation.

## Validation checklist (first deployment)

- `databricks bundle validate -t dev` and `databricks bundle deploy -t dev`.
- Run `fetch` → `bronze` → `silver` → `gold` once; compare row counts with `docs/data_lineage.md`.
- Run `cmapss-model-lifecycle`; the CV metrics must match `reports/model_comparison.md` (same seed, same data
  version), and the alias `champion` must point at version 1 of `ml_dev.cmapss.cmapss_maintenance`.
