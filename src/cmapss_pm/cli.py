"""Command-line entry point: `cmapss <step>` (Spark steps run inside the project container, see README)."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable

from cmapss_pm.config import Config, load_config

log = logging.getLogger("cmapss")


def cmd_fetch(cfg: Config, force: bool) -> None:
    from cmapss_pm.ingest.convert import stage_landing
    from cmapss_pm.ingest.fetch import fetch

    staged = stage_landing(cfg, fetch(cfg, force=force))
    log.info("landing: %s text files, %s Parquet files", len(staged["text"]), len(staged["parquet"]))


def cmd_bronze(cfg: Config, force: bool) -> None:
    from cmapss_pm.ingest.bronze import ingest_landing
    from cmapss_pm.spark import get_spark

    spark = get_spark(cfg)
    results = ingest_landing(spark, cfg, force=force)
    for r in results:
        log.info(
            "%-24s %-8s %-7s %-7s read=%-6s inserted=%s", r.file, r.fmt, r.table, r.status, r.rows_read, r.rows_inserted
        )
    import pandas as pd

    from cmapss_pm.ingest.bronze import bronze_counts

    counts = pd.DataFrame(bronze_counts(spark, cfg))
    out = cfg.path("tables_dir") / "bronze_counts.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out, index=False)
    pd.DataFrame([r.__dict__ for r in results]).to_csv(cfg.path("tables_dir") / "bronze_last_run.csv", index=False)
    log.info("wrote %s", out)
    log.info(
        "summary: %s",
        json.dumps(
            {
                "files": len(results),
                "loaded": sum(r.status == "loaded" for r in results),
                "rows_inserted": sum(r.rows_inserted for r in results),
            }
        ),
    )


def cmd_silver(cfg: Config) -> None:
    from cmapss_pm.quality.contract import silver_cycles_schema, silver_rul_schema, validate
    from cmapss_pm.quality.report import write_report
    from cmapss_pm.quality.silver import build_silver, silver_path
    from cmapss_pm.spark import get_spark

    spark = get_spark(cfg)
    res = build_silver(spark, cfg)
    cycles = spark.read.format("delta").load(silver_path(cfg, "cycles"))
    rul = spark.read.format("delta").load(silver_path(cfg, "rul"))
    errors = validate(silver_cycles_schema(cfg), cycles) + validate(silver_rul_schema(cfg), rul)
    path = write_report(spark, cfg, res, errors)
    log.info("silver %s: %s; quarantine by rule %s; report %s", res.run_id, res.counts, res.quarantine_by_rule, path)
    if errors:
        raise SystemExit(f"silver contract failed ({len(errors)} checks): see {path}")


def cmd_gold(cfg: Config) -> None:
    from cmapss_pm.features.gold import build_gold
    from cmapss_pm.features.report import write_feature_report
    from cmapss_pm.spark import get_spark

    spark = get_spark(cfg)
    res = build_gold(spark, cfg)
    path = write_feature_report(spark, cfg, res)
    log.info(
        "gold %s: %s; spec %s; point-in-time %s; report %s",
        res.run_id,
        res.counts,
        res.spec.version,
        res.point_in_time,
        path,
    )
    pit = res.point_in_time
    if pit["max_abs_diff"] > 1e-9 or pit["null_pattern_mismatches"] or pit["rows_compared"] != pit["rows_expected"]:
        raise SystemExit(f"point-in-time check failed: {pit}")


def cmd_stream(cfg: Config) -> None:
    from cmapss_pm.pipeline.incremental import run_stream
    from cmapss_pm.spark import get_spark

    for s in run_stream(get_spark(cfg), cfg):
        log.info("micro-batch %s", json.dumps(s))


def cmd_replay(cfg: Config, batches: int) -> None:
    from cmapss_pm.pipeline.replay_run import run_replay, write_replay_report
    from cmapss_pm.spark import get_spark

    spark = get_spark(cfg)
    for row in run_replay(spark, cfg, batches):
        log.info("replay batch %s", json.dumps(row))
    path, check = write_replay_report(spark, cfg)
    log.info("replay report %s; incremental = full recomputation: %s", path, check["passed"])
    if not check["passed"]:
        raise SystemExit(f"incremental tables differ from a full recomputation: {check}")


def cmd_train(cfg: Config) -> None:
    from mlflow.tracking import MlflowClient

    from cmapss_pm.models.data import load_model_data
    from cmapss_pm.models.registry import log_and_register
    from cmapss_pm.models.report import evaluate, write_model_report
    from cmapss_pm.models.train import train_all
    from cmapss_pm.spark import get_spark

    data = load_model_data(get_spark(cfg), cfg)
    log.info("model data: %s", json.dumps(data.meta))
    res = train_all(data, cfg)
    ev = evaluate(res, cfg)
    log.info("metrics: %s", json.dumps({k: round(v, 4) for k, v in ev["flat"].items()}))
    info = log_and_register(cfg, res.bundle, ev["flat"], data.test)
    path = write_model_report(res, ev, cfg, info)
    MlflowClient().log_artifact(info["run_id"], str(path), "reports")
    log.info("report %s", path)


STEPS = ["fetch", "bronze", "silver", "gold"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cmapss", description=__doc__)
    parser.add_argument("step", choices=[*STEPS, "all", "stream", "replay", "train"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true", help="fetch: re-download; bronze: re-read already-logged files")
    parser.add_argument("--batches", type=int, default=1, help="replay: number of SYNTHETIC REPLAY batches to deliver")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    steps: dict[str, Callable[[], None]] = {
        "fetch": lambda: cmd_fetch(cfg, args.force),
        "bronze": lambda: cmd_bronze(cfg, args.force),
        "silver": lambda: cmd_silver(cfg),
        "gold": lambda: cmd_gold(cfg),
        "stream": lambda: cmd_stream(cfg),
        "replay": lambda: cmd_replay(cfg, args.batches),
        "train": lambda: cmd_train(cfg),
    }
    for name in STEPS if args.step == "all" else [args.step]:
        steps[name]()


if __name__ == "__main__":
    main()
