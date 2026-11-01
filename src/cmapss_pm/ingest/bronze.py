"""Bronze layer: raw, append-only Delta tables with ingest metadata, loaded idempotently with MERGE.

Tables (under lakehouse/bronze/):
* cycles      one row per delivered engine-cycle record: dataset, split, the 26 values *as delivered*
              (strings), and metadata (_source_file, _source_format, _file_sha256, _ingest_run_id,
              _ingest_ts, _record_hash).
* rul         the true remaining useful life of each test engine (RUL_FD00x files).
* ingest_log  one row per file per run: format, checksum, rows read, rows inserted, skipped or loaded.

Idempotency (decisions.md D-004): each file is deduplicated and MERGEd insert-only on
(dataset, split, unit, cycle, _record_hash). _record_hash is xxhash64 of the values cast to double, so the same
record delivered again, in the same file or as text vs Parquet, matches and is not inserted. A changed value for
the same unit/cycle is a *different* record and is kept (bronze never updates); silver resolves it.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from cmapss_pm.config import Config
from cmapss_pm.ingest.fetch import sha256_file

FILE_PATTERN = re.compile(r"^(?P<split>train|test|RUL|replay)_(?P<dataset>FD00[1-4])(?:_.*)?\.(?P<ext>txt|parquet)$")
META = ["_raw_line", "_source_file", "_source_format", "_file_sha256", "_ingest_run_id", "_ingest_ts", "_record_hash"]


@dataclass(frozen=True)
class SourceFile:
    path: Path
    split: str  # train | test | replay | rul
    dataset: str
    fmt: str  # text | parquet

    @property
    def table(self) -> str:
        return "rul" if self.split == "rul" else "cycles"


def parse_name(path: Path) -> SourceFile:
    m = FILE_PATTERN.match(path.name)
    if not m:
        raise ValueError(f"unrecognised landing file name: {path.name}")
    split = m["split"].lower()
    return SourceFile(path, split, m["dataset"], "text" if m["ext"] == "txt" else "parquet")


def bronze_path(cfg: Config, table: str) -> str:
    return str(cfg.path("lakehouse_dir") / "bronze" / table)


def _columns(cfg: Config, sf: SourceFile) -> list[str]:
    return ["rul"] if sf.table == "rul" else cfg.value_columns


def read_source(spark: SparkSession, cfg: Config, sf: SourceFile) -> DataFrame:
    """Read one landing file into string columns named per the C-MAPSS schema (values as delivered)."""
    cols = _columns(cfg, sf)
    if sf.fmt == "text":
        raw = spark.read.text(str(sf.path)).where(F.trim("value") != "")
        tokens = F.split(F.trim("value"), r"\s+")
        df = raw.select(
            *[tokens.getItem(i).alias(c) for i, c in enumerate(cols)],
            F.size(tokens).alias("_n_tokens"),
            F.col("value").alias("_raw_line"),
        )
        # wrong token count: keep the row (bronze is raw) with its raw line, blank the values; silver quarantines
        df = df.select(*[F.when(F.col("_n_tokens") == len(cols), F.col(c)).alias(c) for c in cols], "_raw_line")
    else:
        df = spark.read.parquet(str(sf.path)).select(
            *[F.col(c).cast("string").alias(c) for c in cols], F.lit(None).cast("string").alias("_raw_line")
        )
    if sf.table == "rul":
        # RUL files list one value per test engine, in unit order
        from pyspark.sql.window import Window

        df = df.withColumn("unit", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())).cast("string"))
    return df


def with_metadata(df: DataFrame, cfg: Config, sf: SourceFile, run_id: str, file_sha: str) -> DataFrame:
    cols = _columns(cfg, sf)
    names = ["unit", *cols] if sf.table == "rul" else cols
    # try_cast: bronze must accept any delivered text (Spark's ANSI cast would fail the whole load on "abc")
    hash_cols = [F.col(c).try_cast("double") for c in names]
    all_numeric = F.lit(True)
    for c in names:
        all_numeric = all_numeric & (F.col(c).isNull() | F.col(c).try_cast("double").isNotNull())
    return (
        df.withColumn("dataset", F.lit(sf.dataset))
        .withColumn("split", F.lit(sf.split))
        .withColumn("_source_file", F.lit(sf.path.name))
        .withColumn("_source_format", F.lit(sf.fmt))
        .withColumn("_file_sha256", F.lit(file_sha))
        .withColumn("_ingest_run_id", F.lit(run_id))
        .withColumn("_ingest_ts", F.current_timestamp())
        # hash of the values (format-independent); malformed text rows hash their raw line instead
        .withColumn(
            "_record_hash",
            F.when(F.col(cols[0]).isNull(), F.xxhash64("_raw_line"))  # malformed line
            .when(all_numeric, F.xxhash64(*hash_cols))  # numeric values: format-independent
            .otherwise(F.xxhash64(*[F.col(c) for c in names])),  # non-numeric text: hash the delivered strings
        )
    )


def merge_keys(table: str) -> list[str]:
    return (
        ["dataset", "split", "unit", "_record_hash"]
        if table == "rul"
        else ["dataset", "split", "unit", "cycle", "_record_hash"]
    )


def _ordered(df: DataFrame, cfg: Config, table: str) -> DataFrame:
    values = ["unit", "rul"] if table == "rul" else cfg.value_columns
    return df.select("dataset", "split", *values, *META)


def merge_into(spark: SparkSession, path: str, source: DataFrame, keys: list[str]) -> int:
    """Insert-only MERGE; returns the number of rows inserted. Creates the table on first use."""
    if not DeltaTable.isDeltaTable(spark, path):
        source.limit(0).write.format("delta").partitionBy("dataset").save(path)
    target = DeltaTable.forPath(spark, path)
    version_before = target.history(1).select("version").first()[0]  # type: ignore[index]
    cond = " AND ".join(f"t.{k} = s.{k}" for k in keys)
    target.alias("t").merge(source.alias("s"), cond).whenNotMatchedInsertAll().execute()
    latest = target.history(1).select("version", "operationMetrics").first()
    # An insert-only MERGE with nothing to insert may not commit a new version; history(1) would then describe the
    # previous operation, so only read metrics from a commit made by this MERGE.
    if latest is None or latest["version"] == version_before or not latest["operationMetrics"]:
        return 0
    return int(latest["operationMetrics"].get("numTargetRowsInserted", 0))


@dataclass
class IngestResult:
    file: str
    fmt: str
    table: str
    status: str  # loaded | skipped
    rows_read: int
    rows_inserted: int
    file_sha256: str


def already_ingested(spark: SparkSession, cfg: Config, file_sha: str) -> bool:
    log_path = bronze_path(cfg, "ingest_log")
    if not DeltaTable.isDeltaTable(spark, log_path):
        return False
    return (
        spark.read.format("delta")
        .load(log_path)
        .where((F.col("file_sha256") == file_sha) & (F.col("status") == "loaded"))
        .limit(1)
        .count()
        > 0
    )


def ingest_file(spark: SparkSession, cfg: Config, path: Path, run_id: str, force: bool = False) -> IngestResult:
    sf = parse_name(path)
    file_sha = sha256_file(path)
    if not force and already_ingested(spark, cfg, file_sha):
        return IngestResult(path.name, sf.fmt, sf.table, "skipped", 0, 0, file_sha)
    df = with_metadata(read_source(spark, cfg, sf), cfg, sf, run_id, file_sha)
    keys = merge_keys(sf.table)
    df = _ordered(df.dropDuplicates(keys), cfg, sf.table).cache()
    rows_read = df.count()
    inserted = merge_into(spark, bronze_path(cfg, sf.table), df, keys)
    df.unpersist()
    return IngestResult(path.name, sf.fmt, sf.table, "loaded", rows_read, inserted, file_sha)


def write_log(spark: SparkSession, cfg: Config, run_id: str, results: list[IngestResult]) -> None:
    if not results:
        return
    now = datetime.now(UTC).replace(tzinfo=None)
    rows = [
        (run_id, now, r.file, r.fmt, r.table, r.status, r.rows_read, r.rows_inserted, r.file_sha256) for r in results
    ]
    schema = (
        "run_id string, ingest_ts timestamp, file string, format string, table string, status string, "
        "rows_read long, rows_inserted long, file_sha256 string"
    )
    spark.createDataFrame(rows, schema).write.format("delta").mode("append").save(bronze_path(cfg, "ingest_log"))


def landing_files(cfg: Config, folders: tuple[str, ...] = ("text", "parquet", "replay")) -> list[Path]:
    """Landing files by folder: text/ and replay/ hold .txt deliveries, parquet/ holds .parquet."""
    landing = cfg.path("landing_dir")
    files: list[Path] = []
    for folder in folders:
        ext = "parquet" if folder == "parquet" else "txt"
        files += sorted((landing / folder).glob(f"*.{ext}"))
    return files


def ingest_landing(
    spark: SparkSession, cfg: Config, files: list[Path] | None = None, force: bool = False
) -> list[IngestResult]:
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
    results = [ingest_file(spark, cfg, p, run_id, force=force) for p in (files or landing_files(cfg))]
    write_log(spark, cfg, run_id, results)
    return results


def bronze_counts(spark: SparkSession, cfg: Config) -> list[dict[str, object]]:
    """Rows, engines and malformed rows per dataset and split in bronze (feeds the lineage table)."""
    rows: list[dict[str, object]] = []
    for table in ("cycles", "rul"):
        path = bronze_path(cfg, table)
        if not DeltaTable.isDeltaTable(spark, path):
            continue
        df = spark.read.format("delta").load(path)
        agg = df.groupBy("dataset", "split").agg(
            F.count("*").alias("rows"),
            F.countDistinct("unit").alias("units"),
            F.sum(F.col("unit").isNull().cast("int")).alias("malformed_rows"),
            F.countDistinct("_source_format").alias("formats_seen"),
        )
        rows += [{"table": table, **r.asDict()} for r in agg.orderBy("dataset", "split").collect()]
    return rows
