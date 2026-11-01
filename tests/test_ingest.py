"""Fetch/convert (no Spark) and bronze ingestion (Spark): idempotency, cross-format dedup, append-only, metadata."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from cmapss_pm.config import Config, with_overrides
from cmapss_pm.ingest import fetch as fetch_mod
from cmapss_pm.ingest.convert import read_cmapss_text
from tests.conftest import FIXTURES

# ------------------------------------------------------------------ no Spark needed


def test_text_parsing_handles_trailing_spaces(cfg: Config) -> None:
    df = read_cmapss_text(FIXTURES / "train_FD001.txt", cfg.value_columns)
    assert df.shape == (10, 26)
    assert list(df.columns[:2]) == ["unit", "cycle"] and df["unit"].nunique() == 2
    with pytest.raises(ValueError, match="expected 1 columns"):
        read_cmapss_text(FIXTURES / "train_FD001.txt", ["rul"])


def test_staging_writes_text_and_parquet_with_same_rows(landing: Config) -> None:
    lz = landing.path("landing_dir")
    assert sorted(p.name for p in (lz / "text").iterdir()) == ["RUL_FD001.txt", "test_FD001.txt", "train_FD001.txt"]
    pq = pd.read_parquet(lz / "parquet" / "train_FD001.parquet")
    txt = read_cmapss_text(lz / "text" / "train_FD001.txt", landing.value_columns)
    pd.testing.assert_frame_equal(pq, txt)


def _fake_archive(path: Path) -> None:
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("train_FD001.txt", "1 1 0 0 100 " + " ".join(["1.0"] * 21) + " \n")
        z.writestr("readme.txt", "notes \x96 with a cp1252 dash\n")
        z.writestr("../evil.txt", "should not be extracted\n")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("6. Turbofan/CMAPSSData.zip", inner.getvalue())


def test_fetch_verifies_checksum_and_extracts_only_data_files(tmp_path: Path, cfg: Config) -> None:
    archive = cfg.path("raw_dir") / "download" / fetch_mod.ARCHIVE_NAME
    archive.parent.mkdir(parents=True)
    _fake_archive(archive)
    with pytest.raises(ValueError, match="does not match the pinned"):
        fetch_mod.download(cfg)
    ok = with_overrides(cfg, root=cfg.root, source={"archive_sha256": fetch_mod.sha256_file(archive)})
    out = fetch_mod.fetch(ok)
    assert sorted(p.name for p in out.iterdir()) == ["readme.txt", "train_FD001.txt"]
    assert not (cfg.path("raw_dir") / "evil.txt").exists()


def test_parse_name() -> None:
    from cmapss_pm.ingest.bronze import parse_name

    sf = parse_name(Path("RUL_FD004.txt"))
    assert (sf.split, sf.dataset, sf.fmt, sf.table) == ("rul", "FD004", "text", "rul")
    assert parse_name(Path("replay_FD002_batch_0003.parquet")).split == "replay"
    with pytest.raises(ValueError):
        parse_name(Path("train_FD009.txt"))


# ------------------------------------------------------------------ Spark


def _count(spark: object, cfg: Config, table: str) -> int:
    from cmapss_pm.ingest.bronze import bronze_path

    return spark.read.format("delta").load(bronze_path(cfg, table)).count()  # type: ignore[attr-defined]


def test_rerunning_the_same_files_inserts_nothing(spark: object, landing: Config) -> None:
    from cmapss_pm.ingest.bronze import ingest_landing, landing_files

    texts = landing_files(landing, ("text",))
    first = ingest_landing(spark, landing, texts)  # type: ignore[arg-type]
    assert {r.file: r.rows_inserted for r in first} == {"RUL_FD001.txt": 2, "test_FD001.txt": 6, "train_FD001.txt": 10}
    before = _count(spark, landing, "cycles")
    forced = ingest_landing(spark, landing, texts, force=True)  # type: ignore[arg-type]
    assert all(r.status == "loaded" and r.rows_inserted == 0 for r in forced)
    assert _count(spark, landing, "cycles") == before == 16
    assert _count(spark, landing, "rul") == 2
    skipped = ingest_landing(spark, landing, texts)  # type: ignore[arg-type]
    assert all(r.status == "skipped" for r in skipped)


def test_parquet_delivery_of_the_same_records_is_not_duplicated(spark: object, landing: Config) -> None:
    from cmapss_pm.ingest.bronze import ingest_landing

    results = ingest_landing(spark, landing)  # type: ignore[arg-type]  # text first, then Parquet
    by_fmt = {(r.file, r.fmt): r.rows_inserted for r in results}
    assert by_fmt[("train_FD001.parquet", "parquet")] == 0
    assert by_fmt[("RUL_FD001.parquet", "parquet")] == 0
    assert _count(spark, landing, "cycles") == 16


def test_changed_value_is_appended_not_updated_and_metadata_is_kept(spark: object, landing: Config) -> None:
    from cmapss_pm.ingest.bronze import bronze_path, ingest_landing

    lz = landing.path("landing_dir") / "text"
    ingest_landing(spark, landing, [lz / "train_FD001.txt"])  # type: ignore[arg-type]
    lines = (lz / "train_FD001.txt").read_text().splitlines()
    tokens = lines[0].split()
    tokens[-1] = "99.9999"  # a corrected sensor value for unit 1, cycle 1
    lines[0] = " ".join(tokens)
    lines.append("1 6 0.0001")  # malformed: 3 tokens
    redelivered = lz / "train_FD001_correction.txt"
    redelivered.write_text("\n".join(lines) + "\n")
    res = ingest_landing(spark, landing, [redelivered])  # type: ignore[arg-type]
    assert res[0].rows_inserted == 2  # the changed record and the malformed line; 9 unchanged rows skipped
    df = spark.read.format("delta").load(bronze_path(landing, "cycles"))  # type: ignore[attr-defined]
    assert df.where("unit = '1' AND cycle = '1'").count() == 2  # append-only: both versions kept
    bad = df.where("unit IS NULL").collect()
    assert len(bad) == 1 and bad[0]["_raw_line"] == "1 6 0.0001"
    meta = df.select("_source_file", "_source_format", "_file_sha256", "_ingest_run_id", "_ingest_ts").dropna()
    assert meta.count() == df.count()
    log = spark.read.format("delta").load(bronze_path(landing, "ingest_log"))  # type: ignore[attr-defined]
    assert {r["file"] for r in log.collect()} == {"train_FD001.txt", "train_FD001_correction.txt"}


def test_rul_units_follow_file_order(spark: object, landing: Config) -> None:
    from cmapss_pm.ingest.bronze import bronze_path, ingest_landing

    ingest_landing(spark, landing, [landing.path("landing_dir") / "text" / "RUL_FD001.txt"])  # type: ignore[arg-type]
    rows = spark.read.format("delta").load(bronze_path(landing, "rul")).orderBy("unit").collect()  # type: ignore[attr-defined]
    assert [(r["unit"], r["rul"]) for r in rows] == [("1", "112"), ("2", "98")]
