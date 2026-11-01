"""Shared fixtures. Spark tests need Linux (the project container or CI); on native Windows they are skipped."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from cmapss_pm.config import Config, load_config, with_overrides

FIXTURES = Path(__file__).parent / "fixtures"
SPARK_UNAVAILABLE = os.name == "nt" and not os.environ.get("HADOOP_HOME")


def lake_cfg(tmp_path: Path) -> Config:
    """Config whose data paths all live under tmp_path."""
    return with_overrides(load_config(), root=tmp_path)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return lake_cfg(tmp_path)


@pytest.fixture
def landing(cfg: Config) -> Config:
    """Fixture files staged as text and as Parquet, like `cmapss fetch` does."""
    from cmapss_pm.ingest.convert import stage_landing

    src = cfg.root / "source"
    src.mkdir(parents=True, exist_ok=True)
    for f in FIXTURES.glob("*.txt"):
        shutil.copyfile(f, src / f.name)
    stage_landing(cfg, src)
    return cfg


@pytest.fixture(scope="session")
def spark() -> Iterator[object]:
    if SPARK_UNAVAILABLE:
        pytest.skip("Spark needs Linux: run `docker compose run --rm pipeline uv run pytest` (D-002)")
    from cmapss_pm.spark import get_spark

    session = get_spark(load_config(), app_name="cmapss-tests")
    yield session
    session.stop()
