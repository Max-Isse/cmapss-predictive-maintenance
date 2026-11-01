"""Stage source files in the landing zone: the NASA text files as delivered, plus Parquet conversions.

Landing layout (one sub-folder per format, so the pipeline must handle both):
    landing/text/{train,test,RUL}_FD00x.txt        copies of the NASA files
    landing/parquet/{train,test,RUL}_FD00x.parquet  same rows, typed columns (no rounding: parsed from the text)
The bronze loader treats them as independent deliveries of the same records; its MERGE keeps one copy.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from cmapss_pm.config import Config

RUL_COLUMNS = ["rul"]


def read_cmapss_text(path: Path, columns: list[str]) -> pd.DataFrame:
    """Parse a whitespace-separated C-MAPSS file (lines carry trailing spaces)."""
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != len(columns):
        raise ValueError(f"{path.name}: expected {len(columns)} columns, found {df.shape[1]}")
    df.columns = columns
    return df


def stage_landing(cfg: Config, source_dir: Path) -> dict[str, list[Path]]:
    landing = cfg.path("landing_dir")
    text_dir, pq_dir = landing / "text", landing / "parquet"
    text_dir.mkdir(parents=True, exist_ok=True)
    pq_dir.mkdir(parents=True, exist_ok=True)
    staged: dict[str, list[Path]] = {"text": [], "parquet": []}
    for path in sorted(source_dir.glob("*_FD00?.txt")):
        shutil.copyfile(path, text_dir / path.name)
        staged["text"].append(text_dir / path.name)
        cols = RUL_COLUMNS if path.name.startswith("RUL_") else cfg.value_columns
        df = read_cmapss_text(path, cols)
        target = pq_dir / f"{path.stem}.parquet"
        df.to_parquet(target, index=False)
        staged["parquet"].append(target)
    return staged
