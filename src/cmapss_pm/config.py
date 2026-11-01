"""Load the YAML config and resolve paths relative to the project root."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# CMAPSS_ROOT / CMAPSS_CONFIG locate the project when the package is installed elsewhere (the API image).
PROJECT_ROOT = Path(os.environ.get("CMAPSS_ROOT") or Path(__file__).resolve().parents[2])
DEFAULT_CONFIG = Path(os.environ.get("CMAPSS_CONFIG") or PROJECT_ROOT / "configs" / "config.yaml")


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    root: Path

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def path(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else self.root / p

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw[name])

    @property
    def sensor_columns(self) -> list[str]:
        return [f"sensor_{i:02d}" for i in range(1, int(self.raw["schema"]["n_sensors"]) + 1)]

    @property
    def value_columns(self) -> list[str]:
        """All 26 columns of a C-MAPSS row, in file order."""
        s = self.raw["schema"]
        return [*s["key"], *s["settings"], *self.sensor_columns]


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: Path | str | None = None, root: Path | None = None) -> Config:
    """Load a YAML config; `extends: <file>` (relative to it) merges it over a base config, e.g. databricks.yaml."""
    file = Path(path or DEFAULT_CONFIG)
    with file.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if "extends" in raw:
        base = load_config(file.parent / raw.pop("extends"), root).raw
        raw = _merge(base, raw)
    return Config(raw=raw, root=root or PROJECT_ROOT)


def with_overrides(cfg: Config, root: Path, **sections: dict[str, Any]) -> Config:
    """Copy of `cfg` rooted elsewhere, with whole config sections merged (used by tests)."""
    raw = dict(cfg.raw)
    for name, values in sections.items():
        raw[name] = {**raw.get(name, {}), **values}
    return Config(raw=raw, root=root)
