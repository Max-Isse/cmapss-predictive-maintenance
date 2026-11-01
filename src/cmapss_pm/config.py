"""Load the YAML config and resolve paths relative to the project root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


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


def load_config(path: Path | str | None = None, root: Path | None = None) -> Config:
    with Path(path or DEFAULT_CONFIG).open(encoding="utf-8") as f:
        return Config(raw=yaml.safe_load(f), root=root or PROJECT_ROOT)


def with_overrides(cfg: Config, root: Path, **sections: dict[str, Any]) -> Config:
    """Copy of `cfg` rooted elsewhere, with whole config sections merged (used by tests)."""
    raw = dict(cfg.raw)
    for name, values in sections.items():
        raw[name] = {**raw.get(name, {}), **values}
    return Config(raw=raw, root=root)
