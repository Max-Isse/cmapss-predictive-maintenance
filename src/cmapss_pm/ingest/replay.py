"""SYNTHETIC REPLAY: re-deliver the NASA test engines batch by batch into landing/replay/, as a live feed would.

What is real and what is simulated (D-018):
- real: every value. Each delivered line is the NASA test-file line, byte for byte.
- simulated: when rows arrive. Batch b carries cycles (b-1)*k+1 .. b*k of every test engine (k =
  `replay.cycles_per_batch`); all engines start together at batch 1; an engine drops out after its last test
  cycle. As a deliberate, SYNTHETIC disturbance, `replay.late_rows` rows per batch are held back and delivered
  with the next batch (late arrival), which the pipeline must absorb.

Files are named replay_<dataset>_SYNTHETIC_b<batch>.txt (split "replay" in bronze), so the label travels with
every row via bronze's _source_file. Progress is kept in landing/replay/_state.json.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from cmapss_pm.config import Config


def replay_dir(cfg: Config) -> Path:
    return cfg.path("landing_dir") / "replay"


def _state_file(cfg: Config) -> Path:
    return replay_dir(cfg) / "_state.json"


@dataclass
class ReplayState:
    next_batch: int = 1
    held: list[dict[str, Any]] = field(default_factory=list)  # late rows: {"dataset", "unit", "cycle", "line"}
    batches: list[dict[str, Any]] = field(default_factory=list)


def load_state(cfg: Config) -> ReplayState:
    path = _state_file(cfg)
    if not path.exists():
        return ReplayState()
    return ReplayState(**json.loads(path.read_text(encoding="utf-8")))


def _save_state(cfg: Config, state: ReplayState) -> None:
    path = _state_file(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=1), encoding="utf-8")


def _test_lines(cfg: Config, dataset: str) -> list[tuple[int, int, str]]:
    """(unit, cycle, line as delivered) for every row of the NASA test file staged in landing/text/."""
    path = cfg.path("landing_dir") / "text" / f"test_{dataset}.txt"
    rows = []
    for line in path.read_text(encoding="latin-1").splitlines():
        tokens = line.split()
        if tokens:
            rows.append((int(tokens[0]), int(tokens[1]), line))
    return rows


@dataclass
class ReplayBatch:
    batch: int
    files: list[Path]
    rows: int
    engines: int
    late_held: int
    late_released: int
    finished: bool  # True when every test row has been delivered


def emit_batch(cfg: Config) -> ReplayBatch | None:
    """Write the next batch's files; None when the replay is complete."""
    rc = cfg.section("replay")
    k = int(rc["cycles_per_batch"])
    state = load_state(cfg)
    b = state.next_batch
    lo, hi = (b - 1) * k, b * k
    sources = {d: _test_lines(cfg, d) for d in rc["datasets"]}
    last_cycle = max(c for rows in sources.values() for _, c, _ in rows)
    if lo >= last_cycle and not state.held:
        return None
    rows = [
        {"dataset": d, "unit": u, "cycle": c, "line": line}
        for d, src in sources.items()
        for u, c, line in src
        if lo < c <= hi
    ]
    # late arrival: hold rows that are not their engine's last row in this batch, so a gap appears now and is
    # filled by the next batch (only while another batch will follow)
    last_in_batch: dict[tuple[str, int], int] = {}
    for r in rows:
        key = (r["dataset"], r["unit"])
        last_in_batch[key] = max(last_in_batch.get(key, 0), r["cycle"])
    candidates = [i for i, r in enumerate(rows) if r["cycle"] < last_in_batch[(r["dataset"], r["unit"])]]
    n_late = min(int(rc["late_rows"]), len(candidates)) if hi < last_cycle else 0
    rng = np.random.default_rng(cfg.seed + b)
    late_idx = set(rng.choice(candidates, size=n_late, replace=False).tolist()) if n_late else set()
    released = state.held
    delivered = released + [r for i, r in enumerate(rows) if i not in late_idx]
    held = [rows[i] for i in sorted(late_idx)]

    out_dir = replay_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for d in rc["datasets"]:
        lines = [r["line"] for r in delivered if r["dataset"] == d]
        if lines:
            path = out_dir / f"replay_{d}_SYNTHETIC_b{b:04d}.txt"
            path.write_text("\n".join(lines) + "\n", encoding="latin-1", newline="\n")
            files.append(path)
    finished = hi >= last_cycle and not held
    batch = ReplayBatch(
        batch=b,
        files=files,
        rows=len(delivered),
        engines=len({(r["dataset"], r["unit"]) for r in delivered}),
        late_held=len(held),
        late_released=len(released),
        finished=finished,
    )
    state.next_batch = b + 1
    state.held = held
    state.batches.append(
        {
            **{k: v for k, v in asdict(batch).items() if k != "files"},
            "files": [p.name for p in files],
            "cycles": [lo + 1, hi],
            "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )
    _save_state(cfg, state)
    return batch
