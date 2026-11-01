"""Download the NASA C-MAPSS archive, verify its checksum, extract the text files, write the manifest.

The archive is NASA PCoE data set 6; it contains a nested `CMAPSSData.zip` with train/test/RUL text files for
FD001-FD004, a readme and the reference paper. Nothing is modified after extraction.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import urllib.request
import zipfile
from pathlib import Path

from cmapss_pm.config import Config

log = logging.getLogger(__name__)

ARCHIVE_NAME = "cmapss_nasa.zip"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(cfg: Config, force: bool = False) -> Path:
    src = cfg.section("source")
    target = cfg.path("raw_dir") / "download" / ARCHIVE_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    if force or not target.exists():
        log.info("downloading %s", src["url"])
        with urllib.request.urlopen(src["url"], timeout=300) as r:  # fixed official NASA URL from config
            target.write_bytes(r.read())
    digest = sha256_file(target)
    if digest != src["archive_sha256"]:
        raise ValueError(
            f"archive checksum {digest} does not match the pinned {src['archive_sha256']}: the published data "
            "changed or the download is corrupt. Results would not be comparable; stop and investigate."
        )
    return target


def extract(cfg: Config, archive: Path) -> dict[str, dict[str, object]]:
    """Extract the .txt data files and readme from the nested archive; return per-file metadata."""
    src = cfg.section("source")
    out_dir = cfg.path("raw_dir") / "cmapss"
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as outer:
        inner_name = next(n for n in outer.namelist() if n.endswith(src["inner_archive"]))
        inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    files: dict[str, dict[str, object]] = {}
    for name in sorted(inner.namelist()):
        if not name.endswith(".txt") or "/" in name or ".." in name:
            continue
        data = inner.read(name)
        (out_dir / name).write_bytes(data)
        rows = sum(1 for line in data.decode("latin-1").splitlines() if line.strip())
        files[name] = {"bytes": len(data), "rows": rows, "sha256": sha256_bytes(data)}
    return files


def fetch(cfg: Config, force: bool = False) -> Path:
    archive = download(cfg, force=force)
    files = extract(cfg, archive)
    src = cfg.section("source")
    manifest = {"source_page": src["page"], "url": src["url"], "archive_sha256": src["archive_sha256"], "files": files}
    cfg.path("manifest").parent.mkdir(parents=True, exist_ok=True)
    cfg.path("manifest").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log.info("extracted %s files to %s", len(files), cfg.path("raw_dir") / "cmapss")
    return cfg.path("raw_dir") / "cmapss"
