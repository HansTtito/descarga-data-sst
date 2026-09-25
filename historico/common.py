from __future__ import annotations

import hashlib
import subprocess
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def daily_dates(start: date, end: date) -> list[date]:
    if start > end:
        raise ValueError("start > end")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def target_coordinates(grid: dict) -> tuple[np.ndarray, np.ndarray]:
    step = float(grid["step_degrees"])
    lat = np.arange(float(grid["lat_start"]), float(grid["lat_end"]) + step / 2, step, dtype=np.float64)
    lon = np.arange(float(grid["lon_start"]), float(grid["lon_end"]) + step / 2, step, dtype=np.float64)
    if (lat.size, lon.size) != (int(grid["height"]), int(grid["width"])):
        raise ValueError(f"grilla {(lat.size, lon.size)} != {(grid['height'], grid['width'])}")
    return lat, lon


def product_path(cfg: dict) -> Path:
    p = cfg["product"]
    return resolve(cfg["product_dir"]) / f"c17_{p['start']}_{p['end']}.nc"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def code_identity() -> dict:
    files = sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.yaml"))
    identity = {f.name: sha256_file(f) for f in files}
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        identity["git_head"] = head.stdout.strip() or None
        identity["git_dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        identity["git_head"] = None
    return identity
