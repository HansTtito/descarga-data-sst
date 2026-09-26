from __future__ import annotations

import argparse
import calendar
import hashlib
import os
import sys
import tempfile
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.upload_to_b2 import (
    ROOT,
    append_manifest_row,
    b2_api,
    get_object_info,
    load_config,
    require_env,
    upload_verified,
    with_retries,
)

DEFAULT_CONFIG = ROOT / "configs" / "era5.yaml"
PRELIMINARY_EXPVER = "0005"


def month_chunks(start: date, end: date) -> list[tuple[date, date]]:
    if start > end:
        raise SystemExit(f"--start {start} es posterior a --end {end}")
    chunks = []
    first = start
    while first <= end:
        last_day = calendar.monthrange(first.year, first.month)[1]
        last = min(date(first.year, first.month, last_day), end)
        chunks.append((first, last))
        first = last + timedelta(days=1)
    return chunks


def file_name(first: date, last: date, preliminary: bool = False) -> str:
    suffix = "_prelim" if preliminary else ""
    return f"era5_{first:%Y%m%d}_{last:%Y%m%d}{suffix}.nc"


def build_request(cfg: dict, first: date, last: date) -> dict:
    bbox = cfg["bbox"]
    return {
        "product_type": ["reanalysis"],
        "variable": list(cfg["variables"]),
        "year": [f"{first.year}"],
        "month": [f"{first.month:02d}"],
        "day": [f"{d:02d}" for d in range(first.day, last.day + 1)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [bbox["lat_max"], bbox["lon_min"], bbox["lat_min"], bbox["lon_max"]],
    }


def open_download(path: Path, work: Path) -> xr.Dataset:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = [m for m in archive.namelist() if m.endswith(".nc")]
            if not members:
                raise ValueError(f"{path.name}: zip sin NetCDF")
            archive.extractall(work, members)
        parts = []
        for member in members:
            with xr.open_dataset(work / member) as part:
                parts.append(part.load())
        ds = xr.merge(parts, join="exact", compat="no_conflicts")
    else:
        with xr.open_dataset(path) as single:
            ds = single.load()
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    return ds.drop_vars([v for v in ("number",) if v in ds.variables])


def expver_values(ds: xr.Dataset) -> list[str]:
    if "expver" not in ds.variables:
        return []
    return sorted({str(v) for v in np.atleast_1d(ds["expver"].values)})


def validate(ds: xr.Dataset, cfg: dict, first: date, last: date) -> dict:
    missing = [short for short in cfg["variables"].values() if short not in ds.data_vars]
    if missing:
        raise ValueError(f"variables ausentes {missing}; presentes {sorted(ds.data_vars)}")
    expected = np.arange(
        np.datetime64(first.isoformat()),
        np.datetime64((last + timedelta(days=1)).isoformat()),
        np.timedelta64(1, "h"),
    )
    observed = ds["time"].values.astype("datetime64[h]")
    if observed.shape != expected.shape or not np.array_equal(observed, expected):
        raise ValueError(f"tiempos {observed.size} (esperados {expected.size}) {observed[:1]}..{observed[-1:]}")
    bbox = cfg["bbox"]
    lat, lon = ds["latitude"].values, ds["longitude"].values
    tol = 0.26
    if lat.min() > bbox["lat_min"] + tol or lat.max() < bbox["lat_max"] - tol:
        raise ValueError(f"latitud {lat.min()}..{lat.max()} no cubre {bbox['lat_min']}..{bbox['lat_max']}")
    if lon.min() > bbox["lon_min"] + tol or lon.max() < bbox["lon_max"] - tol:
        raise ValueError(f"longitud {lon.min()}..{lon.max()} no cubre {bbox['lon_min']}..{bbox['lon_max']}")
    finite = {short: float(np.isfinite(ds[short].values).mean()) for short in cfg["variables"].values()}
    bad = {k: v for k, v in finite.items() if v < 0.999}
    if bad:
        raise ValueError(f"fraccion finita baja {bad}")
    return {"hours": int(observed.size), "lat": [float(lat.min()), float(lat.max())], "lon": [float(lon.min()), float(lon.max())]}


def write_compressed(ds: xr.Dataset, cfg: dict, target: Path) -> None:
    encoding = {short: {"zlib": True, "complevel": 4, "dtype": "float32"} for short in cfg["variables"].values()}
    tmp = target.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp, encoding=encoding)
    tmp.replace(target)


def retrieve(client, cfg: dict, first: date, last: date, target: Path) -> None:
    client.retrieve(cfg["dataset"], build_request(cfg, first, last)).download(str(target))


def process_chunk(client, bucket, cfg: dict, first: date, last: date, local_dir: Path, delete_local: bool) -> str:
    prefix = cfg["b2"]["prefix"]
    final_key = prefix + file_name(first, last)
    if with_retries(get_object_info, bucket, final_key) is not None:
        print(f"SKIP ya en B2: {final_key}", flush=True)
        return "skipped"
    with tempfile.TemporaryDirectory(prefix="era5_") as work_dir:
        work = Path(work_dir)
        raw = work / "download"
        print(f"CDS {first} -> {last} ...", flush=True)
        with_retries(retrieve, client, cfg, first, last, raw, retries=3, base_delay=60.0)
        ds = open_download(raw, work)
        checks = validate(ds, cfg, first, last)
        expvers = expver_values(ds)
        preliminary = PRELIMINARY_EXPVER in expvers
        name = file_name(first, last, preliminary)
        local_dir.mkdir(parents=True, exist_ok=True)
        local = local_dir / name
        write_compressed(ds, cfg, local)
    payload = local.read_bytes()
    key = prefix + name
    if with_retries(get_object_info, bucket, key) is not None:
        print(f"SKIP ya en B2: {key}", flush=True)
        return "skipped"
    size, sha1 = with_retries(upload_verified, bucket, payload, key)
    append_manifest_row(
        bucket,
        cfg["b2"]["manifest_prefix"],
        {
            "uploaded_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": cfg["dataset"],
            "start": first.isoformat(),
            "end": last.isoformat(),
            "variables": " ".join(cfg["variables"].values()),
            "expver": " ".join(expvers),
            "preliminary": preliminary,
            "hours": checks["hours"],
            "b2_key": key,
            "bytes": size,
            "sha1": sha1,
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    )
    print(f"OK {key} {size / 1e6:.1f} MB{' (ERA5T preliminar)' if preliminary else ''}", flush=True)
    if delete_local:
        local.unlink()
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--local-dir", type=Path, default=ROOT / "data" / "raw" / "era5")
    parser.add_argument("--delete-local", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    cfg = load_config(args.config)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    newest = date.today() - timedelta(days=int(cfg.get("latency_days", 6)))
    if end > newest:
        print(f"--end {end} recortado a {newest} por latencia de ERA5", flush=True)
        end = newest
    chunks = month_chunks(start, end)
    print(f"{len(chunks)} bloques mensuales {chunks[0][0]} -> {chunks[-1][1]}", flush=True)
    if args.dry_run:
        for first, last in chunks:
            print(cfg["b2"]["prefix"] + file_name(first, last))
        print(build_request(cfg, *chunks[0]))
        return 0
    import cdsapi

    client = cdsapi.Client(
        url=os.getenv("CDSAPI_URL", "https://cds.climate.copernicus.eu/api").strip(),
        key=require_env("CDSAPI_KEY"),
    )
    api, bucket_name = b2_api()
    bucket = api.get_bucket_by_name(bucket_name)
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    for first, last in chunks:
        try:
            counts[process_chunk(client, bucket, cfg, first, last, args.local_dir, args.delete_local)] += 1
        except Exception as exc:
            counts["failed"] += 1
            print(f"FALLO {first} -> {last}: {exc}", flush=True)
    print(f"ok={counts['ok']} skipped={counts['skipped']} failed={counts['failed']}", flush=True)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
