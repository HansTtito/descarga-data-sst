from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from b2 import B2
from common import HERE, code_identity, daily_dates, load_config, product_path, resolve, sha256_file, target_coordinates


def date_from_name(path: Path) -> date | None:
    try:
        return datetime.strptime(path.name[:8], "%Y%m%d").date()
    except ValueError:
        return None


def raw_index(raw_dir: Path, start: date, end: date) -> dict[date, Path]:
    found: dict[date, list[Path]] = {}
    for path in raw_dir.glob("*.nc"):
        day = date_from_name(path)
        if day is not None and start <= day <= end:
            found.setdefault(day, []).append(path)
    expected = daily_dates(start, end)
    missing = [d.isoformat() for d in expected if d not in found]
    duplicates = {d.isoformat(): [p.name for p in v] for d, v in found.items() if len(v) != 1}
    if missing or duplicates:
        raise SystemExit(f"raw incompleto: missing={len(missing)} {missing[:20]} duplicates={duplicates}")
    return {d: found[d][0] for d in expected}


def to_celsius(data: xr.DataArray) -> xr.DataArray:
    units = str(data.attrs.get("units", "")).strip().lower()
    if units in {"k", "kelvin"}:
        return data - 273.15
    if units in {"c", "celsius", "degree_celsius", "degrees_celsius"}:
        return data
    raise ValueError(f"unidades SST no reconocidas: {units!r}")


def remap(path: Path, day: date, sst_var: str, lat: np.ndarray, lon: np.ndarray, method: str) -> np.ndarray:
    with xr.open_dataset(path) as ds:
        if ds.sizes.get("time") != 1:
            raise ValueError(f"{path.name}: se esperaba un tiempo")
        observed = ds.indexes["time"][0].date()
        if observed != day:
            raise ValueError(f"{path.name}: time={observed} esperado={day}")
        field = to_celsius(ds[sst_var].isel(time=0)).interp(lat=lat, lon=lon, method=method)
        field = field.transpose("lat", "lon").load()
    values = field.values.astype(np.float32)
    if not np.isfinite(values).any():
        raise ValueError(f"{path.name}: remap sin SST valida")
    return values


def year_block(year: int, days: list[date], files: dict[date, Path], cfg: dict, lat, lon, cache_dir: Path) -> np.ndarray:
    cache = cache_dir / f"{year}_{days[0].isoformat()}_{days[-1].isoformat()}.npy"
    if cache.exists():
        block = np.load(cache)
        if block.shape == (len(days), lat.size, lon.size):
            print(f"{year}: cache {cache.name}", flush=True)
            return block
    out = np.empty((len(days), lat.size, lon.size), dtype=np.float32)
    for i, day in enumerate(days):
        out[i] = remap(files[day], day, cfg["sst_var"], lat, lon, cfg["grid"]["interpolation"])
        if (i + 1) % 30 == 0 or i + 1 == len(days):
            print(f"{year}: {i + 1}/{len(days)} {day}", flush=True)
    tmp = cache.with_suffix(".tmp.npy")
    np.save(tmp, out)
    tmp.replace(cache)
    return out


def quality(values: np.ndarray, dates: list[date], q: dict) -> dict:
    finite = np.isfinite(values)
    frac = finite.reshape(values.shape[0], -1).mean(axis=1)
    vals = values[finite]
    stats = {
        "nan_pct": float((~finite).mean() * 100),
        "daily_valid_fraction_min": float(frac.min()),
        "daily_valid_fraction_max": float(frac.max()),
        "daily_valid_fraction_min_date": dates[int(frac.argmin())].isoformat(),
        "daily_valid_fraction_max_date": dates[int(frac.argmax())].isoformat(),
        "sst_min_c": float(vals.min()),
        "sst_max_c": float(vals.max()),
    }
    by_year = {}
    for year in sorted({d.year for d in dates}):
        idx = [i for i, d in enumerate(dates) if d.year == year]
        by_year[str(year)] = {"min": float(frac[idx].min()), "max": float(frac[idx].max())}
    stats["daily_valid_fraction_by_year"] = by_year
    errors = []
    if stats["daily_valid_fraction_min"] < float(q["min_daily_valid_fraction"]):
        errors.append("fraccion valida minima")
    if stats["daily_valid_fraction_max"] - stats["daily_valid_fraction_min"] > float(q["max_daily_valid_fraction_span"]):
        errors.append("mascara inestable")
    if stats["sst_min_c"] < float(q["min_sst_c"]) or stats["sst_max_c"] > float(q["max_sst_c"]):
        errors.append("sst fuera de rango")
    stats["errors"] = errors
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    b2 = None if args.no_upload else B2(cfg["b2"]["root_prefix"])
    start = date.fromisoformat(cfg["product"]["start"])
    end = date.fromisoformat(cfg["product"]["end"])
    out_path = product_path(cfg)
    out_dir = out_path.parent
    done = out_path.with_suffix(".json")
    if out_path.exists() and done.exists():
        prev = json.loads(done.read_text(encoding="utf-8"))
        if prev.get("status") == "created" and prev.get("product_sha256") == sha256_file(out_path):
            print(f"producto ya creado: {out_path} sha256={prev['product_sha256']}")
            return 0
    cache_dir = out_dir / "years"
    cache_dir.mkdir(parents=True, exist_ok=True)
    lat, lon = target_coordinates(cfg["grid"])
    files = raw_index(resolve(cfg["raw_dir"]), start, end)
    dates = list(files)
    blocks = []
    for year in range(start.year, end.year + 1):
        days = [d for d in dates if d.year == year]
        blocks.append(year_block(year, days, files, cfg, lat, lon, cache_dir))
    values = np.concatenate(blocks)
    stats = quality(values, dates, cfg["quality"])
    report_path = out_path.with_suffix(".json")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "days": len(dates),
        "shape": list(values.shape),
        "raw_dir": str(resolve(cfg["raw_dir"])),
        "code": code_identity(),
        **stats,
    }
    if stats["errors"]:
        report["status"] = "failed_quality"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if b2:
            b2.upload_file(report_path, cfg["b2"]["product_prefix"] + report_path.name)
        print(json.dumps(report, indent=2))
        return 1
    ds = xr.Dataset(
        {"sst": (("time", "lat", "lon"), values)},
        coords={"time": np.array(dates, dtype="datetime64[D]"), "lat": lat, "lon": lon},
        attrs={"source": "MUR-JPL-L4-GLOB-v4.1", "interpolation": cfg["grid"]["interpolation"]},
    )
    ds["sst"].attrs["units"] = "degree_Celsius"
    tmp = out_path.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp, encoding={"sst": {"zlib": True, "complevel": 4, "dtype": "float32"}})
    tmp.replace(out_path)
    report["status"] = "created"
    report["product"] = str(out_path)
    report["product_sha256"] = sha256_file(out_path)
    if b2:
        report["b2_product"] = b2.upload_file(out_path, cfg["b2"]["product_prefix"] + out_path.name)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if b2:
        b2.upload_file(report_path, cfg["b2"]["product_prefix"] + report_path.name)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
