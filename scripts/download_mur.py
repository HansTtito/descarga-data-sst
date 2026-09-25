from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import sys
import tempfile
from calendar import monthrange
from datetime import date, datetime, time, timezone
from pathlib import Path
from uuid import uuid4
import time as _time

import earthaccess
import xarray as xr
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "mur_c17.yaml"


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"Falta {name} en `.env`.")
    return value


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"Fecha invalida '{value}'; usa YYYY-MM-DD.") from exc


def validate_date_range(start: str, end: str) -> tuple[date, date]:
    start_date, end_date = parse_date(start), parse_date(end)
    if start_date > end_date:
        raise SystemExit("El inicio no puede ser posterior al fin.")
    return start_date, end_date


def month_windows(start_date: date, end_date: date):
    year, month = start_date.year, start_date.month
    while date(year, month, 1) <= end_date:
        last = date(year, month, monthrange(year, month)[1])
        yield max(start_date, date(year, month, 1)), min(end_date, last)
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1


def dataset_date(ds: xr.Dataset) -> date:
    if "time" not in ds.coords or ds.sizes.get("time") != 1:
        raise ValueError("Se esperaba exactamente una coordenada temporal por granule.")
    return ds.indexes["time"][0].date()


def date_from_granule_name(name: str) -> str:
    try:
        return datetime.strptime(name[:8], "%Y%m%d").date().isoformat()
    except (ValueError, TypeError):
        return ""


def subset_to_bytes(ds: xr.Dataset, bbox: dict, sst_var: str) -> bytes:
    lat_min, lat_max = bbox["lat_min"], bbox["lat_max"]
    lon_min, lon_max = bbox["lon_min"], bbox["lon_max"]

    for coord in ("lat", "lon"):
        if coord not in ds.coords:
            raise ValueError(f"El granule no contiene la coordenada '{coord}'.")
    if sst_var not in ds.data_vars:
        raise ValueError(f"El granule no contiene la variable SST '{sst_var}'.")

    sub = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    if sub.sizes.get("lat", 0) == 0 or sub.sizes.get("lon", 0) == 0:
        raise ValueError("El recorte espacial quedo vacio; revisa bbox y coordenadas.")
    if sub[sst_var].count().item() == 0:
        raise ValueError("El recorte SST no contiene ningun valor valido.")

    keep = [sst_var]
    for optional in ("sea_ice_fraction", "mask", "analysis_error"):
        if optional in sub:
            keep.append(optional)
    sub = sub[keep]

    encoding = {v: {"zlib": True, "complevel": 4} for v in sub.data_vars}
    loaded = sub.load()
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        loaded.to_netcdf(tmp_path, encoding=encoding)
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


def write_verified(payload: bytes, out_path: Path) -> tuple[int, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    size = len(payload)
    checksum = hashlib.sha1(payload).hexdigest()
    out_path.write_bytes(payload)
    written = out_path.read_bytes()
    if len(written) != size or hashlib.sha1(written).hexdigest() != checksum:
        raise IOError(f"Verificacion fallo al escribir {out_path}.")
    return size, checksum


def append_manifest_row(manifest_dir: Path, row: dict) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = manifest_dir / f"mur_{ts}_{uuid4().hex[:8]}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"  manifest -> {path}")


def earthdata_login() -> None:
    token = os.getenv("EARTHDATA_TOKEN", "").strip()
    user = os.getenv("EARTHDATA_USERNAME", "").strip()
    password = os.getenv("EARTHDATA_PASSWORD", "").strip()
    if not token and not (user and password):
        raise SystemExit(
            "Falta auth Earthdata en `.env`: usa EARTHDATA_TOKEN "
            "o EARTHDATA_USERNAME + EARTHDATA_PASSWORD."
        )
    auth = earthaccess.login(strategy="environment")
    if not auth:
        raise SystemExit("Login Earthdata fallo. Revisa token o usuario/contrasena.")


def with_retries(fn, *args, retries: int = 5, base_delay: float = 10.0, **kwargs):
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt == retries:
                break
            wait = base_delay * attempt
            print(f"  Reintento {attempt}/{retries} tras error de conexion: {exc}. Esperando {wait:.0f}s...")
            _time.sleep(wait)
    raise last_exc


def process_range(
    start: str,
    end: str,
    cfg: dict,
    skip_existing: bool,
    local_dir: Path,
    *,
    already_logged_in: bool = False,
) -> tuple[int, int, int]:
    start_date, end_date = validate_date_range(start, end)
    if not already_logged_in:
        earthdata_login()

    short_name = cfg["dataset"]["short_name"]
    version = str(cfg["dataset"].get("version", "4.1"))
    sst_var = cfg["dataset"].get("sst_var", "analysed_sst")
    bbox = cfg["bbox"]
    manifest_dir = local_dir.parent / "manifests" / "downloads"

    temporal_start = datetime.combine(start_date, time.min, tzinfo=timezone.utc).isoformat()
    temporal_end = datetime.combine(end_date, time.max, tzinfo=timezone.utc).isoformat()
    print(f"Buscando {short_name} {start} -> {end} (fin inclusivo) ...")
    results = with_retries(
        earthaccess.search_data,
        short_name=short_name,
        version=version,
        cloud_hosted=True,
        temporal=(temporal_start, temporal_end),
    )
    if not results:
        print(f"FAIL: Earthdata no devolvio granules para {start} -> {end}.")
        return 0, 0, 1

    print(f"Granules: {len(results)}")
    fileset = list(with_retries(earthaccess.open, results))
    if len(fileset) != len(results):
        raise RuntimeError(
            f"Earthdata devolvio {len(fileset)} handles para {len(results)} granules; "
            "se aborta para no truncar el lote silenciosamente."
        )
    ok, skipped, failed = 0, 0, 0

    for granule, handle in zip(results, fileset):
        try:
            native = Path(granule.data_links()[0]).name
        except Exception:
            native = f"mur_{ok + skipped + failed:04d}.nc"
        out_name = native.replace(".nc", "_HCS.nc")
        if not out_name.endswith(".nc"):
            out_name += "_HCS.nc"
        out_path = local_dir / out_name

        if skip_existing and out_path.exists():
            existing_bytes = out_path.read_bytes()
            append_manifest_row(
                manifest_dir,
                {
                    "granule": native,
                    "local_path": str(out_path),
                    "bytes": len(existing_bytes),
                    "sha1": hashlib.sha1(existing_bytes).hexdigest(),
                    "granule_date": date_from_granule_name(native),
                    "dataset_version": version,
                    "start": start,
                    "end": end,
                    "status": "skip_existing",
                    "utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            print(f"SKIP (ya en disco): {out_path}")
            skipped += 1
            continue

        print(f"Procesando: {out_name}")
        size = 0
        checksum = ""
        phase = "process"
        try:
            with xr.open_dataset(handle) as ds:
                granule_date = dataset_date(ds)
                if not start_date <= granule_date <= end_date:
                    print(f"  SKIP fuera de rango: {granule_date}")
                    skipped += 1
                    continue
                payload = subset_to_bytes(ds, bbox, sst_var)
            size, checksum = write_verified(payload, out_path)
            phase = "write_verified"
            print(f"  OK -> {out_path} ({size / 1e6:.1f} MB)")
            append_manifest_row(
                manifest_dir,
                {
                    "granule": native,
                    "local_path": str(out_path),
                    "bytes": size,
                    "sha1": checksum,
                    "granule_date": granule_date.isoformat(),
                    "dataset_version": version,
                    "start": start,
                    "end": end,
                    "status": "ok",
                    "utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"  FAIL durante {phase}: {exc}")
            append_manifest_row(
                manifest_dir,
                {
                    "granule": native,
                    "local_path": str(out_path),
                    "bytes": size,
                    "sha1": checksum,
                    "granule_date": "",
                    "dataset_version": version,
                    "start": start,
                    "end": end,
                    "status": f"fail:{phase}:{type(exc).__name__}",
                    "utc": datetime.now(timezone.utc).isoformat(),
                },
            )

    print()
    print(f"Resumen lote {start} -> {end}: ok={ok} skipped={skipped} failed={failed}")
    return ok, skipped, failed


def process_by_month(
    start: str,
    end: str,
    cfg: dict,
    skip_existing: bool,
    local_dir: Path,
) -> tuple[int, int, int]:
    start_date, end_date = validate_date_range(start, end)
    earthdata_login()
    total_ok = total_skip = total_fail = 0
    windows = list(month_windows(start_date, end_date))
    print(f"Periodo {start} -> {end}: {len(windows)} lote(s) mensual(es).")
    for i, (w_start, w_end) in enumerate(windows, start=1):
        print(f"\n===== Lote {i}/{len(windows)}: {w_start} -> {w_end} =====")
        ok, skipped, failed = process_range(
            start=w_start.isoformat(),
            end=w_end.isoformat(),
            cfg=cfg,
            skip_existing=skip_existing,
            local_dir=local_dir,
            already_logged_in=True,
        )
        total_ok += ok
        total_skip += skipped
        total_fail += failed
    print()
    print(f"Resumen total: ok={total_ok} skipped={total_skip} failed={total_fail}")
    return total_ok, total_skip, total_fail


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Descarga MUR SST recortado a disco local")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--start", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--local-dir", type=Path, default=ROOT / "data" / "raw" / "mur")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if bool(args.start) != bool(args.end):
        parser.error("--start y --end deben proporcionarse juntos")
    start = args.start or cfg["smoke"]["start"]
    end = args.end or cfg["smoke"]["end"]
    start_date, end_date = validate_date_range(start, end)
    use_months = (end_date.year, end_date.month) != (start_date.year, start_date.month)
    runner = process_by_month if use_months else process_range
    totals = runner(
        start=start,
        end=end,
        cfg=cfg,
        skip_existing=not args.no_skip_existing,
        local_dir=args.local_dir,
    )
    if isinstance(totals, tuple) and totals[2]:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
