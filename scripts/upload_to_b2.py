from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

MIN_AGE_SECONDS_DEFAULT = 30.0

import yaml
from b2sdk.v2 import B2Api, InMemoryAccountInfo
from b2sdk.v2.exception import FileNotPresent
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "mur_c17.yaml"


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"Falta {name} en `.env` (ver `.env.example`).")
    return value


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def b2_api() -> tuple[B2Api, str]:
    key_id = require_env("B2_APPLICATION_KEY_ID")
    app_key = require_env("B2_APPLICATION_KEY")
    bucket_name = os.getenv("B2_BUCKET_NAME", "temperatura-modelos-hcs").strip()
    info = InMemoryAccountInfo()
    api = B2Api(info)
    api.authorize_account("production", key_id, app_key)
    return api, bucket_name


def get_object_info(bucket, key: str):
    try:
        return bucket.get_file_info_by_name(key)
    except FileNotPresent:
        return None


def upload_verified(bucket, payload: bytes, b2_key: str) -> tuple[int, str]:
    size = len(payload)
    checksum = hashlib.sha1(payload).hexdigest()
    uploaded = bucket.upload_bytes(payload, b2_key, content_type="application/netcdf")
    remote_size = getattr(uploaded, "size", getattr(uploaded, "content_length", None))
    remote_sha1 = getattr(uploaded, "content_sha1", None)
    if remote_size != size:
        raise IOError(f"B2 reporto tamano {remote_size}; se esperaban {size} bytes.")
    if remote_sha1 != checksum:
        raise IOError(f"B2 reporto SHA-1 {remote_sha1}; se esperaba {checksum}.")
    return size, checksum


def append_manifest_row(bucket, prefix: str, row: dict) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    key = f"{prefix}upload_{ts}_{uuid4().hex[:8]}.csv"
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(row.keys()))
    writer.writeheader()
    writer.writerow(row)
    data = buf.getvalue().encode("utf-8")
    bucket.upload_bytes(data, key, content_type="text/csv")


def iter_local_files(local_dir: Path):
    if not local_dir.exists():
        return []
    return sorted(p for p in local_dir.glob("*.nc") if p.is_file())


def run(
    files,
    prefix,
    manifest_prefix,
    skip_existing,
    dry_run,
    bucket=None,
    bucket_name="",
    min_age_seconds: float = MIN_AGE_SECONDS_DEFAULT,
    now: float | None = None,
):
    now = _time.time() if now is None else now
    ok = skipped = failed = pending = 0
    for i, path in enumerate(files, start=1):
        b2_key = f"{prefix}{path.name}"

        print(f"[{i}/{len(files)}] {path.name}")
        age = now - path.stat().st_mtime
        if age < min_age_seconds:
            print(f"  PENDIENTE (modificado hace {age:.0f}s, podria seguir escribiendose; se reintenta despues)")
            pending += 1
            continue

        payload = path.read_bytes()
        size = len(payload)
        checksum = hashlib.sha1(payload).hexdigest()

        if skip_existing:
            existing = with_retries(get_object_info, bucket, b2_key)
            if existing is not None:
                remote_size = getattr(existing, "size", getattr(existing, "content_length", None))
                remote_sha1 = getattr(existing, "content_sha1", None)
                if remote_size == size and remote_sha1 == checksum:
                    print(f"  SKIP (ya en B2, coincide): {b2_key}")
                    skipped += 1
                    continue
                failed += 1
                print(
                    f"  CONFLICTO: ya existe en B2 pero NO coincide "
                    f"(local {size}b/{checksum} vs B2 {remote_size}b/{remote_sha1}). "
                    "Se revisa a mano, no se sobreescribe."
                )
                continue

        if dry_run:
            print(f"  (dry-run) subiria -> b2://{bucket_name}/{b2_key} ({size / 1e6:.1f} MB)")
            ok += 1
            continue

        try:
            with_retries(upload_verified, bucket, payload, b2_key)
            print(f"  OK -> b2://{bucket_name}/{b2_key} ({size / 1e6:.1f} MB)")
            try:
                with_retries(
                    append_manifest_row,
                    bucket,
                    manifest_prefix,
                    {
                        "local_path": str(path),
                        "b2_key": b2_key,
                        "bytes": size,
                        "sha1": checksum,
                        "status": "ok",
                        "utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
            except Exception as exc:
                print(f"  (aviso) no se pudo escribir manifiesto: {exc}")
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"  FAIL subiendo {path.name}: {exc}")

    print()
    print(
        f"Resumen: ok={ok} skipped={skipped} failed={failed} pending={pending} "
        f"de {len(files)} archivos"
    )
    return ok, skipped, failed, pending


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Sube a B2 los NetCDF ya descargados en disco")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--local-dir", type=Path, default=ROOT / "data" / "raw" / "mur")
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefijo B2 para los objetos (default: b2.prefix del config, o raw/mur/)",
    )
    parser.add_argument(
        "--manifest-prefix",
        type=str,
        default=None,
        help="Prefijo B2 para los manifiestos (default: b2.manifest_prefix del config, o manifests/uploads/)",
    )
    parser.add_argument("--no-skip-existing", action="store_true", help="Re-sube aunque ya exista en B2")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No sube ni escribe nada; solo consulta B2 (lectura) y muestra que se saltaria/subiria/conflicto",
    )
    parser.add_argument(
        "--min-age-seconds",
        type=float,
        default=MIN_AGE_SECONDS_DEFAULT,
        help="Ignora (por ahora) archivos modificados hace menos de esto; util si download_mur.py "
        "esta corriendo al mismo tiempo sobre la misma carpeta (default: 30s, usa 0 para desactivar)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    prefix = args.prefix or cfg.get("b2", {}).get("prefix", "raw/mur/")
    manifest_prefix = args.manifest_prefix or cfg.get("b2", {}).get("manifest_prefix", "manifests/uploads/")
    skip_existing = not args.no_skip_existing

    files = iter_local_files(args.local_dir)
    if not files:
        print(f"No se encontraron archivos .nc en {args.local_dir}")
        return 0
    print(f"Encontrados {len(files)} archivos locales en {args.local_dir}")

    api, bucket_name = b2_api()
    bucket = api.get_bucket_by_name(bucket_name)
    if args.dry_run:
        print(f"(dry-run) conectado a b2://{bucket_name} en modo solo-lectura, no se sube ni escribe nada\n")

    ok, skipped, failed, pending = run(
        files,
        prefix=prefix,
        manifest_prefix=manifest_prefix,
        skip_existing=skip_existing,
        dry_run=args.dry_run,
        bucket=bucket,
        bucket_name=bucket_name,
        min_age_seconds=args.min_age_seconds,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
