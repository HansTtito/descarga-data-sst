from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from common import ROOT


class B2:
    def __init__(self, root_prefix: str):
        from b2sdk.v2 import B2Api, InMemoryAccountInfo

        load_dotenv(ROOT / ".env")
        key_id = os.getenv("B2_APPLICATION_KEY_ID", "").strip()
        key = os.getenv("B2_APPLICATION_KEY", "").strip()
        if not key_id or not key:
            raise SystemExit("faltan B2_APPLICATION_KEY_ID / B2_APPLICATION_KEY en .env")
        api = B2Api(InMemoryAccountInfo())
        api.authorize_account("production", key_id, key)
        self.bucket_name = os.getenv("B2_BUCKET_NAME", "temperatura-modelos-hcs").strip()
        self.bucket = api.get_bucket_by_name(self.bucket_name)
        self.root = (os.getenv("B2_ROOT_PREFIX", "").strip() or root_prefix).strip("/")

    def key(self, key: str) -> str:
        return f"{self.root}/{key.lstrip('/')}" if self.root else key.lstrip("/")

    def upload_file(self, path: Path, key: str) -> str:
        full = self.key(key)
        info = self.bucket.upload_local_file(local_file=str(path), file_name=full)
        size = getattr(info, "size", None)
        if size is not None and size != Path(path).stat().st_size:
            raise IOError(f"B2 {full}: {size} bytes, local {Path(path).stat().st_size}")
        print(f"b2://{self.bucket_name}/{full}", flush=True)
        return full

    def upload_bytes(self, payload: bytes, key: str) -> str:
        full = self.key(key)
        self.bucket.upload_bytes(payload, full)
        return full
