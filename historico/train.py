from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from b2 import B2
from common import HERE, code_identity, daily_dates, load_config, product_path, resolve, sha256_file
from unet import C17UNet3D, masked_mse

DAYS_IN_YEAR = 366
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


class Windows(Dataset):
    def __init__(self, values, windows, mean, std):
        self.values = values
        self.windows = windows
        self.mean = float(mean)
        self.std = float(std)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        xi, yi = self.windows[i]
        raw_x = self.values[xi]
        raw_y = self.values[yi]
        mx = np.isfinite(raw_x)
        my = np.isfinite(raw_y)
        x = np.where(mx, (raw_x - self.mean) / self.std, 0.0).astype(np.float32)
        y = np.where(my, (raw_y - self.mean) / self.std, 0.0).astype(np.float32)
        return torch.from_numpy(x[None]), torch.from_numpy(y[None]), torch.from_numpy(my[None])


def split_windows(dates: list[date], start: date, end: date, context: int, horizon: int):
    idx = [i for i, d in enumerate(dates) if start <= d <= end]
    if not idx or idx != list(range(idx[0], idx[-1] + 1)):
        raise ValueError(f"split no contiguo o vacio {start}->{end}")
    count = len(idx) - context - horizon + 1
    if count <= 0:
        raise ValueError(f"split corto {start}->{end}")
    o = idx[0]
    return [(np.arange(o + s, o + s + context), np.arange(o + s + context, o + s + context + horizon)) for s in range(count)]


def day_of_year(d: date) -> int:
    return d.timetuple().tm_yday


def trimmed_mean(block: np.ndarray, trim: float) -> np.ndarray:
    cut = int(np.floor(block.shape[0] * trim))
    ordered = np.sort(block, axis=0)
    if cut > 0 and block.shape[0] - 2 * cut >= 1:
        ordered = ordered[cut: block.shape[0] - cut]
    valid = ~np.isnan(ordered)
    count = valid.sum(axis=0)
    total = np.where(valid, ordered, 0.0).sum(axis=0)
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)


def climatology_table(values: np.ndarray, dates: list[date], window: int, trim: float) -> np.ndarray:
    doys = np.array([day_of_year(d) for d in dates])
    table = np.full((DAYS_IN_YEAR + 1,) + values.shape[1:], np.nan, dtype=np.float32)
    for t in range(1, DAYS_IN_YEAR + 1):
        dist = np.abs(doys - t)
        dist = np.minimum(dist, DAYS_IN_YEAR - dist)
        table[t] = trimmed_mean(values[dist <= window], trim)
    return table


def coastal_masks(lat: np.ndarray, lon: np.ndarray, land: np.ndarray, coastal_km: float):
    dlat = float(np.mean(np.abs(np.diff(lat)))) * 111.32
    dlon = float(np.mean(np.abs(np.diff(lon)))) * 111.32 * np.cos(np.deg2rad(float(np.mean(lat))))
    threshold = max(coastal_km, 1.01 * min(dlat, dlon))
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    distance = np.full(lat2d.shape, np.nan)
    ocean = ~land
    if land.any():
        la = lat2d[land]
        lo = lon2d[land]
        oa = lat2d[ocean][:, None]
        oo = lon2d[ocean][:, None]
        distance[ocean] = np.sqrt(((oa - la) * 111.32) ** 2 + ((oo - lo) * 111.32 * np.cos(np.deg2rad(oa))) ** 2).min(axis=1)
    coastal = ocean & np.isfinite(distance) & (distance <= threshold)
    offshore = ocean & np.isfinite(distance) & (distance > threshold)
    return {"domain": ocean, "coastal": coastal, "offshore": offshore}, threshold


def region_metrics(series: dict, target: np.ndarray, regions: dict, sample_ids: np.ndarray | None = None) -> dict:
    out = {}
    t_all = target if sample_ids is None else target[sample_ids]
    for name, arr in series.items():
        p_all = arr if sample_ids is None else arr[sample_ids]
        out[name] = {}
        for region, mask in regions.items():
            rows = []
            for lead in range(t_all.shape[1]):
                p = p_all[:, lead][:, mask]
                t = t_all[:, lead][:, mask]
                valid = np.isfinite(p) & np.isfinite(t)
                if not valid.any():
                    rows.append(None)
                    continue
                e = (p[valid] - t[valid]).astype(np.float64)
                rows.append({"lead_day": lead + 1, "count": int(e.size), "rmse_c": float(np.sqrt(np.mean(e ** 2))), "mae_c": float(np.mean(np.abs(e))), "bias_c": float(np.mean(e))})
            out[name][region] = rows
    return out


def grouped_metrics(series: dict, target: np.ndarray, regions: dict, target_dates: list[list[date]], key) -> dict:
    groups: dict[str, dict[int, list[int]]] = {}
    for s, row in enumerate(target_dates):
        for lead, d in enumerate(row):
            groups.setdefault(key(d), {}).setdefault(lead, []).append(s)
    out = {}
    for g in sorted(groups):
        out[g] = {}
        for name, arr in series.items():
            out[g][name] = {}
            for region, mask in regions.items():
                rows = []
                for lead in range(target.shape[1]):
                    ids = groups[g].get(lead)
                    if not ids:
                        rows.append(None)
                        continue
                    p = arr[ids, lead][:, mask]
                    t = target[ids, lead][:, mask]
                    valid = np.isfinite(p) & np.isfinite(t)
                    if not valid.any():
                        rows.append(None)
                        continue
                    e = (p[valid] - t[valid]).astype(np.float64)
                    rows.append({"lead_day": lead + 1, "samples": len(ids), "count": int(e.size), "rmse_c": float(np.sqrt(np.mean(e ** 2))), "bias_c": float(np.mean(e))})
                out[g][name][region] = rows
    return out


def validate(model, loader, device, std):
    model.eval()
    total, batches = 0.0, 0
    sq = cnt = None
    with torch.no_grad():
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            p = model(x)
            total += float(masked_mse(p, y, m).item())
            batches += 1
            w = m.to(p.dtype)
            e = ((p - y) * std) ** 2
            a = (e * w).sum(dim=(0, 1, 3, 4))
            b = w.sum(dim=(0, 1, 3, 4))
            sq = a if sq is None else sq + a
            cnt = b if cnt is None else cnt + b
    rmse = [float(np.sqrt(float(s) / float(n))) if float(n) > 0 else None for s, n in zip(sq.tolist(), cnt.tolist())]
    return total / batches, rmse


def predict(model, dataset, device, batch_size, mean, std):
    model.eval()
    n = len(dataset)
    x0, y0, _ = dataset[0]
    shape = (n,) + tuple(y0.shape[1:])
    pred = np.empty(shape, dtype=np.float32)
    targ = np.empty(shape, dtype=np.float32)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    k = 0
    with torch.no_grad():
        for x, y, m in loader:
            p = model(x.to(device)).cpu().numpy()[:, 0] * std + mean
            t = y.numpy()[:, 0] * std + mean
            mk = m.numpy()[:, 0]
            p[~mk] = np.nan
            t[~mk] = np.nan
            pred[k:k + len(p)] = p
            targ[k:k + len(p)] = t
            k += len(p)
    return pred, targ


def save_atomic(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def upload_results(b2, cfg: dict, name: str, run_dir: Path) -> None:
    exp = cfg["b2"]["experiment_prefix"] + name
    b2.upload_file(run_dir / "best.pt", f"outputs/checkpoints/{exp}/best.pt")
    b2.upload_file(run_dir / "predictions_validation.nc", f"outputs/predictions/{exp}/predictions_validation.nc")
    b2.upload_file(run_dir / "metrics.json", f"outputs/metrics/{exp}/metrics.json")


def upload_progress(b2, cfg: dict, name: str, payload: dict) -> None:
    if b2 is None:
        return
    try:
        b2.upload_bytes(json.dumps(payload, indent=2).encode("utf-8"), f"outputs/metrics/{cfg['b2']['experiment_prefix']}{name}/progress.json")
    except Exception as exc:
        print(f"aviso: progress no subido: {exc}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    horizon = int(cfg["horizon_days"])
    tr = cfg["training"]
    name = f"proj{args.context}_seed{args.seed}"
    run_dir = resolve(cfg["runs_dir"]) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    b2 = None if args.no_upload else B2(cfg["b2"]["root_prefix"])
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        print(f"{name}: ya terminado ({metrics_path})")
        if b2:
            upload_results(b2, cfg, name, run_dir)
        return 0
    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ppath = product_path(cfg)
    product_sha = sha256_file(ppath)
    with xr.open_dataset(ppath) as ds:
        values = np.asarray(ds["sst"].values, dtype=np.float32)
        dates = [np.datetime64(v, "D").astype(object) for v in ds["time"].values]
        lat = np.asarray(ds["lat"].values)
        lon = np.asarray(ds["lon"].values)
    if dates != daily_dates(dates[0], dates[-1]):
        raise ValueError("producto con fechas no diarias")
    b = cfg["splits"]
    tr_start, tr_end = date.fromisoformat(b["train"]["start"]), date.fromisoformat(b["train"]["end"])
    va_start, va_end = date.fromisoformat(b["validation"]["start"]), date.fromisoformat(b["validation"]["end"])
    if not (tr_end < va_start):
        raise ValueError("train y validacion se solapan")
    windows = {
        "train": split_windows(dates, tr_start, tr_end, args.context, horizon),
        "validation": split_windows(dates, va_start, va_end, args.context, horizon),
    }
    train_idx = [i for i, d in enumerate(dates) if tr_start <= d <= tr_end]
    finite = values[train_idx][np.isfinite(values[train_idx])]
    mean, std = float(finite.mean()), float(finite.std())
    del finite
    datasets = {k: Windows(values, w, mean, std) for k, w in windows.items()}
    print(f"{name}: product={ppath.name} train={len(windows['train'])} validation={len(windows['validation'])} mean={mean:.4f} std={std:.4f} threads={torch.get_num_threads()}", flush=True)

    signature = hashlib.sha256(json.dumps({"cfg": cfg, "context": args.context, "seed": args.seed, "product": product_sha}, sort_keys=True).encode()).hexdigest()
    device = torch.device("cpu")
    model = C17UNet3D(args.context, horizon, int(cfg["model"]["base_channels"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tr["learning_rate"]))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(datasets["train"], batch_size=int(tr["batch_size"]), shuffle=True, generator=generator)
    val_loader = DataLoader(datasets["validation"], batch_size=int(tr["batch_size"]), shuffle=False)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    patience_count = 0
    history = []
    epoch_done = 0
    stopped = False
    last_path = run_dir / "last.pt"
    if last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        if ck["signature"] != signature:
            raise SystemExit(f"{last_path} pertenece a otra config/producto; moverlo antes de relanzar")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        best_state = ck["best_model"]
        best_val = ck["best_val"]
        best_epoch = ck["best_epoch"]
        patience_count = ck["patience_count"]
        history = ck["history"]
        epoch_done = ck["epoch_done"]
        stopped = ck["stopped"]
        generator.set_state(ck["generator"])
        torch.set_rng_state(ck["torch_rng"])
        print(f"{name}: reanuda tras epoca {epoch_done} stopped={stopped}", flush=True)

    for epoch in range(epoch_done + 1, int(tr["epochs"]) + 1):
        if stopped:
            break
        t0 = time.monotonic()
        model.train()
        total, batches = 0.0, 0
        for x, y, m in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(x.to(device)), y.to(device), m.to(device))
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            batches += 1
        val_loss, val_rmse = validate(model, val_loader, device, std)
        seconds = round(time.monotonic() - t0, 3)
        history.append({"epoch": epoch, "train_mse_normalized": total / batches, "val_mse_normalized": val_loss, "val_rmse_c_by_lead": val_rmse, "seconds": seconds})
        print(f"{name} epoch={epoch} train={total / batches:.6f} val={val_loss:.6f} D1={val_rmse[0]:.4f} D7={val_rmse[-1]:.4f} s={seconds:.0f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= int(tr["early_stopping_patience"]):
                stopped = True
        save_atomic({
            "signature": signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "best_model": best_state, "best_val": best_val, "best_epoch": best_epoch,
            "patience_count": patience_count, "history": history, "epoch_done": epoch,
            "stopped": stopped, "generator": generator.get_state(), "torch_rng": torch.get_rng_state(),
            "normalization": {"mean_c": mean, "std_c": std},
        }, last_path)
        upload_progress(b2, cfg, name, {"run": name, "updated_utc": datetime.now(timezone.utc).isoformat(), "best_epoch": best_epoch, "stopped": stopped, "code": code_identity(), "history": history})
        if stopped:
            print(f"{name}: early stopping, mejor epoca {best_epoch}", flush=True)

    model.load_state_dict(best_state)
    save_atomic({"model": best_state, "context": args.context, "horizon": horizon, "base_channels": int(cfg["model"]["base_channels"]), "normalization": {"mean_c": mean, "std_c": std}, "best_epoch": best_epoch, "signature": signature, "config": cfg}, run_dir / "best.pt")

    print(f"{name}: evaluando validacion", flush=True)
    pred, target = predict(model, datasets["validation"], device, int(tr["batch_size"]), mean, std)
    vwin = windows["validation"]
    persistence = np.stack([np.repeat(values[xi][-1:], horizon, axis=0) for xi, _ in vwin])
    clim_table = climatology_table(values[train_idx], [dates[i] for i in train_idx], int(cfg["climatology"]["window_days"]), float(cfg["climatology"]["trim_fraction"]))
    target_dates = [[dates[i] for i in yi] for _, yi in vwin]
    climatology = np.stack([np.stack([clim_table[day_of_year(d)] for d in row]) for row in target_dates])
    land = ~np.isfinite(target).any(axis=(0, 1))
    regions, threshold = coastal_masks(lat, lon, land, float(cfg["coastal_km"]))
    series = {"forecast_model": pred, "persistence": persistence, "climatology": climatology}
    overall = region_metrics(series, target, regions)
    coastal_d1_d5 = {k: (float(np.mean([r["rmse_c"] for r in v["coastal"][:5]])) if all(v["coastal"][:5]) else None) for k, v in overall.items()}

    ds_out = xr.Dataset(
        {
            "forecast_model": (("sample", "lead", "lat", "lon"), pred),
            "target": (("sample", "lead", "lat", "lon"), target),
            "persistence": (("sample", "lead", "lat", "lon"), persistence.astype(np.float32)),
            "climatology": (("sample", "lead", "lat", "lon"), climatology.astype(np.float32)),
            "target_time": (("sample", "lead"), np.array(target_dates, dtype="datetime64[D]")),
        },
        coords={"sample": np.arange(len(vwin)), "lead": np.arange(1, horizon + 1), "lat": lat, "lon": lon},
    )
    pred_path = run_dir / "predictions_validation.nc"
    ds_out.to_netcdf(pred_path.with_suffix(".tmp.nc"), encoding={k: {"zlib": True, "complevel": 4} for k in ("forecast_model", "target", "persistence", "climatology")})
    pred_path.with_suffix(".tmp.nc").replace(pred_path)

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run": name,
        "context_days": args.context,
        "horizon_days": horizon,
        "seed": args.seed,
        "evaluation_split": "validation",
        "splits": cfg["splits"],
        "product": str(ppath),
        "product_sha256": product_sha,
        "code": code_identity(),
        "normalization": {"mean_c": mean, "std_c": std, "fit_on": "train"},
        "window_counts": {k: len(v) for k, v in windows.items()},
        "first_target_start": target_dates[0][0].isoformat(),
        "last_target_start": target_dates[-1][0].isoformat(),
        "coastal_threshold_km": threshold,
        "cells": {k: int(v.sum()) for k, v in regions.items()},
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "timing": {"threads": torch.get_num_threads(), "epoch_seconds_mean": float(np.mean([h["seconds"] for h in history])) if history else None, "wall_seconds_this_process": round(time.monotonic() - started, 3)},
        "history": history,
        "coastal_rmse_d1_d5_mean": coastal_d1_d5,
        "metrics": overall,
        "metrics_by_year": grouped_metrics(series, target, regions, target_dates, lambda d: str(d.year)),
        "metrics_by_season": grouped_metrics(series, target, regions, target_dates, lambda d: SEASONS[d.month]),
        "artifacts": {"best": str(run_dir / "best.pt"), "last": str(last_path), "predictions": str(pred_path)},
    }
    tmp = metrics_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tmp.replace(metrics_path)
    print(f"{name}: coastal D1-D5 {coastal_d1_d5}", flush=True)
    if b2:
        upload_results(b2, cfg, name, run_dir)
    print(f"{name}: listo -> {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
