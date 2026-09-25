
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from b2 import B2
from common import HERE, load_config, product_path, resolve
from train import coastal_masks

MONTHS = ("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")
SST_CMAP = "YlGnBu_r"
ERR_CMAP = "RdBu_r"


def fmt(d: date) -> str:
    return f"{d.day:02d}-{MONTHS[d.month - 1]}-{d.year}"


def to_date(value) -> date:
    return np.datetime64(value, "D").astype(object)


def rmse(p: np.ndarray, t: np.ndarray, mask: np.ndarray) -> float | None:
    a, b = p[mask], t[mask]
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[ok] - b[ok]).astype(np.float64) ** 2))) if ok.any() else None


def nan_if_none(v: float | None) -> float:
    return np.nan if v is None else v


def png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", facecolor="white")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def default_emissions(emissions: list[date], coastal_gain: np.ndarray) -> list[tuple[date, str]]:
    years = sorted({d.year for d in emissions})
    year = years[1] if len(years) > 1 else years[0]
    picks: list[tuple[date, str]] = []
    for month, season in ((1, "verano"), (4, "otoño"), (7, "invierno"), (10, "primavera")):
        wanted = date(year, month, 15)
        near = min(emissions, key=lambda d: abs((d - wanted).days))
        if abs((near - wanted).days) <= 15 and near not in [p for p, _ in picks]:
            picks.append((near, f"{season} {year}"))
    if np.isfinite(coastal_gain).any():
        worst = emissions[int(np.nanargmin(coastal_gain))]
        if worst not in [p for p, _ in picks]:
            picks.append((worst, "día en que el modelo más pierde contra persistencia en costa D1"))
    return picks


def geo_ticks(ax, left: bool, bottom: bool) -> None:
    lats = [-38, -34, -30, -26, -22]
    lons = [-78, -74, -70]
    ax.set_yticks(lats if left else [])
    ax.set_yticklabels([f"{-v}°S" for v in lats] if left else [], fontsize=7)
    ax.set_xticks(lons if bottom else [])
    ax.set_xticklabels([f"{-v}°W" for v in lons] if bottom else [], fontsize=7)


def input_figure(sst: np.ndarray, days: list[date], extent, vmin: float, vmax: float) -> str:
    import matplotlib.pyplot as plt

    cols = 7
    rows = int(np.ceil(len(days) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.9 * cols, 3.1 * rows), squeeze=False)
    image = None
    for k, ax in enumerate(axes.ravel()):
        geo_ticks(ax, k % cols == 0, k // cols == rows - 1)
        if k >= len(days):
            ax.axis("off")
            continue
        image = ax.imshow(sst[k], origin="lower", extent=extent, cmap=SST_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
        last = k == len(days) - 1
        ax.set_title(fmt(days[k]) + ("\n(emisión)" if last else ""), fontsize=9, fontweight="bold" if last else "normal")
        if last:
            for s in ax.spines.values():
                s.set_linewidth(2.5)
    fig.colorbar(image, ax=axes, shrink=0.8, label="SST (°C)")
    out = png(fig)
    plt.close(fig)
    return out


def output_figure(obs, model, pers, targets: list[date], extent, vmin, vmax, elim) -> str:
    import matplotlib.pyplot as plt

    rows = (("MUR observado", obs, SST_CMAP, vmin, vmax),
            ("Pronóstico modelo", model, SST_CMAP, vmin, vmax),
            ("Error modelo", model - obs, ERR_CMAP, -elim, elim),
            ("Error persistencia", pers - obs, ERR_CMAP, -elim, elim))
    horizon = obs.shape[0]
    fig, axes = plt.subplots(len(rows), horizon, figsize=(1.9 * horizon, 3.0 * len(rows)), squeeze=False)
    images = {}
    for r, (label, values, cmap, lo, hi) in enumerate(rows):
        for c in range(horizon):
            ax = axes[r, c]
            images[r] = ax.imshow(values[c], origin="lower", extent=extent, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
            geo_ticks(ax, c == 0, r == len(rows) - 1)
            if r == 0:
                ax.set_title(f"D{c + 1}\n{fmt(targets[c])}", fontsize=9)
            if c == 0:
                ax.set_ylabel(label, fontsize=10, fontweight="bold")
    fig.colorbar(images[1], ax=axes[:2].ravel().tolist(), shrink=0.85, label="SST (°C)")
    fig.colorbar(images[3], ax=axes[2:].ravel().tolist(), shrink=0.85, label="Error (°C, + = más cálido que MUR)")
    out = png(fig)
    plt.close(fig)
    return out


def timeline_svg(emissions: list[date], model: np.ndarray, pers: np.ndarray, marks: list[date]) -> str:
    w, h, left, right, top, bottom = 1000, 280, 48, 16, 16, 30
    finite = np.concatenate([model[np.isfinite(model)], pers[np.isfinite(pers)]])
    ymax = float(np.nanpercentile(finite, 99.5)) * 1.05 if finite.size else 1.0
    n = len(emissions)

    def x(i):
        return left + (w - left - right) * (i / max(n - 1, 1))

    def y(v):
        return top + (h - top - bottom) * (1 - min(v, ymax) / ymax)

    def path(values):
        parts, pen = [], "M"
        for i, v in enumerate(values):
            if not np.isfinite(v):
                pen = "M"
                continue
            parts.append(f"{pen}{x(i):.1f},{y(v):.1f}")
            pen = "L"
        return " ".join(parts)

    grid = []
    for k in range(5):
        v = ymax * k / 4
        grid.append(f'<line x1="{left}" x2="{w - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>'
                    f'<text x="{left - 6}" y="{y(v) + 4:.1f}" class="tick" text-anchor="end">{v:.2f}</text>')
    for i, d in enumerate(emissions):
        if d.month == 1 and d.day == 1 or i == 0:
            grid.append(f'<text x="{x(i):.1f}" y="{h - 8}" class="tick" text-anchor="start">{d.year}</text>')
    for d in marks:
        if d in emissions:
            i = emissions.index(d)
            grid.append(f'<line x1="{x(i):.1f}" x2="{x(i):.1f}" y1="{top}" y2="{h - bottom}" class="mark"/>')
    data = json.dumps({"d": [d.isoformat() for d in emissions],
                       "m": [None if not np.isfinite(v) else round(float(v), 4) for v in model],
                       "p": [None if not np.isfinite(v) else round(float(v), 4) for v in pers]})
    return f"""
<div class="chart" id="timeline" data-left="{left}" data-right="{right}" data-w="{w}">
  <div class="legend"><span><i class="key s1"></i>Modelo</span><span><i class="key s2"></i>Persistencia</span><span><i class="key mk"></i>Fechas con ficha</span></div>
  <svg viewBox="0 0 {w} {h}" role="img" aria-label="RMSE costero D1 por fecha de emisión">
    {''.join(grid)}
    <path d="{path(pers)}" class="line s2"/>
    <path d="{path(model)}" class="line s1"/>
    <line class="cross" y1="{top}" y2="{h - bottom}" x1="-10" x2="-10"/>
    <rect x="{left}" y="{top}" width="{w - left - right}" height="{h - top - bottom}" fill="transparent" class="hit"/>
  </svg>
  <div class="tip" hidden></div>
  <script type="application/json" class="series">{data}</script>
</div>"""


def metrics_table(metrics: dict) -> str:
    names = [("forecast_model", "Modelo"), ("persistence", "Persistencia"), ("climatology", "Climatología")]
    head = "".join(f"<th>D{k}</th>" for k in range(1, 8))
    body = []
    for region, label in (("coastal", "Costa ≤50 km"), ("offshore", "Mar abierto"), ("domain", "Dominio")):
        for key, name in names:
            rows = metrics.get(key, {}).get(region) or []
            cells = "".join(f"<td>{r['rmse_c']:.3f}</td>" if r else "<td>—</td>" for r in rows)
            body.append(f"<tr><th>{label}</th><td class='name'>{name}</td>{cells}</tr>")
    return f"<table><thead><tr><th>Región</th><th>Serie</th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


CSS = """
:root{--surface:#fcfcfb;--panel:#ffffff;--text:#0b0b0b;--text2:#52514e;--line:#e4e3de;--s1:#2a78d6;--s2:#eb6834;--mk:#8a8984}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;--panel:#232322;--text:#ffffff;--text2:#c3c2b7;--line:#3a3a38;--s1:#3987e5;--s2:#d95926;--mk:#8a8984}}
:root[data-theme="dark"]{--surface:#1a1a19;--panel:#232322;--text:#ffffff;--text2:#c3c2b7;--line:#3a3a38;--s1:#3987e5;--s2:#d95926;--mk:#8a8984}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--text);font:15px/1.5 system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:36px 0 8px}h3{font-size:16px;margin:18px 0 6px}
.sub,.note{color:var(--text2)}.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:14px 0}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px 24px;margin:8px 0}.facts div b{display:block;font-size:12px;color:var(--text2);font-weight:600;text-transform:uppercase;letter-spacing:.03em}
.flow{display:flex;flex-wrap:wrap;gap:8px;align-items:center;font-weight:600;margin:6px 0 10px}.flow span{border:1px solid var(--line);border-radius:6px;padding:4px 10px}.flow i{color:var(--text2);font-style:normal}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td.name{text-align:left}thead th{color:var(--text2);font-weight:600}
td.win{font-weight:700}.scroll{overflow-x:auto}img{max-width:100%;height:auto;border-radius:6px;background:#fff;display:block}
.chart{position:relative}.chart svg{width:100%;height:auto;display:block}.grid{stroke:var(--line);stroke-width:1}.tick{fill:var(--text2);font-size:11px}
.line{fill:none;stroke-width:2;stroke-linejoin:round}.line.s1{stroke:var(--s1)}.line.s2{stroke:var(--s2);stroke-opacity:.85}.mark{stroke:var(--mk);stroke-dasharray:3 3}
.cross{stroke:var(--text2);stroke-width:1}.legend{display:flex;gap:18px;font-size:13px;color:var(--text2);margin-bottom:4px}.key{display:inline-block;width:16px;height:0;border-top:2px solid;vertical-align:middle;margin-right:6px}
.key.s1{border-color:var(--s1)}.key.s2{border-color:var(--s2)}.key.mk{border-top:2px dashed var(--mk)}
.tip{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:6px 10px;font-size:13px;box-shadow:0 2px 8px rgba(0,0,0,.15);white-space:nowrap}
.tip b{font-variant-numeric:tabular-nums}.tip .k{display:inline-block;width:12px;border-top:2px solid;vertical-align:middle;margin-right:6px}
"""

JS = """
document.querySelectorAll('.chart').forEach(function(c){
  var s=JSON.parse(c.querySelector('.series').textContent),svg=c.querySelector('svg'),hit=c.querySelector('.hit'),
      cross=c.querySelector('.cross'),tip=c.querySelector('.tip'),L=+c.dataset.left,R=+c.dataset.right,W=+c.dataset.w,n=s.d.length;
  function row(color,val,label){var d=document.createElement('div'),k=document.createElement('i'),b=document.createElement('b');
    k.className='k';k.style.borderColor=color;b.textContent=val==null?'—':val.toFixed(3)+' °C';d.append(k,b,document.createTextNode(' '+label));return d;}
  hit.addEventListener('pointermove',function(e){var box=svg.getBoundingClientRect(),vx=(e.clientX-box.left)*W/box.width,
      i=Math.max(0,Math.min(n-1,Math.round((vx-L)/(W-L-R)*(n-1)))),px=L+(W-L-R)*i/Math.max(n-1,1);
    cross.setAttribute('x1',px);cross.setAttribute('x2',px);tip.hidden=false;tip.replaceChildren();
    var h=document.createElement('div');h.textContent='Emisión '+s.d[i];tip.append(h,
      row(getComputedStyle(c).getPropertyValue('--s1'),s.m[i],'modelo'),row(getComputedStyle(c).getPropertyValue('--s2'),s.p[i],'persistencia'));
    var x=px*box.width/W;tip.style.left=Math.min(x+12,box.width-tip.offsetWidth)+'px';tip.style.top='28px';});
  hit.addEventListener('pointerleave',function(){tip.hidden=true;cross.setAttribute('x1',-10);cross.setAttribute('x2',-10);});
});
"""


def card(title: str, emission: date, context_days: list[date], targets: list[date], inputs_png: str, outputs_png: str, rows: list[dict]) -> str:
    table = []
    for k, r in enumerate(rows):
        def cell(a, b):
            if a is None:
                return "<td>—</td>"
            return f"<td class='{'win' if b is not None and a < b else ''}'>{a:.3f}</td>"
        table.append(f"<tr><th>D{k + 1}</th><td class='name'>{fmt(targets[k])}</td>"
                     f"{cell(r['cm'], r['cp'])}{cell(r['cp'], r['cm'])}{cell(r['dm'], r['dp'])}{cell(r['dp'], r['dm'])}</tr>")
    return f"""
<section class="panel">
  <h3>Emisión {fmt(emission)} — {html.escape(title)}</h3>
  <div class="flow"><span>Entra: MUR {fmt(context_days[0])} → {fmt(context_days[-1])} ({len(context_days)} días)</span><i>→</i>
  <span>Sale: {fmt(targets[0])} (D1) … {fmt(targets[-1])} (D{len(targets)})</span></div>
  <p class="note">Lo que vio el modelo. El último día (borde grueso) es la fecha de emisión y es también lo que repite la persistencia.</p>
  <img src="{inputs_png}" alt="SST observada de entrada">
  <p class="note">Lo que predijo, fecha por fecha. Error = pronóstico − MUR: rojo, más cálido que lo observado; azul, más frío.</p>
  <img src="{outputs_png}" alt="Pronóstico, observado y errores por fecha objetivo">
  <div class="scroll"><table><thead><tr><th>Lead</th><th>Fecha objetivo</th><th>Costa modelo</th><th>Costa persist.</th><th>Dominio modelo</th><th>Dominio persist.</th></tr></thead>
  <tbody>{''.join(table)}</tbody></table></div>
  <p class="note">RMSE en °C de ese día; en negrita, el menor de cada par.</p>
</section>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--run", required=True, help="p. ej. proj14_seed42")
    parser.add_argument("--dates", nargs="*", default=None, help="fechas de emisión YYYY-MM-DD (último día observado)")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")

    cfg = load_config(args.config)
    run_dir = resolve(cfg["runs_dir"]) / args.run
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    context = int(metrics["context_days"])
    pred = xr.open_dataset(run_dir / "predictions_validation.nc")
    product = xr.open_dataset(product_path(cfg))
    lat, lon = pred["lat"].values, pred["lon"].values
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    target_times = [[to_date(v) for v in row] for row in pred["target_time"].values]
    emissions = [row[0] - timedelta(days=1) for row in target_times]

    target_d1 = pred["target"].isel(lead=0).values
    land = ~np.isfinite(pred["target"].isel(sample=0).values).any(axis=0)
    regions, threshold = coastal_masks(lat, lon, land, float(cfg["coastal_km"]))
    coast = regions["coastal"]
    model_d1 = pred["forecast_model"].isel(lead=0).values
    pers_d1 = pred["persistence"].isel(lead=0).values
    cm = np.array([nan_if_none(rmse(model_d1[s], target_d1[s], coast)) for s in range(len(emissions))])
    cp = np.array([nan_if_none(rmse(pers_d1[s], target_d1[s], coast)) for s in range(len(emissions))])
    del target_d1, model_d1, pers_d1

    if args.dates:
        picks = []
        for text in args.dates:
            d = date.fromisoformat(text)
            if d not in emissions:
                raise SystemExit(f"{d} no es fecha de emisión de validación ({emissions[0]} → {emissions[-1]})")
            picks.append((d, "fecha elegida"))
    else:
        picks = default_emissions(emissions, cp - cm)

    product_dates = [to_date(v) for v in product["time"].values]
    cards = []
    for emission, title in picks:
        s = emissions.index(emission)
        e_idx = product_dates.index(emission)
        context_days = product_dates[e_idx - context + 1: e_idx + 1]
        inputs = product["sst"].isel(time=slice(e_idx - context + 1, e_idx + 1)).values
        obs = pred["target"].isel(sample=s).values
        model = pred["forecast_model"].isel(sample=s).values
        pers = pred["persistence"].isel(sample=s).values
        vals = np.concatenate([inputs[np.isfinite(inputs)], obs[np.isfinite(obs)], model[np.isfinite(model)]])
        vmin, vmax = float(np.percentile(vals, 1)), float(np.percentile(vals, 99))
        errs = np.abs(np.concatenate([(model - obs).ravel(), (pers - obs).ravel()]))
        elim = float(np.nanpercentile(errs, 98)) or 1.0
        rows = [{"cm": rmse(model[k], obs[k], coast), "cp": rmse(pers[k], obs[k], coast),
                 "dm": rmse(model[k], obs[k], regions["domain"]), "dp": rmse(pers[k], obs[k], regions["domain"])}
                for k in range(obs.shape[0])]
        cards.append(card(title, emission, context_days, target_times[s],
                          input_figure(inputs, context_days, extent, vmin, vmax),
                          output_figure(obs, model, pers, target_times[s], extent, vmin, vmax, elim), rows))
        print(f"ficha {emission} lista", flush=True)

    splits = metrics["splits"]
    wins = int(np.sum(cm < cp))
    valid = int(np.sum(np.isfinite(cm) & np.isfinite(cp)))
    doc = f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pronósticos {html.escape(args.run)}</title><style>{CSS}</style></head><body><main>
<h1>Pronósticos SST C17 — {html.escape(args.run)}</h1>
<p class="sub">U-Net residual con proyección temporal · {context} días de entrada → {metrics['horizon_days']} días de salida · semilla {metrics['seed']}</p>
<div class="panel facts">
  <div><b>Aprendió con</b>{splits['train']['start']} → {splits['train']['end']}</div>
  <div><b>Se evalúa en</b>{splits['validation']['start']} → {splits['validation']['end']} ({metrics['window_counts']['validation']} fechas de emisión)</div>
  <div><b>Checkpoint</b>época {metrics['best_epoch']} de {metrics['epochs_run']}</div>
  <div><b>Costa</b>franja ≤{threshold:.0f} km, {int(coast.sum())} celdas</div>
</div>
<p class="note">Estas fechas de validación también decidieron el early stopping. Sirven para ver cómo se comporta el modelo, pero no son un test independiente.</p>

<h2>Cómo leer una ficha</h2>
<p>Cada ficha es un pronóstico concreto. Se toma una <b>fecha de emisión</b> (el último día con MUR observado),
el modelo recibe esos {context} días y predice los {metrics['horizon_days']} días siguientes. D1 es el día después de la emisión y D{metrics['horizon_days']} el último.
Después se compara cada día con lo que MUR observó y con la persistencia, que repite el último día observado.</p>

<h2>Cada día de validación</h2>
<p>RMSE costero de D1 para cada fecha de emisión. En {wins} de {valid} fechas el modelo queda por debajo de la persistencia.</p>
<div class="panel">{timeline_svg(emissions, cm, cp, [p for p, _ in picks])}</div>

<h2>Fichas de pronóstico</h2>
{''.join(cards)}

<h2>Resumen de toda la validación</h2>
<p class="note">RMSE medio en °C sobre las {metrics['window_counts']['validation']} fechas de emisión.</p>
<div class="panel scroll">{metrics_table(metrics['metrics'])}</div>
<p class="note">Producto {html.escape(Path(metrics['product']).name)} · sha256 {metrics['product_sha256'][:12]} · código {html.escape(str(metrics['code'].get('git_head') or '')[:12])}</p>
</main><script>{JS}</script></body></html>"""

    out = run_dir / "report.html"
    out.write_text(doc, encoding="utf-8")
    print(f"reporte -> {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    if not args.no_upload:
        B2(cfg["b2"]["root_prefix"]).upload_file(out, f"outputs/reports/{cfg['b2']['experiment_prefix']}{args.run}/report.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
