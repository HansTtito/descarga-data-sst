# Proyecto-IA-Temperatura

Descarga de datos satelitales MUR SST (NASA/JPL) para pronostico oceanografico.

## Setup

```bash
source entorno/bin/activate
pip install -r requirements.txt
cp .env.example .env
# completa EARTHDATA_TOKEN en .env (cuenta gratuita en https://urs.earthdata.nasa.gov/)
```

## Uso

```bash
python scripts/download_mur.py --start 2002-06-01 --end YYYY-MM-DD
```

Guarda los archivos recortados en `data/raw/mur/` y el manifiesto en
`data/manifests/downloads/`. Correr en `screen`/`tmux` para dejarlo en segundo
plano en un servidor.
