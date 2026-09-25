#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
THREADS="${THREADS:-0}"
mkdir -p data/runs/logs
python historico/prepare.py 2>&1 | tee -a data/runs/logs/prepare.log
IFS=";" read -ra RUN_LIST <<< "${RUNS:-14 42;7 42;14 43;7 43}"
for run in "${RUN_LIST[@]}"; do
  set -- $run
  python historico/train.py --context "$1" --seed "$2" --threads "$THREADS" 2>&1 | tee -a "data/runs/logs/proj$1_seed$2.log"
  python historico/report.py --run "proj$1_seed$2" 2>&1 | tee -a "data/runs/logs/proj$1_seed$2.log"
done
