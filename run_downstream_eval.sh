#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
CSV_PATH="${PROJECT_ROOT}/datasets/Lipophilicity/Lipophilicity.csv"
PRETRAIN_DIR="/hy-tmp/result_flash_1m/run_20260312_214955"
SMOKE_TEST=0
EXECUTION_MODE="all_cpu"
SEED=42

usage() {
  cat <<EOF
Usage: bash run_downstream_eval.sh [--smoke-test] [--all-cpu|--hybrid] [--seed N]

Pipeline:
  1) Convert CSV -> LMDB (multi-process)
  2) Train/eval downstream regression for checkpoints epoch_1,2,5,9
  3) Aggregate RMSE/MAE/R2 summary + performance evolution plot

Execution modes:
  --all-cpu : featurization + downstream training/eval all on CPU
  --hybrid  : featurization on CPU, backbone inference on GPU, downstream head train/val/test on CPU

Smoke test:
  - first 100 rows
  - 2 downstream epochs
  - only epoch_1 checkpoint
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke-test)
      SMOKE_TEST=1
      shift
      ;;
    --all-cpu)
      EXECUTION_MODE="all_cpu"
      shift
      ;;
    --hybrid)
      EXECUTION_MODE="hybrid"
      shift
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERR] Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$CSV_PATH" ]]; then
  echo "[ERR] CSV not found: $CSV_PATH"
  exit 2
fi
if [[ ! -d "$PRETRAIN_DIR" ]]; then
  echo "[ERR] Pretrained directory not found: $PRETRAIN_DIR"
  exit 2
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${PROJECT_ROOT}/downstream_runs/${RUN_TS}"
mkdir -p "$RUN_ROOT"

if [[ "$SMOKE_TEST" -eq 1 ]]; then
  LIMIT=100
  EPOCHS=2
  BATCH_SIZE=32
  CHECKPOINT_EPOCHS=(1)
else
  LIMIT=0
  EPOCHS=40
  BATCH_SIZE=64
  CHECKPOINT_EPOCHS=(1 2 5 9)
fi

LMDB_PATH="${RUN_ROOT}/lipophilicity.lmdb"
WORKERS="$(python - <<'PY'
import os
print(max(1, (os.cpu_count() or 2) - 1))
PY
)"

resolve_checkpoint() {
  local epoch="$1"
  local plain="${PRETRAIN_DIR}/epoch_${epoch}.pth"
  if [[ -f "$plain" ]]; then
    echo "$plain"
    return 0
  fi
  local found
  found="$(find "$PRETRAIN_DIR" -maxdepth 1 -type f -name "*_epoch_${epoch}.pth" | sort | head -n 1)"
  if [[ -n "$found" ]]; then
    echo "$found"
    return 0
  fi
  return 1
}

echo "[INFO] Run root       : $RUN_ROOT"
echo "[INFO] Smoke test     : $SMOKE_TEST"
echo "[INFO] Exec mode      : $EXECUTION_MODE"
echo "[INFO] Pretrain dir   : $PRETRAIN_DIR"
echo "[INFO] LMDB target    : $LMDB_PATH"

echo "[STEP 1/3] Building LMDB (CPU featurization)..."
python "$PROJECT_ROOT/downstream/prepare_lipophilicity_lmdb.py" \
  --csv "$CSV_PATH" \
  --output-lmdb "$LMDB_PATH" \
  --workers "$WORKERS" \
  --limit "$LIMIT"

echo "[STEP 2/3] Running downstream training/eval across checkpoints..."
for epoch in "${CHECKPOINT_EPOCHS[@]}"; do
  ckpt="$(resolve_checkpoint "$epoch")" || {
    echo "[ERR] Could not resolve checkpoint for epoch_${epoch}.pth in ${PRETRAIN_DIR}"
    exit 3
  }
  out_dir="${RUN_ROOT}/checkpoint_epoch_${epoch}"
  mkdir -p "$out_dir"
  echo "[INFO] -> checkpoint epoch ${epoch}: ${ckpt}"
  python "$PROJECT_ROOT/downstream/train_lipophilicity_regression.py" \
    --lmdb "$LMDB_PATH" \
    --checkpoint "$ckpt" \
    --output-dir "$out_dir" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate 1e-4 \
    --weight-decay 1e-5 \
    --seed "$SEED" \
    --execution-mode "$EXECUTION_MODE"
done

echo "[STEP 3/3] Aggregating summary + evolution plot..."
python "$PROJECT_ROOT/downstream/summarize_downstream_results.py" \
  --run-root "$RUN_ROOT"

echo "[OK] Downstream evaluation pipeline completed."
echo "[OK] Results root: $RUN_ROOT"
