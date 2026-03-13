#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
CSV_PATH="${PROJECT_ROOT}/datasets/Lipophilicity/Lipophilicity.csv"
PRETRAIN_DIR="/hy-tmp/result_flash_1m/run_20260312_214955"
SMOKE_TEST=0
SEED=42
DEVICE="cuda"

usage() {
  cat <<EOF
Usage: bash run_downstream_eval_mlp_twostage.sh [--smoke-test] [--seed N] [--device cuda|cpu]

Pipeline:
  1) CPU LMDB conversion
  2) Two-stage MLP training/eval on checkpoints epoch_1,2,5,9
  3) Summary table + performance evolution plot

Smoke test:
  - first 100 rows
  - warmup=1, total epochs=2 (forces stage transition)
  - only epoch_1 checkpoint
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke-test)
      SMOKE_TEST=1
      shift
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
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
RUN_ROOT="${PROJECT_ROOT}/downstream_runs_mlp_twostage/${RUN_TS}"
mkdir -p "$RUN_ROOT"

if [[ "$SMOKE_TEST" -eq 1 ]]; then
  LIMIT=100
  EPOCHS=2
  WARMUP_EPOCHS=1
  EARLY_STOP_PATIENCE=2
  BATCH_SIZE=32
  CHECKPOINT_EPOCHS=(1)
else
  LIMIT=0
  EPOCHS=50
  WARMUP_EPOCHS=10
  EARLY_STOP_PATIENCE=8
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
echo "[INFO] Device         : $DEVICE"
echo "[INFO] Pretrain dir   : $PRETRAIN_DIR"
echo "[INFO] LMDB target    : $LMDB_PATH"

echo "[STEP 1/3] Building LMDB (CPU featurization)..."
python "$PROJECT_ROOT/downstream/prepare_lipophilicity_lmdb.py" \
  --csv "$CSV_PATH" \
  --output-lmdb "$LMDB_PATH" \
  --workers "$WORKERS" \
  --limit "$LIMIT"

echo "[STEP 2/3] Running two-stage MLP downstream training across checkpoints..."
for epoch in "${CHECKPOINT_EPOCHS[@]}"; do
  ckpt="$(resolve_checkpoint "$epoch")" || {
    echo "[ERR] Could not resolve checkpoint for epoch_${epoch}.pth in ${PRETRAIN_DIR}"
    exit 3
  }

  out_dir="${RUN_ROOT}/checkpoint_epoch_${epoch}"
  mkdir -p "$out_dir"
  echo "[INFO] -> checkpoint epoch ${epoch}: ${ckpt}"

  python "$PROJECT_ROOT/downstream/train_lipophilicity_regression_mlp_twostage.py" \
    --lmdb "$LMDB_PATH" \
    --checkpoint "$ckpt" \
    --output-dir "$out_dir" \
    --warmup-epochs "$WARMUP_EPOCHS" \
    --epochs "$EPOCHS" \
    --warmup-lr 1e-3 \
    --finetune-backbone-lr 1e-5 \
    --finetune-head-lr 5e-4 \
    --early-stop-patience "$EARLY_STOP_PATIENCE" \
    --early-stop-min-delta 1e-4 \
    --batch-size "$BATCH_SIZE" \
    --weight-decay 1e-5 \
    --seed "$SEED" \
    --device "$DEVICE"
done

echo "[STEP 3/3] Aggregating summary + evolution plot..."
python "$PROJECT_ROOT/downstream/summarize_downstream_results_mlp_twostage.py" \
  --run-root "$RUN_ROOT"

echo "[OK] Two-stage MLP downstream evaluation pipeline completed."
echo "[OK] Results root: $RUN_ROOT"
