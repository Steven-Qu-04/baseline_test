#!/usr/bin/env bash
set -euo pipefail

# CONFIGURATION
CSV_PATH="${CSV_PATH:-pretraining.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-/hy-tmp/result}"
FULL_LMDB_PATH="${FULL_LMDB_PATH:-/hy-tmp/result/pretraining.lmdb}"
MEDIUM_LMDB_PATH="${MEDIUM_LMDB_PATH:-/hy-tmp/result/ddp_medium_50k.lmdb}"
MODEL_PATH="${MODEL_PATH:-/hy-tmp/result/model_final.pth}"
AUTOTUNE_RESULT_PATH="${AUTOTUNE_RESULT_PATH:-/hy-tmp/result/autotune_result.json}"
PLATFORM_SUMMARY_PATH="${PLATFORM_SUMMARY_PATH:-/hy-tmp/result/platform_summary.json}"
GPUS="${GPUS:-4}"
CPU_RESERVE_THREADS="${CPU_RESERVE_THREADS:-4}"
PREPROCESS_WORKER_COUNT="${PREPROCESS_WORKER_COUNT:-0}"
TASK_QUEUE_MAXSIZE="${TASK_QUEUE_MAXSIZE:-0}"
RESULT_QUEUE_MAXSIZE="${RESULT_QUEUE_MAXSIZE:-0}"
WRITER_BATCH_SIZE="${WRITER_BATCH_SIZE:-64}"
MEDIUM_ROWS="${MEDIUM_ROWS:-50000}"
FORCE_REBUILD_MEDIUM="${FORCE_REBUILD_MEDIUM:-0}"
BATCH_PER_GPU="${BATCH_PER_GPU:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-1e-4}"
SCALED_LEARNING_RATE="${SCALED_LEARNING_RATE:-0}"
MASK_RATIO="${MASK_RATIO:-0.1}"
NUM_MASKED_VIEWS="${NUM_MASKED_VIEWS:-6}"
EPOCHS="${EPOCHS:-3}"
TEMPERATURE="${TEMPERATURE:-0.07}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
LAP_PE_DIM="${LAP_PE_DIM:-8}"
STRESS_STEPS="${STRESS_STEPS:-1000}"
AUTOTUNE_BATCH_MIN="${AUTOTUNE_BATCH_MIN:-2}"
AUTOTUNE_BATCH_MAX="${AUTOTUNE_BATCH_MAX:-64}"
AUTOTUNE_STEPS="${AUTOTUNE_STEPS:-24}"
AUTOTUNE_WARMUP_STEPS="${AUTOTUNE_WARMUP_STEPS:-4}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-8}"
BASELINE_GRAD_ACCUM_STEPS="${BASELINE_GRAD_ACCUM_STEPS:-4}"
MASTER_PORT_MIN="${MASTER_PORT_MIN:-20000}"
MASTER_PORT_MAX="${MASTER_PORT_MAX:-65000}"
MASTER_PORT="${MASTER_PORT:-$(python - <<PY
import random
print(random.randint(${MASTER_PORT_MIN}, ${MASTER_PORT_MAX}))
PY
)}"

echo "Using MASTER_PORT=${MASTER_PORT}"

python check_platform.py --output-dir "${OUTPUT_DIR}" --summary-path "${PLATFORM_SUMMARY_PATH}"

if [[ "${FORCE_REBUILD_MEDIUM}" == "1" || ! -f "${MEDIUM_LMDB_PATH}" ]]; then
  python -m mol_gtn.preprocess \
    --csv "${CSV_PATH}" \
    --medium-lmdb "${MEDIUM_LMDB_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --workers "${PREPROCESS_WORKER_COUNT}" \
    --cpu-reserve-threads "${CPU_RESERVE_THREADS}" \
    --task-queue-maxsize "${TASK_QUEUE_MAXSIZE}" \
    --result-queue-maxsize "${RESULT_QUEUE_MAXSIZE}" \
    --writer-batch-size "${WRITER_BATCH_SIZE}" \
    --lap-pe-dim "${LAP_PE_DIM}" \
    --medium-test \
    --medium-rows "${MEDIUM_ROWS}"
fi

COMMON_TRAIN_ARGS=(
  --ddp
  --backend nccl
  --medium-lmdb "${MEDIUM_LMDB_PATH}"
  --lmdb "${FULL_LMDB_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --model-path "${MODEL_PATH}"
  --autotune-result-path "${AUTOTUNE_RESULT_PATH}"
  --mask-ratio "${MASK_RATIO}"
  --num-masked-views "${NUM_MASKED_VIEWS}"
  --temperature "${TEMPERATURE}"
  --hidden-dim "${HIDDEN_DIM}"
  --lap-pe-dim "${LAP_PE_DIM}"
  --baseline-batch-size "${BASELINE_BATCH_SIZE}"
  --baseline-grad-accum-steps "${BASELINE_GRAD_ACCUM_STEPS}"
)

torchrun --nproc_per_node="${GPUS}" --master_port="${MASTER_PORT}" -m mol_gtn.train \
  "${COMMON_TRAIN_ARGS[@]}" \
  --medium-test \
  --autotune \
  --batch-per-gpu "${BATCH_PER_GPU}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --learning-rate "${BASE_LEARNING_RATE}" \
  --autotune-batch-min "${AUTOTUNE_BATCH_MIN}" \
  --autotune-batch-max "${AUTOTUNE_BATCH_MAX}" \
  --autotune-steps "${AUTOTUNE_STEPS}" \
  --autotune-warmup-steps "${AUTOTUNE_WARMUP_STEPS}"

readarray -t TUNED_VALUES < <(python - <<PY
import json
from pathlib import Path
payload = json.loads(Path("${AUTOTUNE_RESULT_PATH}").read_text())
print(payload["batch_per_gpu"])
print(payload["grad_accum_steps"])
print(payload["scaled_learning_rate"])
PY
)
TUNED_BATCH_PER_GPU="${TUNED_VALUES[0]}"
TUNED_GRAD_ACCUM_STEPS="${TUNED_VALUES[1]}"
TUNED_SCALED_LR="${TUNED_VALUES[2]}"

if [[ "${SCALED_LEARNING_RATE}" != "0" ]]; then
  TUNED_SCALED_LR="${SCALED_LEARNING_RATE}"
fi

echo "Final ratio batch_per_gpu:grad_accum_steps=${TUNED_BATCH_PER_GPU}:${TUNED_GRAD_ACCUM_STEPS}"
echo "MASTER_PORT=${MASTER_PORT}"

torchrun --nproc_per_node="${GPUS}" --master_port="${MASTER_PORT}" -m mol_gtn.train \
  "${COMMON_TRAIN_ARGS[@]}" \
  --medium-test \
  --batch-per-gpu "${TUNED_BATCH_PER_GPU}" \
  --grad-accum-steps "${TUNED_GRAD_ACCUM_STEPS}" \
  --scaled-learning-rate "${TUNED_SCALED_LR}" \
  --epochs 1 \
  --max-steps "${STRESS_STEPS}" \
  --skip-save

echo "Starting full training with batch_per_gpu=${TUNED_BATCH_PER_GPU} grad_accum_steps=${TUNED_GRAD_ACCUM_STEPS} scaled_lr=${TUNED_SCALED_LR}"

torchrun --nproc_per_node="${GPUS}" --master_port="${MASTER_PORT}" -m mol_gtn.train \
  "${COMMON_TRAIN_ARGS[@]}" \
  --batch-per-gpu "${TUNED_BATCH_PER_GPU}" \
  --grad-accum-steps "${TUNED_GRAD_ACCUM_STEPS}" \
  --scaled-learning-rate "${TUNED_SCALED_LR}" \
  --learning-rate "${BASE_LEARNING_RATE}" \
  --epochs "${EPOCHS}"
