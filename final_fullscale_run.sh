#!/usr/bin/env bash
set -euo pipefail

# CONFIGURATION
CSV_PATH="${CSV_PATH:-pretraining.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-/hy-tmp/result}"
FULL_LMDB_PATH="${FULL_LMDB_PATH:-/hy-tmp/result/pretraining.lmdb}"
MEDIUM_LMDB_PATH="${MEDIUM_LMDB_PATH:-/hy-tmp/result/ddp_medium_50k.lmdb}"
MODEL_PATH="${MODEL_PATH:-}"
PLATFORM_SUMMARY_PATH="${PLATFORM_SUMMARY_PATH:-}"
PROJECT_LOG_PATH="${PROJECT_LOG_PATH:-}"
UPLOAD_SCRIPT_PATH="${UPLOAD_SCRIPT_PATH:-/root/upload.sh}"
AUTO_UPLOAD_AND_SHUTDOWN="${AUTO_UPLOAD_AND_SHUTDOWN:-0}"
OSS_TARGET_DIR="${OSS_TARGET_DIR:-oss://backup/}"
SHUTDOWN_CMD="${SHUTDOWN_CMD:-shutdown}"
GPUS="${GPUS:-4}"
CPU_RESERVE_THREADS="${CPU_RESERVE_THREADS:-4}"
PREPROCESS_WORKER_COUNT="${PREPROCESS_WORKER_COUNT:-96}"
TASK_QUEUE_MAXSIZE="${TASK_QUEUE_MAXSIZE:-192}"
RESULT_QUEUE_MAXSIZE="${RESULT_QUEUE_MAXSIZE:-192}"
WRITER_BATCH_SIZE="${WRITER_BATCH_SIZE:-256}"
LAP_PE_DIM="${LAP_PE_DIM:-8}"
MASK_RATIO="${MASK_RATIO:-0.1}"
NUM_MASKED_VIEWS="${NUM_MASKED_VIEWS:-6}"
DATA_MODE="${DATA_MODE:-auto}"
AUDIT_REPORT_PATH="${AUDIT_REPORT_PATH:-}"
AUDIT_SAMPLE_COUNT="${AUDIT_SAMPLE_COUNT:-512}"
RESUME_FROM="${RESUME_FROM:-}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-6}"
TEMPERATURE="${TEMPERATURE:-0.07}"
BATCH_PER_GPU="${BATCH_PER_GPU:-32}"
MAX_NODES_PER_BATCH="${MAX_NODES_PER_BATCH:-2048}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
SCALED_LEARNING_RATE="${SCALED_LEARNING_RATE:-0.0003}"
EPOCHS="${EPOCHS:-20}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-}"
MASTER_PORT_MIN="${MASTER_PORT_MIN:-20000}"
MASTER_PORT_MAX="${MASTER_PORT_MAX:-65000}"
MASTER_PORT="${MASTER_PORT:-$(python - <<PY
import random
print(random.randint(${MASTER_PORT_MIN}, ${MASTER_PORT_MAX}))
PY
)}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
RUN_MEDIUM_STRESS_TEST="${RUN_MEDIUM_STRESS_TEST:-1}"
STRESS_STEPS="${STRESS_STEPS:-200}"
LOSS_LOG_EVERY_STEPS="${LOSS_LOG_EVERY_STEPS:-10}"
LOSS_FLUSH_EVERY_STEPS="${LOSS_FLUSH_EVERY_STEPS:-100}"
HANDLING_FAILURE=0

if [[ -n "${RESUME_FROM}" && -z "${RUN_TIMESTAMP}" ]]; then
  resume_parent="$(basename "$(dirname "${RESUME_FROM}")")"
  if [[ "${resume_parent}" == run_* ]]; then
    RUN_TIMESTAMP="${resume_parent#run_}"
  fi
fi
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
RUN_DIR="${OUTPUT_DIR}/run_${RUN_TIMESTAMP}"
MODEL_PATH="${MODEL_PATH:-${RUN_DIR}/${RUN_TIMESTAMP}_model_best.pth}"
PLATFORM_SUMMARY_PATH="${PLATFORM_SUMMARY_PATH:-${RUN_DIR}/${RUN_TIMESTAMP}_platform_summary.json}"
PROJECT_LOG_PATH="${PROJECT_LOG_PATH:-${RUN_DIR}/${RUN_TIMESTAMP}_project.log}"
AUDIT_REPORT_PATH="${AUDIT_REPORT_PATH:-${RUN_DIR}/${RUN_TIMESTAMP}_data_audit.json}"

upload_with_cli() {
  local source_file="$1"
  local target_dir="$2"
  if command -v ossutil >/dev/null 2>&1; then
    ossutil cp "${source_file}" "${target_dir}"
    return 0
  fi
  if command -v oss >/dev/null 2>&1; then
    oss cp "${source_file}" "${target_dir}"
    return 0
  fi
  echo "No OSS upload CLI found (expected ossutil or oss)." >&2
  return 1
}

handle_failure() {
  local exit_code="${1:-1}"
  local line_no="${2:-unknown}"

  if [[ "${HANDLING_FAILURE}" == "1" ]]; then
    exit "${exit_code}"
  fi
  HANDLING_FAILURE=1
  trap - ERR
  set +e

  local timestamp
  timestamp="$(date '+%Y%m%d-%H%M%S')"
  local crash_dir="/tmp/mol_gtn_crash_report_${timestamp}"
  local crash_zip="/tmp/crash_report_${timestamp}.zip"

  mkdir -p "${crash_dir}"

  echo "ERROR detected at line ${line_no}, exit_code=${exit_code}" | tee "${crash_dir}/failure_summary.txt"
  echo "full_lmdb=${FULL_LMDB_PATH}" >> "${crash_dir}/failure_summary.txt"
  echo "medium_lmdb=${MEDIUM_LMDB_PATH}" >> "${crash_dir}/failure_summary.txt"
  echo "batch_per_gpu=${BATCH_PER_GPU}" >> "${crash_dir}/failure_summary.txt"
  echo "grad_accum_steps=${GRAD_ACCUM_STEPS}" >> "${crash_dir}/failure_summary.txt"
  echo "scaled_learning_rate=${SCALED_LEARNING_RATE}" >> "${crash_dir}/failure_summary.txt"
  echo "master_port=${MASTER_PORT}" >> "${crash_dir}/failure_summary.txt"

  find "${OUTPUT_DIR}" -maxdepth 3 -type f -name '*.log' -exec cp -f {} "${crash_dir}/" \; 2>/dev/null || true
  find "${OUTPUT_DIR}" -maxdepth 3 -type f -name 'autotune_*.json' -exec cp -f {} "${crash_dir}/" \; 2>/dev/null || true

  local latest_checkpoint
  latest_checkpoint="$(find "${OUTPUT_DIR}" -maxdepth 4 -type f -name '*.pth' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | awk '{print $2}')"
  if [[ -n "${latest_checkpoint:-}" && -f "${latest_checkpoint}" ]]; then
    cp -f "${latest_checkpoint}" "${crash_dir}/"
  fi

  (
    cd /tmp
    zip -q -r "${crash_zip}" "$(basename "${crash_dir}")"
  )

  local run_scoped_oss_dir="${OSS_TARGET_DIR%/}/run_${RUN_TIMESTAMP}/crash/"
  if upload_with_cli "${crash_zip}" "${run_scoped_oss_dir}"; then
    echo "Crash report uploaded to ${run_scoped_oss_dir}: ${crash_zip##*/}"
  else
    echo "WARNING: Crash report upload failed. Keeping machine alive for manual inspection." >&2
    rm -rf "${crash_dir}" "${crash_zip}"
    exit "${exit_code}"
  fi

  rm -rf "${crash_dir}" "${crash_zip}"

  if [[ "${AUTO_UPLOAD_AND_SHUTDOWN}" == "1" ]]; then
    echo "CRITICAL: Training aborted. Crash report uploaded. Shutting down to save costs."
    "${SHUTDOWN_CMD}"
  else
    echo "CRITICAL: Training aborted. Crash report uploaded. AUTO_UPLOAD_AND_SHUTDOWN=0, keeping machine alive for debugging."
    exit "${exit_code}"
  fi
}

trap 'handle_failure $? $LINENO' ERR

# NCCL / DDP runtime robustness
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${RUN_DIR}"

if [[ "${SKIP_PREPROCESS}" == "1" && ! -f "${FULL_LMDB_PATH}" ]]; then
  echo "ERROR: SKIP_PREPROCESS=1 but LMDB is missing: ${FULL_LMDB_PATH}" >&2
  exit 1
fi

echo "==== Molecular GTN Full-Scale Run ===="
echo "MASTER_PORT=${MASTER_PORT}"
echo "batch_per_gpu=${BATCH_PER_GPU}"
echo "max_nodes_per_batch=${MAX_NODES_PER_BATCH}"
echo "data_mode=${DATA_MODE}"
echo "run_timestamp=${RUN_TIMESTAMP}"
echo "grad_accum_steps=${GRAD_ACCUM_STEPS}"
echo "scaled_learning_rate=${SCALED_LEARNING_RATE}"
echo "full_lmdb=${FULL_LMDB_PATH}"
echo "project_log=${PROJECT_LOG_PATH}"
echo "auto_upload_and_shutdown=${AUTO_UPLOAD_AND_SHUTDOWN}"

python check_platform.py --output-dir "${OUTPUT_DIR}" --summary-path "${PLATFORM_SUMMARY_PATH}"

if [[ -f "${FULL_LMDB_PATH}" ]]; then
  echo "==== Data Audit: ${FULL_LMDB_PATH} ===="
  python -m mol_gtn.data_audit \
    --lmdb "${FULL_LMDB_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --report-path "${AUDIT_REPORT_PATH}" \
    --data-mode "${DATA_MODE}" \
    --sample-count "${AUDIT_SAMPLE_COUNT}"
  EFFECTIVE_DATA_MODE="$(python - <<PY
import json
from pathlib import Path
report = Path("${AUDIT_REPORT_PATH}")
if report.exists():
    payload = json.loads(report.read_text(encoding="utf-8"))
    print(payload.get("branch_decision", "online"))
else:
    print("online")
PY
)"
else
  EFFECTIVE_DATA_MODE="${DATA_MODE}"
  if [[ "${EFFECTIVE_DATA_MODE}" == "auto" ]]; then
    EFFECTIVE_DATA_MODE="online"
  fi
fi
echo "effective_data_mode=${EFFECTIVE_DATA_MODE}"

if [[ "${SKIP_PREPROCESS}" != "1" || "${EFFECTIVE_DATA_MODE}" == "offline" ]]; then
  echo "==== Step 1/3: full preprocessing ===="
  python -m mol_gtn.preprocess \
    --csv "${CSV_PATH}" \
    --lmdb "${FULL_LMDB_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --workers "${PREPROCESS_WORKER_COUNT}" \
    --cpu-reserve-threads "${CPU_RESERVE_THREADS}" \
    --task-queue-maxsize "${TASK_QUEUE_MAXSIZE}" \
    --result-queue-maxsize "${RESULT_QUEUE_MAXSIZE}" \
    --writer-batch-size "${WRITER_BATCH_SIZE}" \
    --lap-pe-dim "${LAP_PE_DIM}" \
    --data-mode "${EFFECTIVE_DATA_MODE}"
else
  echo "==== Step 1/3: skipping preprocessing, reusing ${FULL_LMDB_PATH} ===="
fi

if [[ "${RUN_MEDIUM_STRESS_TEST}" == "1" ]]; then
  if [[ ! -f "${MEDIUM_LMDB_PATH}" ]]; then
    echo "==== Step 2/3: skipping medium DDP stress test; missing ${MEDIUM_LMDB_PATH} ===="
  else
  echo "==== Step 2/3: medium DDP stress test ===="
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${GPUS}" \
    --master_port="${MASTER_PORT}" \
    -m mol_gtn.train \
    --ddp \
    --medium-test \
    --medium-lmdb "${MEDIUM_LMDB_PATH}" \
    --batch-per-gpu "${BATCH_PER_GPU}" \
    --max-nodes-per-batch "${MAX_NODES_PER_BATCH}" \
    --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
    --scaled-learning-rate "${SCALED_LEARNING_RATE}" \
    --epochs 1 \
    --max-steps "${STRESS_STEPS}" \
    --skip-save \
    --mask-ratio "${MASK_RATIO}" \
    --num-masked-views "${NUM_MASKED_VIEWS}" \
    --temperature "${TEMPERATURE}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --num-layers "${NUM_LAYERS}" \
    --lap-pe-dim "${LAP_PE_DIM}" \
    --output-dir "${OUTPUT_DIR}" \
    --log-path "${PROJECT_LOG_PATH}" \
    --run-timestamp "${RUN_TIMESTAMP}" \
    --data-mode "${EFFECTIVE_DATA_MODE}" \
    --loss-log-every-steps "${LOSS_LOG_EVERY_STEPS}" \
    --loss-flush-every-steps "${LOSS_FLUSH_EVERY_STEPS}" \
    --resume-from "${RESUME_FROM}"
  fi
fi

echo "==== Step 3/3: full-scale distributed pretraining ===="
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  -m mol_gtn.train \
  --ddp \
  --lmdb "${FULL_LMDB_PATH}" \
  --batch-per-gpu "${BATCH_PER_GPU}" \
  --max-nodes-per-batch "${MAX_NODES_PER_BATCH}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --scaled-learning-rate "${SCALED_LEARNING_RATE}" \
  --epochs "${EPOCHS}" \
  --mask-ratio "${MASK_RATIO}" \
  --num-masked-views "${NUM_MASKED_VIEWS}" \
  --temperature "${TEMPERATURE}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --num-layers "${NUM_LAYERS}" \
  --lap-pe-dim "${LAP_PE_DIM}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-path "${MODEL_PATH}" \
  --log-path "${PROJECT_LOG_PATH}" \
  --run-timestamp "${RUN_TIMESTAMP}" \
  --data-mode "${EFFECTIVE_DATA_MODE}" \
  --loss-log-every-steps "${LOSS_LOG_EVERY_STEPS}" \
  --loss-flush-every-steps "${LOSS_FLUSH_EVERY_STEPS}" \
  --resume-from "${RESUME_FROM}"

echo "==== Done ===="
echo "model_path=${MODEL_PATH}"
echo "log_path=${PROJECT_LOG_PATH}"

if [[ "${AUTO_UPLOAD_AND_SHUTDOWN}" == "1" ]]; then
  echo "==== Step 4/4: upload results and shutdown ===="
  RESULT_DIR="${OUTPUT_DIR}/run_${RUN_TIMESTAMP}" OSS_TARGET_DIR="${OSS_TARGET_DIR}/run_${RUN_TIMESTAMP}/" "${UPLOAD_SCRIPT_PATH}"
fi
