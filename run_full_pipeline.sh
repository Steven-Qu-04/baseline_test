#!/usr/bin/env bash
set -euo pipefail

# CONFIGURATION
CSV_PATH="${CSV_PATH:-pretraining.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-/hy-tmp/result}"
LMDB_PATH="${LMDB_PATH:-/hy-tmp/result/pretraining.lmdb}"
SMOKE_LMDB_PATH="${SMOKE_LMDB_PATH:-/hy-tmp/result/smoke_pretraining.lmdb}"
MODEL_PATH="${MODEL_PATH:-/hy-tmp/result/model_final.pth}"
EMBEDDINGS_PATH="${EMBEDDINGS_PATH:-/hy-tmp/result/top10_embeddings.csv}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MASK_RATIO="${MASK_RATIO:-0.1}"
NUM_MASKED_VIEWS="${NUM_MASKED_VIEWS:-6}"
SMOKE_NUM_MASKED_VIEWS="${SMOKE_NUM_MASKED_VIEWS:-2}"
CPU_WORKERS="${CPU_WORKERS:-48}"
EPOCHS="${EPOCHS:-3}"
TEMPERATURE="${TEMPERATURE:-0.07}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
LAP_PE_DIM="${LAP_PE_DIM:-8}"
QUEUE_SIZE="${QUEUE_SIZE:-256}"
WRITER_BATCH_SIZE="${WRITER_BATCH_SIZE:-64}"
SMOKE_TEST="${SMOKE_TEST:-0}"

COMMON_ARGS=(
  --output-dir "${OUTPUT_DIR}"
)

python -m mol_gtn.check_env "${COMMON_ARGS[@]}"

PREPROCESS_ARGS=(
  --csv "${CSV_PATH}"
  --lmdb "${LMDB_PATH}"
  --smoke-lmdb "${SMOKE_LMDB_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --workers "${CPU_WORKERS}"
  --queue-size "${QUEUE_SIZE}"
  --writer-batch-size "${WRITER_BATCH_SIZE}"
  --lap-pe-dim "${LAP_PE_DIM}"
)

TRAIN_ARGS=(
  --lmdb "${LMDB_PATH}"
  --smoke-lmdb "${SMOKE_LMDB_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --model-path "${MODEL_PATH}"
  --batch-size "${BATCH_SIZE}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --epochs "${EPOCHS}"
  --mask-ratio "${MASK_RATIO}"
  --num-masked-views "${NUM_MASKED_VIEWS}"
  --smoke-num-masked-views "${SMOKE_NUM_MASKED_VIEWS}"
  --temperature "${TEMPERATURE}"
  --hidden-dim "${HIDDEN_DIM}"
  --lap-pe-dim "${LAP_PE_DIM}"
)

INFER_ARGS=(
  --lmdb "${LMDB_PATH}"
  --smoke-lmdb "${SMOKE_LMDB_PATH}"
  --model-path "${MODEL_PATH}"
  --embeddings-path "${EMBEDDINGS_PATH}"
  --hidden-dim "${HIDDEN_DIM}"
  --lap-pe-dim "${LAP_PE_DIM}"
)

if [[ "${SMOKE_TEST}" == "1" ]]; then
  PREPROCESS_ARGS+=(--smoke-test)
  TRAIN_ARGS+=(--smoke-test)
  INFER_ARGS+=(--smoke-test)
fi

python -m mol_gtn.preprocess "${PREPROCESS_ARGS[@]}"
python -m mol_gtn.train "${TRAIN_ARGS[@]}"
python -m mol_gtn.infer "${INFER_ARGS[@]}"
