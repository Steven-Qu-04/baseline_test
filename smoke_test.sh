#!/usr/bin/env bash
set -euo pipefail

SMOKE_TEST=1 \
BATCH_SIZE="${BATCH_SIZE:-4}" \
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}" \
EPOCHS="${EPOCHS:-1}" \
CPU_WORKERS="${CPU_WORKERS:-4}" \
./run_full_pipeline.sh
