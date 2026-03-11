#!/usr/bin/env bash
set -euo pipefail

TARGET_PID="${1:-}"
RECIPIENT="${2:-2243387748@qq.com}"

if [[ -z "${TARGET_PID}" ]]; then
  echo "Usage: $0 <final_fullscale_run_pid> [recipient_email]" >&2
  exit 1
fi

if ! kill -0 "${TARGET_PID}" 2>/dev/null; then
  echo "PID ${TARGET_PID} is not running" >&2
  exit 1
fi

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate mol_gtn

nohup python monitor_training.py \
  --pid "${TARGET_PID}" \
  --recipient "${RECIPIENT}" \
  --log-path /hy-tmp/result/project.log \
  --output-dir /hy-tmp/result \
  --state-path /hy-tmp/result/monitor_state.json \
  --send-start-email \
  >/hy-tmp/result/monitor_stdout.log 2>/hy-tmp/result/monitor_stderr.log &

echo "Monitor started for PID ${TARGET_PID}"
echo "stdout: /hy-tmp/result/monitor_stdout.log"
echo "stderr: /hy-tmp/result/monitor_stderr.log"
