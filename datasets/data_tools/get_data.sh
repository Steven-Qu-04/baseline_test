#!/usr/bin/env bash
set -euo pipefail

# ========== Config ==========
BUCKET="oss://data_for_protein_ligand_project"
# OSS 上对象名约定：<dataset>.tar.gz
# 支持的数据集名称（与你当前目录一致）
ALLOWED=("Lipophilicity" "muv" "pcba" "toxcast")
# ============================

usage() {
  echo "Usage: bash $0 <dataset_name>"
  echo "Allowed: ${ALLOWED[*]}"
  echo "Example: bash $0 pcba"
  exit 1
}

if [[ $# -ne 1 ]]; then
  usage
fi

NAME="$1"
ok=0
for d in "${ALLOWED[@]}"; do
  if [[ "$NAME" == "$d" ]]; then
    ok=1
    break
  fi
done
if [[ $ok -ne 1 ]]; then
  echo "[ERR] Unknown dataset name: $NAME"
  usage
fi

# scripts are in: <project>/datasets/data_tools/
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATASETS_DIR="$PROJECT_ROOT/datasets"

ARCHIVE="${NAME}.tar.gz"
REMOTE="${BUCKET}/${ARCHIVE}"
LOCAL="${DATASETS_DIR}/${ARCHIVE}"

VERIFY_PY="$SCRIPT_DIR/verify_hash.py"

echo "[INFO] Project root : $PROJECT_ROOT"
echo "[INFO] Datasets dir : $DATASETS_DIR"
echo "[INFO] Dataset      : $NAME"
echo "[INFO] Remote       : $REMOTE"
echo "[INFO] Local        : $LOCAL"

mkdir -p "$DATASETS_DIR"

# 1) Ensure OSS is accessible; if not, login interactively
echo "[INFO] Checking OSS access..."
if ! oss ls "$BUCKET" >/dev/null 2>&1; then
  echo "[WARN] OSS not accessible yet. Running: oss login"
  echo "       Please input your credentials interactively (do NOT hardcode them in scripts)."
  oss login
fi

# 2) Download archive
echo "[INFO] Downloading..."
oss cp "$REMOTE" "$LOCAL"

# 3) Extract into datasets/
echo "[INFO] Extracting..."
tar -xzf "$LOCAL" -C "$DATASETS_DIR"

# 4) Verify hashes (dataset integrity)
echo "[INFO] Verifying hashes via verify_hash.py ..."
if [[ -f "$VERIFY_PY" ]]; then
  python "$VERIFY_PY"
else
  echo "[ERR] Not found: $VERIFY_PY"
  exit 3
fi

echo "[OK] Done. Dataset '$NAME' is downloaded into $DATASETS_DIR and verified."
