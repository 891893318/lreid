#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

export DATA_DIR="${DATA_DIR:-/root/data}"
export CUDA_DEVICE="${CUDA_DEVICE:-0}"
export SEED="${SEED:-0}"

bash scripts/train/train.sh 1
bash scripts/train/train.sh 2
