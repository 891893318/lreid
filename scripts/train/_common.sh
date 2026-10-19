#!/usr/bin/env bash
# Common config for the prompt-free GSCM ablation suite.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SETTING="${SETTING:-1}"
SEED="${SEED:-0}"
DATA_DIR="${DATA_DIR:-/root/data}"

SDAC_ARGS=(
  --sdac
  --sdac-weight 0.2
  --sdac-text-weight 1.0
  --sdac-feat-weight 0.25
  --sdac-logit-weight 0.5
  --sdac-ema 0.997
  --sdac-start-phase 2
)

GSCM_ARGS=(
  --gscm
  --gscm-weight "${GSCM_WEIGHT:-0.10}"
  --gscm-visual-temp "${GSCM_VISUAL_TEMP:-0.07}"
  --gscm-text-temp "${GSCM_TEXT_TEMP:-0.07}"
  --gscm-base-margin "${GSCM_BASE_MARGIN:-0.05}"
  --gscm-semantic-margin "${GSCM_SEMANTIC_MARGIN:-0.10}"
  --gscm-anchor-groups semantic
  --gscm-entropy-confidence
  --gscm-cross-camera
  --gscm-start-phase 1
  --gscm-start-epoch 0
  --gscm-warmup-epochs "${GSCM_WARMUP_EPOCHS:-5}"
)

SA_RSTKT_ARGS=(
  --sa-rstkt
  --sa-rstkt-weight "${SA_RSTKT_WEIGHT:-0.12}"
  --sa-rstkt-text-temp 0.07
  --sa-rstkt-semantic-weight "${SA_RSTKT_SEMANTIC_WEIGHT:-0.7}"
  --sa-rstkt-neg-weight "${SA_RSTKT_NEG_WEIGHT:-0.2}"
  --sa-rstkt-conf-floor "${SA_RSTKT_CONF_FLOOR:-0.03}"
  --sa-rstkt-start-phase 2
  --sa-rstkt-start-epoch 5
  --sa-rstkt-warmup-epochs 10
)

SA_LTKC_ARGS=(
  --sa-ltkc
  --sa-ltkc-relation-weight 1.0
  --sa-ltkc-semantic-weight "${SA_LTKC_SEMANTIC_WEIGHT:-0.5}"
  --sa-ltkc-text-temp 0.07
  --sa-ltkc-min-alpha 0.55
  --sa-ltkc-max-alpha 0.98
  --sa-ltkc-low-scale 0.85
  --sa-ltkc-high-scale 1.05
  --sa-ltkc-head-alpha 0.95
)

run_exp() {
  local name="$1"
  shift
  local launch_log_dir="${LAUNCH_LOG_DIR:-$PROJECT_ROOT/reproduce}"
  local launch_log_prefix="${LAUNCH_LOG_PREFIX:-ablation_v3}"
  local launch_log="$launch_log_dir/${launch_log_prefix}_setting${SETTING}_${name}_seed${SEED}.log"
  local experiment_logs_dir
  if [[ -n "${EXPERIMENT_LOGS_ROOT:-}" ]]; then
    experiment_logs_dir="$EXPERIMENT_LOGS_ROOT/${name}_setting${SETTING}_seed${SEED}"
  else
    experiment_logs_dir="$PROJECT_ROOT/reproduce/ablation_v3_${name}_setting${SETTING}_seed${SEED}"
  fi

  mkdir -p "$PROJECT_ROOT/reproduce" "$launch_log_dir" "$experiment_logs_dir"
  cd "$PROJECT_ROOT"
  : > "$launch_log"
  printf 'Experiment: %s\nSetting: %s\nSeed: %s\nGPU: %s\nLaunch log: %s\nOriginal logs dir: %s\n' \
    "$name" "$SETTING" "$SEED" "$CUDA_DEVICE" "$launch_log" "$experiment_logs_dir" \
    | tee -a "$launch_log"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python continual_train.py \
    --MODEL clip_vit \
    --optimizer Adam \
    --lr 5e-6 \
    --head-lr 3.5e-4 \
    --weight-decay 1e-4 \
    --warmup-factor 0.1 \
    --warmup-step 3 \
    --data-dir "$DATA_DIR" \
    --logs-dir "$experiment_logs_dir" \
    --setting "$SETTING" \
    --seed "$SEED" \
    --no-sa-rstkt --no-sa-ltkc \
    --no-gscm --no-sdac \
    -b 64 --num-instances 8 \
    "$@" 2>&1 | tee -a "$launch_log"
}
