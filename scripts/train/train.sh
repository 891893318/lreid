#!/usr/bin/env bash
# A6: full prompt-free method.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETTING="${1:-${SETTING:-1}}"
export SETTING
source "$SCRIPT_DIR/_common.sh"
run_exp "train" \
  "${SDAC_ARGS[@]}" "${GSCM_ARGS[@]}" \
  "${SA_RSTKT_ARGS[@]}" "${SA_LTKC_ARGS[@]}"
