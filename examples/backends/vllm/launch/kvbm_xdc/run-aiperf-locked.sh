#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/hardware-profiles.sh"
kvbm_xdc_apply_hardware_profile model-only

URL=${URL:?URL is required}
ARTIFACT_DIR=${ARTIFACT_DIR:?ARTIFACT_DIR is required}

: "${MODEL:?MODEL must be set by KVBM_HARDWARE_PROFILE or env override}"

mkdir -p "$ARTIFACT_DIR"

exec aiperf profile \
  --model "$MODEL" \
  --url "$URL" \
  --endpoint-type chat \
  --streaming \
  --concurrency 8 \
  --request-count 200 \
  --synthetic-input-tokens-mean 1101 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 256 \
  --output-tokens-stddev 0 \
  --artifact-dir "$ARTIFACT_DIR" \
  --ui none
