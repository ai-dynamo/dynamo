#!/usr/bin/env bash
set -euo pipefail
ROOT=/lustre/fsw/coreai_comparch_trtllm/jothomson/dynamo-velo-response-20260923
mount --bind "$ROOT/cache/mdc" /home/jothomson/.cache/dynamo/mdc
exec "$@"
