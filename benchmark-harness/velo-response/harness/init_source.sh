#!/usr/bin/env bash
set -euo pipefail
ROOT=/lustre/fsw/coreai_comparch_trtllm/jothomson/dynamo-velo-response-20260923
REPO="$ROOT/src/dynamo"
git -C "$REPO" init
git -C "$REPO" fetch --depth=1 https://github.com/ai-dynamo/dynamo jthomson04/velo-response
git -C "$REPO" reset --mixed FETCH_HEAD
mkdir -p "$ROOT/cache/mdc" "$ROOT/harness" "$ROOT/configs"
cp "$REPO/benchmark-harness/velo-response/harness/"* "$ROOT/harness/"
cp "$REPO/benchmark-harness/velo-response/configs/template.json" "$ROOT/configs/template.json"
cp /lustre/fsw/coreai_comparch_trtllm/jothomson/dynamo-tyche-main-profile-20260923/harness/swagger-ui-v5.17.14.zip "$ROOT/harness/"
