#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

IMAGE=${IMAGE:-dynamo-sglang-decode-migration:dev}
DYNAMO_ROOT=${DYNAMO_ROOT:-/root/.codex/worktrees/decode-migration/dynamo}
SGLANG_ROOT=${SGLANG_ROOT:-/root/.codex/worktrees/decode-migration/sglang}
MODEL_ROOT=${MODEL_ROOT:-/root/models/qwen3-0.6b}
MODEL_PATH_IN_CONTAINER=${MODEL_PATH_IN_CONTAINER:-/models/model}
RESULT_DIR=${RESULT_DIR:-/tmp/decode-migration-results}

mkdir -p "$RESULT_DIR"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    IMAGE="$IMAGE" DYNAMO_ROOT="$DYNAMO_ROOT" \
        "$DYNAMO_ROOT/examples/backends/sglang/decode_migration/build_image.sh"
fi

docker run --rm --user root --gpus all --ipc host --network host \
    --entrypoint bash \
    -e PYTHONPATH=/workspace/dynamo/components/src:/workspace/sglang/python \
    -e MODEL_PATH="$MODEL_PATH_IN_CONTAINER" \
    -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3-0.6B}" \
    -e SOURCE_TP="${SOURCE_TP:-1}" \
    -e DESTINATION_TP="${DESTINATION_TP:-1}" \
    -e SOURCE_GPUS="${SOURCE_GPUS:-0}" \
    -e DESTINATION_GPUS="${DESTINATION_GPUS:-1}" \
    -e SGLANG_DISAGG_STAGING_BUFFER="${SGLANG_DISAGG_STAGING_BUFFER:-0}" \
    -e ATTENTION_BACKEND="${ATTENTION_BACKEND:-}" \
    -e ENABLE_DETERMINISTIC_INFERENCE="${ENABLE_DETERMINISTIC_INFERENCE:-0}" \
    -e STREAM_INTERVAL="${STREAM_INTERVAL:-1}" \
    -e LOG_DIR="/results/stream-${STREAM_INTERVAL:-1}" \
    -v "$DYNAMO_ROOT:/workspace/dynamo" \
    -v "$SGLANG_ROOT:/workspace/sglang" \
    -v "$MODEL_ROOT:$MODEL_PATH_IN_CONTAINER:ro" \
    -v "$RESULT_DIR:/results" \
    "$IMAGE" \
    /workspace/dynamo/examples/backends/sglang/decode_migration/launch_local.sh
