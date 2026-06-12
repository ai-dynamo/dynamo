#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
IMAGE=${IMAGE:-dynamo-sglang-decode-migration:dev}
SGLANG_WORKTREE=${SGLANG_WORKTREE:-/root/.codex/worktrees/decode-migration/sglang}
DYNAMO_WORKTREE=${DYNAMO_WORKTREE:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}
MODEL_DIR=${MODEL_DIR:-/root/models/qwen3-8b}
RESULT_DIR=${RESULT_DIR:-/tmp/qwen3-8b-gsm8k-$(date -u +%Y%m%dT%H%M%SZ)}
CONTAINER_NAME=${CONTAINER_NAME:-qwen3-8b-migration-gsm8k-$(date -u +%H%M%S)}

mkdir -p "$RESULT_DIR"

docker run --rm \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --network host \
    --ipc host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e PYTHONPATH=/workspace/dynamo/components/src:/workspace/sglang/python \
    -e NUM_EXAMPLES="${NUM_EXAMPLES:-200}" \
    -e NUM_THREADS="${NUM_THREADS:-1}" \
    -e MAX_TOKENS="${MAX_TOKENS:-4096}" \
    -e NUM_SHOTS="${NUM_SHOTS:-5}" \
    -e MINIMUM_MIGRATION_COVERAGE="${MINIMUM_MIGRATION_COVERAGE:-1.0}" \
    -e MINIMUM_REASONING_COVERAGE="${MINIMUM_REASONING_COVERAGE:-1.0}" \
    -e MAX_SCORE_REGRESSION="${MAX_SCORE_REGRESSION:-0.02}" \
    -e ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}" \
    -v "$DYNAMO_WORKTREE:/workspace/dynamo" \
    -v "$SGLANG_WORKTREE:/workspace/sglang" \
    -v "$MODEL_DIR:/models/qwen3-8b:ro" \
    -v "$SCRIPT_DIR:/workspace/benchmark-driver:ro" \
    -v "$RESULT_DIR:/results" \
    "$IMAGE" \
    bash /workspace/benchmark-driver/container_qwen3_gsm8k.sh

printf 'Results: %s\n' "$RESULT_DIR"
