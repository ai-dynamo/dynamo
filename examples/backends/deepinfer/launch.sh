#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --verify ) ]]; then
    echo "Usage: $0 [--verify]" >&2
    exit 2
fi
EXAMPLE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$EXAMPLE_DIR/../../.." && pwd)
source "$REPO_ROOT/examples/common/launch_utils.sh"

PYTHON=${DEEPINFER_PYTHON:-$REPO_ROOT/.venv/bin/python}
MODEL_DIR=${DEEPINFER_MODEL_DIR:-$EXAMPLE_DIR/.cache/model}
ENGINE=${DEEPINFER_ENGINE:-mock}
RUN_DIR=${DEEPINFER_RUN_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/deepinfer-openengine.XXXXXX")}
HTTP_PORT=${DYN_HTTP_PORT:-8000}
GRPC_PORT=${DEEPINFER_GRPC_PORT:-30001}
NAMESPACE=${DYN_NAMESPACE:-deepinfer}
mkdir -p "$RUN_DIR"
RUN_DIR=$(cd -- "$RUN_DIR" && pwd)
children=()
cleanup() {
    local pid alive attempt
    for pid in "${children[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for attempt in {1..100}; do
        alive=0
        for pid in "${children[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
        [[ "$alive" == 0 ]] && break
        sleep 0.1
    done
    for pid in "${children[@]}"; do
        kill -KILL "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    printf 'Logs retained in %s\n' "$RUN_DIR"
}
trap cleanup EXIT
trap 'exit 0' INT TERM

cd "$REPO_ROOT"
if [[ -z ${PROTOC:-} && -x "$REPO_ROOT/.venv/bin/protoc" ]]; then
    export PROTOC="$REPO_ROOT/.venv/bin/protoc"
fi
if [[ ${DEEPINFER_SKIP_BUILD:-0} != 1 ]]; then
    PYO3_PYTHON="$PYTHON" cargo build --locked -p dynamo-deepinfer-openengine -p dynamo-trtllm-sidecar --bins
    PATH="$(dirname -- "$PYTHON"):$PATH" \
        VIRTUAL_ENV=$("$PYTHON" -c 'import sys; print(sys.prefix)') \
        PYO3_PYTHON="$PYTHON" maturin develop --locked \
        --manifest-path "$EXAMPLE_DIR/Cargo.toml"
fi
"$PYTHON" -c 'import _deepinfer_openengine'
TARGET_DIR=$(cargo metadata --no-deps --format-version 1 | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')
BIN_DIR=${DEEPINFER_BIN_DIR:-$TARGET_DIR/debug}
SERVER="$BIN_DIR/deepinfer-openengine"
SIDECAR="$BIN_DIR/dynamo-trtllm-sidecar"
FRONTEND="$BIN_DIR/deepinfer-example-frontend"
for binary in "$SERVER" "$SIDECAR" "$FRONTEND"; do
    [[ -x "$binary" ]] || { printf 'Missing executable: %s\n' "$binary" >&2; exit 1; }
done
if [[ "$ENGINE" == mock ]]; then
    prepare=("$PYTHON" "$EXAMPLE_DIR/prepare_model.py" --model-dir "$MODEL_DIR")
    [[ -z ${DEEPINFER_TOKENIZER_JSON:-} ]] || prepare+=(--tokenizer-json "$DEEPINFER_TOKENIZER_JSON")
    "${prepare[@]}"
fi
MODEL_DIR=$(cd -- "$MODEL_DIR" && pwd)
export DYN_FILE_KV="$RUN_DIR/discovery"
export DYN_DISCOVERY_BACKEND=file DYN_REQUEST_PLANE=tcp DYN_RESPONSE_PLANE=tcp DYN_EVENT_PLANE=zmq
export DYN_SYSTEM_HOST=127.0.0.1 DYN_SYSTEM_PORT=0
export DYN_HEALTH_CHECK_ENABLED=false
export RUST_LOG=${RUST_LOG:-info}

engine_args=()
[[ -z ${DEEPINFER_ENGINE_ARGS:-} ]] || engine_args+=(--engine-args "$DEEPINFER_ENGINE_ARGS")
"$PYTHON" "$EXAMPLE_DIR/server.py" --listen "127.0.0.1:$GRPC_PORT" --model-dir "$MODEL_DIR" \
    --engine "$ENGINE" --tick-ms "${DEEPINFER_TICK_MS:-2}" \
    "${engine_args[@]}" >"$RUN_DIR/engine.log" 2>&1 &
children+=("$!")
ready=0
for attempt in {1..600}; do
    kill -0 "${children[0]}" 2>/dev/null || { cat "$RUN_DIR/engine.log" >&2; exit 1; }
    if "$SERVER" probe --endpoint "http://127.0.0.1:$GRPC_PORT" >"$RUN_DIR/engine-ready.json" 2>/dev/null; then ready=1; break; fi
    sleep 0.1
done
[[ "$ready" == 1 ]] || { cat "$RUN_DIR/engine.log" >&2; exit 1; }

"$SIDECAR" --grpc-endpoint "http://127.0.0.1:$GRPC_PORT" --model-path "$MODEL_DIR" \
    --namespace "$NAMESPACE" --component deepinfer --dyn-tool-call-parser deepseek_v41 \
    --dyn-reasoning-parser deepseek_v41 --dyn-default-thinking-mode enabled \
    --grpc-connections 1 --grpc-startup-deadline-secs 30 \
    >"$RUN_DIR/sidecar.log" 2>&1 &
children+=("$!")
"$FRONTEND" --http-port "$HTTP_PORT" --namespace "$NAMESPACE" \
    >"$RUN_DIR/frontend.log" 2>&1 &
children+=("$!")

"$PYTHON" "$EXAMPLE_DIR/verify.py" --url "http://127.0.0.1:$HTTP_PORT" \
    --wait-only --timeout 60 --model "$MODEL_DIR"
"$PYTHON" - "$RUN_DIR/processes.json" "$HTTP_PORT" "$GRPC_PORT" "${children[@]}" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"http_port":int(sys.argv[2]), "grpc_port":int(sys.argv[3]),
    "engine_pid":int(sys.argv[4]), "sidecar_pid":int(sys.argv[5]), "frontend_pid":int(sys.argv[6])}, indent=2) + "\n")
PY
print_launch_banner --no-curl "DeepInfer OpenEngine: CPU mock → sidecar → Dynamo frontend" "$MODEL_DIR" "$HTTP_PORT" \
    "OpenEngine: http://127.0.0.1:$GRPC_PORT" "Logs: $RUN_DIR" \
    "Verify: $PYTHON $EXAMPLE_DIR/verify.py --url http://127.0.0.1:$HTTP_PORT --engine-endpoint http://127.0.0.1:$GRPC_PORT --server-bin $SERVER"
if [[ ${1:-} == --verify ]]; then
    "$PYTHON" "$EXAMPLE_DIR/check_bridge.py" --model-dir "$MODEL_DIR" >"$RUN_DIR/bridge-checks.log" 2>&1
    "$PYTHON" "$EXAMPLE_DIR/verify.py" --url "http://127.0.0.1:$HTTP_PORT" \
        --engine-endpoint "http://127.0.0.1:$GRPC_PORT" --server-bin "$SERVER" \
        --output "$RUN_DIR/results.json"
    exit 0
fi
wait_any_exit
