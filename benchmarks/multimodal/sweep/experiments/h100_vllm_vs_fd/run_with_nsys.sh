#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run vllm-baseline vs dynamo-fd with nsys profiling.
#
# Reads parameters from sweep.yaml via the Python config loader.
# For each config x request_rate: launches server under nsys, runs aiperf, collects traces.
#
# Usage:
#   bash benchmarks/multimodal/sweep/experiments/h100_vllm_vs_fd/run_with_nsys.sh
#   bash benchmarks/multimodal/sweep/experiments/h100_vllm_vs_fd/run_with_nsys.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
SWEEP_YAML="$SCRIPT_DIR/sweep.yaml"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
die() { echo "FATAL: $*" >&2; exit 1; }

wait_for_model() {
    local url="http://localhost:${PORT}/v1/models"
    local deadline=$((SECONDS + TIMEOUT))
    echo "  Waiting for $url (timeout ${TIMEOUT}s)..."
    while [[ $SECONDS -lt $deadline ]]; do
        if curl -sf "$url" 2>/dev/null | grep -q "$MODEL"; then
            echo "  Server ready (${SECONDS}s elapsed from script start)."
            return 0
        fi
        sleep 5
    done
    echo "  ERROR: Server not ready within ${TIMEOUT}s"
    return 1
}

cleanup_port() {
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    pkill -f "dynamo.vllm" 2>/dev/null || true
    pkill -f "dynamo.frontend" 2>/dev/null || true
    sleep 3
}

kill_group() {
    local pgid="$1"
    kill -TERM -- -"$pgid" 2>/dev/null || true
    # Wait up to 30s for nsys to finalize
    for _ in $(seq 1 30); do
        kill -0 -- -"$pgid" 2>/dev/null || break
        sleep 1
    done
    kill -9 -- -"$pgid" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Load config from sweep.yaml via the Python config loader
# ---------------------------------------------------------------------------
cd "$REPO_ROOT"

declare -a REQUEST_RATES
declare -a CONFIG_LABELS
declare -a CONFIG_WORKFLOWS
declare -a CONFIG_EXTRA_ARGS

eval "$(python -c "
import sys
sys.path.insert(0, '.')
from benchmarks.multimodal.sweep.config import load_config
cfg = load_config('$SWEEP_YAML')
print(f'MODEL={cfg.model!r}')
print(f'PORT={cfg.port}')
print(f'OSL={cfg.osl}')
print(f'REQUEST_COUNT={cfg.request_count}')
print(f'WARMUP={cfg.warmup_count}')
print(f'TIMEOUT={cfg.timeout}')
print(f'OUTPUT_BASE=\"{cfg.output_dir}\"')
print(f'REQUEST_RATES=({\" \".join(str(r) for r in cfg.request_rates)})')
for k, v in cfg.env.items():
    print(f'export {k}={v!r}')
for i, c in enumerate(cfg.configs):
    print(f'CONFIG_LABELS[{i}]={c.label!r}')
    print(f'CONFIG_WORKFLOWS[{i}]={c.workflow!r}')
    extra = ' '.join(c.extra_args)
    print(f'CONFIG_EXTRA_ARGS[{i}]={extra!r}')
print(f'INPUT_FILE={cfg.input_files[0]!r}')
")"

# Make output_dir absolute
OUTPUT_BASE="$REPO_ROOT/$OUTPUT_BASE"

export HF_HOME="${HF_HOME:-/data/huggingface}"

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
echo "========================================"
echo "  Preflight Checks"
echo "========================================"

command -v nsys   >/dev/null 2>&1 || die "nsys not found on PATH"
command -v aiperf >/dev/null 2>&1 || die "aiperf not found on PATH"
pgrep -x nats-server >/dev/null   || die "nats-server not running"
pgrep -x etcd >/dev/null          || die "etcd not running"

MODEL_DIR="$HF_HOME/hub/models--$(echo "$MODEL" | sed 's|/|--|g')"
[ -d "$MODEL_DIR" ] || die "Model not cached at $MODEL_DIR"
[ -f "$INPUT_FILE" ] || die "JSONL dataset not found: $INPUT_FILE"

python -c "from dynamo.llm import MediaDecoder" 2>/dev/null \
    || die "Frontend decoding not available (dynamo.llm.MediaDecoder import failed)"

echo "  All checks passed."

# ---------------------------------------------------------------------------
# nsys flags
# ---------------------------------------------------------------------------
BACKEND_NSYS_FLAGS=(
    --trace=cuda,nvtx,cublas,cudnn
    --cuda-memory-usage=true
    --backtrace=dwarf
    --sample=process-tree
    --duration=600
)

FRONTEND_NSYS_FLAGS=(
    --trace=osrt,nvtx
    --sample=cpu
    --cuda-memory-usage=false
    --duration=600
)

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  H100 vLLM vs Dynamo-FD Benchmark"
echo "========================================"
echo "  Model:         $MODEL"
echo "  Request rates: ${REQUEST_RATES[*]}"
echo "  OSL:           $OSL"
echo "  Requests:      $REQUEST_COUNT"
echo "  Input:         $INPUT_FILE"
echo "  Configs:       ${CONFIG_LABELS[*]}"
echo "  Output:        $OUTPUT_BASE"
echo "  Dry run:       $DRY_RUN"
echo "========================================"
echo ""

# ---------------------------------------------------------------------------
# Determine if a config uses frontend decoding
# ---------------------------------------------------------------------------
uses_frontend_decoding() {
    [[ "$1" == *"--frontend-decoding"* ]]
}

is_vllm_baseline() {
    [[ "$1" == *"vllm_serve"* ]]
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
NUM_CONFIGS=${#CONFIG_LABELS[@]}
TOTAL_RUNS=$((NUM_CONFIGS * ${#REQUEST_RATES[@]}))
RUN_NUM=0

for ((i = 0; i < NUM_CONFIGS; i++)); do
    LABEL="${CONFIG_LABELS[$i]}"
    WORKFLOW="${CONFIG_WORKFLOWS[$i]}"
    EXTRA_ARGS="${CONFIG_EXTRA_ARGS[$i]}"

    CONFIG_DIR="$OUTPUT_BASE/$LABEL"
    NSYS_DIR="$CONFIG_DIR/nsys"
    mkdir -p "$NSYS_DIR"

    echo ""
    echo "########################################"
    echo "  Config: $LABEL"
    echo "########################################"

    for RATE in "${REQUEST_RATES[@]}"; do
        RUN_NUM=$((RUN_NUM + 1))
        ARTIFACT_DIR="$CONFIG_DIR/requestrate${RATE}"
        mkdir -p "$ARTIFACT_DIR"

        NSYS_BACKEND="$NSYS_DIR/${LABEL}_r${RATE}_backend"
        NSYS_FRONTEND="$NSYS_DIR/${LABEL}_r${RATE}_frontend"

        echo ""
        echo "  ---- Run $RUN_NUM/$TOTAL_RUNS: $LABEL @ rate=$RATE ----"

        # Build aiperf command
        AIPERF_CMD=(
            aiperf profile
            -m "$MODEL"
            -u "http://localhost:$PORT"
            --request-rate "$RATE"
            --request-count "$REQUEST_COUNT"
            --warmup-request-count "$WARMUP"
            --input-file "$INPUT_FILE"
            --custom-dataset-type single_turn
            --extra-inputs "max_tokens:$OSL"
            --extra-inputs "min_tokens:$OSL"
            --extra-inputs "ignore_eos:true"
            --extra-inputs "stream:true"
            --streaming
            --artifact-dir "$ARTIFACT_DIR"
            --ui none
            --no-server-metrics
        )

        if $DRY_RUN; then
            echo "  [dry-run] Server: $WORKFLOW --model $MODEL $EXTRA_ARGS"
            echo "  [dry-run] nsys backend: $NSYS_BACKEND"
            if uses_frontend_decoding "$EXTRA_ARGS"; then
                echo "  [dry-run] nsys frontend: $NSYS_FRONTEND"
            fi
            echo "  [dry-run] aiperf: ${AIPERF_CMD[*]}"
            continue
        fi

        # Clean port
        cleanup_port

        # Launch server with nsys
        SERVER_LOG="$ARTIFACT_DIR/server.log"

        if is_vllm_baseline "$WORKFLOW"; then
            echo "  Launching vllm serve under nsys..."
            setsid nsys profile \
                "${BACKEND_NSYS_FLAGS[@]}" \
                -o "$NSYS_BACKEND" \
                -- bash "$REPO_ROOT/$WORKFLOW" \
                --model "$MODEL" \
                $EXTRA_ARGS \
                > "$SERVER_LOG" 2>&1 &
            SERVER_PGID=$!

        elif uses_frontend_decoding "$EXTRA_ARGS"; then
            echo "  Launching dynamo (dual nsys: frontend + backend)..."
            setsid bash -c "
                export DYN_REQUEST_PLANE=tcp
                export DYN_ENABLE_RUST_NVTX=1
                export DYN_NVTX=1
                nsys profile ${FRONTEND_NSYS_FLAGS[*]} \
                    -o '$NSYS_FRONTEND' \
                    -- python -m dynamo.frontend &
                FRONTEND_PID=\$!

                nsys profile ${BACKEND_NSYS_FLAGS[*]} \
                    -o '$NSYS_BACKEND' \
                    -- python -m dynamo.vllm \
                    --enable-multimodal \
                    --model '$MODEL' \
                    $EXTRA_ARGS &
                BACKEND_PID=\$!

                wait \$FRONTEND_PID \$BACKEND_PID
            " > "$SERVER_LOG" 2>&1 &
            SERVER_PGID=$!

        else
            echo "  Launching dynamo (nsys on backend only)..."
            setsid bash -c "
                export DYN_REQUEST_PLANE=tcp
                python -m dynamo.frontend &
                FRONTEND_PID=\$!

                nsys profile ${BACKEND_NSYS_FLAGS[*]} \
                    -o '$NSYS_BACKEND' \
                    -- python -m dynamo.vllm \
                    --enable-multimodal \
                    --model '$MODEL' \
                    $EXTRA_ARGS &
                BACKEND_PID=\$!

                wait \$FRONTEND_PID \$BACKEND_PID
            " > "$SERVER_LOG" 2>&1 &
            SERVER_PGID=$!
        fi

        # Wait for server readiness
        if ! wait_for_model; then
            echo "  SKIPPING: server failed to start. Check $SERVER_LOG"
            kill_group "$SERVER_PGID"
            continue
        fi

        # Run aiperf
        echo "  Running aiperf (rate=$RATE, requests=$REQUEST_COUNT)..."
        "${AIPERF_CMD[@]}" 2>&1 | tee "$ARTIFACT_DIR/aiperf.log"
        AIPERF_EXIT=${PIPESTATUS[0]}

        if [[ $AIPERF_EXIT -ne 0 ]]; then
            echo "  WARNING: aiperf exited with code $AIPERF_EXIT"
        else
            echo "  aiperf completed successfully."
        fi

        # Shutdown server — SIGTERM lets nsys finalize .nsys-rep
        echo "  Shutting down server (waiting for nsys finalization)..."
        kill_group "$SERVER_PGID"

        # Verify nsys output
        if [[ -f "${NSYS_BACKEND}.nsys-rep" ]]; then
            SIZE=$(stat -c%s "${NSYS_BACKEND}.nsys-rep" 2>/dev/null || stat -f%z "${NSYS_BACKEND}.nsys-rep" 2>/dev/null || echo 0)
            echo "  nsys backend trace: ${NSYS_BACKEND}.nsys-rep ($((SIZE / 1024 / 1024))MB)"
        else
            echo "  WARNING: Backend nsys trace not found at ${NSYS_BACKEND}.nsys-rep"
        fi

        if uses_frontend_decoding "$EXTRA_ARGS" && [[ -f "${NSYS_FRONTEND}.nsys-rep" ]]; then
            SIZE=$(stat -c%s "${NSYS_FRONTEND}.nsys-rep" 2>/dev/null || stat -f%z "${NSYS_FRONTEND}.nsys-rep" 2>/dev/null || echo 0)
            echo "  nsys frontend trace: ${NSYS_FRONTEND}.nsys-rep ($((SIZE / 1024 / 1024))MB)"
        fi

        echo "  ---- Done: $LABEL @ rate=$RATE ----"
    done
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  Benchmark Complete"
echo "========================================"
echo "  Results: $OUTPUT_BASE"
for ((i = 0; i < NUM_CONFIGS; i++)); do
    LABEL="${CONFIG_LABELS[$i]}"
    echo "    $LABEL:"
    for RATE in "${REQUEST_RATES[@]}"; do
        DIR="$OUTPUT_BASE/$LABEL/requestrate${RATE}"
        if [[ -f "$DIR/profile_export_aiperf.json" ]]; then
            echo "      rate=${RATE}: $DIR/ [OK]"
        else
            echo "      rate=${RATE}: $DIR/ [MISSING profile_export_aiperf.json]"
        fi
    done
    echo "      nsys: $OUTPUT_BASE/$LABEL/nsys/"
done
echo "========================================"
