#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

RUN_ID="${1:?usage: run_nsys.sh RUN_ID [OUTPUT_BASE]}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUTPUT_BASE="${2:-${DYN_NSYS_OUTPUT_BASE:-$REPO_ROOT/logs/nsys/vllm_prompt_embeddings}}"
RUN_DIR="$OUTPUT_BASE/$RUN_ID"

MODEL="${DYN_PROMPT_EMBEDS_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
REQUESTS="${DYN_PROMPT_EMBEDS_REQUESTS:-100}"
WARMUP_REQUESTS="${DYN_PROMPT_EMBEDS_WARMUP_REQUESTS:-20}"
PROMPT_TOKENS="${DYN_PROMPT_EMBEDS_PROMPT_TOKENS:-515}"
OUTPUT_TOKENS="${DYN_PROMPT_EMBEDS_OUTPUT_TOKENS:-75}"
BLOCK_SIZE="${DYN_PROMPT_EMBEDS_BLOCK_SIZE:-16}"
MAX_MODEL_LEN="${DYN_PROMPT_EMBEDS_MAX_MODEL_LEN:-1024}"
GPU_MEMORY_UTILIZATION="${DYN_PROMPT_EMBEDS_GPU_MEMORY_UTILIZATION:-0.90}"
SEED="${DYN_PROMPT_EMBEDS_SEED:-0}"
CONTAINER_IMAGE="${DYN_CONTAINER_IMAGE:-unknown}"
PYTHON_BIN="${DYN_PYTHON_BIN:-python3}"
NSYS_PID=""

if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to overwrite existing run directory: $RUN_DIR" >&2
    exit 2
fi
mkdir -p "$RUN_DIR/nsys" "$RUN_DIR/recipe"
cp "$SCRIPT_DIR"/*.py "$SCRIPT_DIR"/*.sh "$RUN_DIR/recipe/"
sha256sum "$RUN_DIR"/recipe/* > "$RUN_DIR/recipe-sha256sums.txt"

NSYS_STAGING_ROOT="${DYN_NSYS_STAGING_ROOT:-${TMPDIR:-/tmp}/dynamo-nsys-staging}"
NSYS_STAGING="$NSYS_STAGING_ROOT/${RUN_ID:0:32}"
mkdir -p "$NSYS_STAGING"
export TMPDIR="$NSYS_STAGING"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_NVTX_SCOPES_FOR_PROFILING=1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

NSYS_BIN="$(command -v nsys || true)"
if [[ -z "$NSYS_BIN" ]] || ! "$NSYS_BIN" --version >/dev/null 2>&1; then
    echo "missing working nsys executable" >&2
    exit 2
fi

PREFIX_CACHE_REMAINDER=$((PROMPT_TOKENS % BLOCK_SIZE))
if [[ "$PREFIX_CACHE_REMAINDER" -eq 0 ]]; then
    PREFIX_CACHE_REMAINDER="$BLOCK_SIZE"
fi
if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    SOURCE_REVISION="$(git -C "$REPO_ROOT" rev-parse HEAD)"
    if [[ -n "$(git -C "$REPO_ROOT" status --short)" ]]; then
        SOURCE_DIRTY=true
    else
        SOURCE_DIRTY=false
    fi
else
    SOURCE_REVISION=unknown
    SOURCE_DIRTY=unknown
fi

forward_signal() {
    local signal="$1"
    local exit_code="$2"
    if [[ -n "$NSYS_PID" ]] && kill -0 "$NSYS_PID" 2>/dev/null; then
        kill -INT "$NSYS_PID" 2>/dev/null || true
        for _ in $(seq 1 180); do
            kill -0 "$NSYS_PID" 2>/dev/null || break
            sleep 1
        done
    fi
    exit "$exit_code"
}
trap 'forward_signal INT 130' INT
trap 'forward_signal TERM 143' TERM

PYTHON_COMMAND=(
    "$PYTHON_BIN" -m benchmarks.profiling.vllm_prompt_embeddings.main
    --model "$MODEL"
    --requests "$REQUESTS"
    --warmup-requests "$WARMUP_REQUESTS"
    --prompt-tokens "$PROMPT_TOKENS"
    --output-tokens "$OUTPUT_TOKENS"
    --block-size "$BLOCK_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --seed "$SEED"
    --output-dir "$RUN_DIR"
)
NSYS_COMMAND=(
    "$NSYS_BIN" profile
    --output "$RUN_DIR/nsys/vllm-prompt-embeds"
    --trace=cuda,nvtx,cublas,cudnn,osrt
    --cuda-memory-usage=true
    --backtrace=dwarf
    --sample=process-tree
    --cpuctxsw=process-tree
    --trace-fork-before-exec=true
    --cuda-graph-trace=node
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --kill=sigterm
    --wait=primary
    --
    "${PYTHON_COMMAND[@]}"
)
printf '%q ' "${PYTHON_COMMAND[@]}" > "$RUN_DIR/python-command.txt"
printf '\n' >> "$RUN_DIR/python-command.txt"
printf '%q ' "${NSYS_COMMAND[@]}" > "$RUN_DIR/nsys-command.txt"
printf '\n' >> "$RUN_DIR/nsys-command.txt"

{
    echo "run_id=$RUN_ID"
    echo "model=$MODEL"
    echo "requests=$REQUESTS"
    echo "warmup_requests=$WARMUP_REQUESTS"
    echo "prompt_tokens=$PROMPT_TOKENS"
    echo "output_tokens=$OUTPUT_TOKENS"
    echo "block_size=$BLOCK_SIZE"
    echo "cuda_graph_mode=FULL"
    echo "cuda_graph_capture_sizes=1,$PREFIX_CACHE_REMAINDER,$PROMPT_TOKENS"
    echo "prefix_caching=true"
    echo "container_image=$CONTAINER_IMAGE"
    echo "nsys=$($NSYS_BIN --version | sed -n '1p')"
    echo "vllm=$($PYTHON_BIN -c 'import vllm; print(vllm.__version__)')"
    echo "torch=$($PYTHON_BIN -c 'import torch; print(torch.__version__)')"
    echo "transformers=$($PYTHON_BIN -c 'import transformers; print(transformers.__version__)')"
    echo "source_revision=$SOURCE_REVISION"
    echo "source_dirty=$SOURCE_DIRTY"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi \
        --query-gpu=name,uuid,memory.total,driver_version \
        --format=csv,noheader
} > "$RUN_DIR/provenance.txt"

cd "$REPO_ROOT"
"${NSYS_COMMAND[@]}" > "$RUN_DIR/stdout.log" 2> "$RUN_DIR/stderr.log" &
NSYS_PID=$!
set +e
wait "$NSYS_PID"
NSYS_RC=$?
set -e
NSYS_PID=""
trap - INT TERM
if [[ "$NSYS_RC" -ne 0 ]]; then
    echo "nsys profile failed with exit code $NSYS_RC" >&2
    exit "$NSYS_RC"
fi

mapfile -t REPORTS < <(compgen -G "$RUN_DIR/nsys/*.nsys-rep" || true)
if [[ "${#REPORTS[@]}" -ne 1 ]]; then
    echo "expected one .nsys-rep, found ${#REPORTS[@]}" >&2
    exit 1
fi
REP="${REPORTS[0]}"
SQLITE="${REP%.nsys-rep}.sqlite"
"$NSYS_BIN" export \
    --type=sqlite \
    --force-overwrite=true \
    --output "$SQLITE" \
    "$REP" > "$RUN_DIR/nsys-export.log" 2>&1
"$PYTHON_BIN" -m benchmarks.profiling.vllm_prompt_embeddings.audit_nsys \
    "$REP" \
    "$SQLITE" \
    --summary "$RUN_DIR/summary.json" \
    --output "$RUN_DIR/nsys-audit.json"

sha256sum "$REP" "$SQLITE" "$RUN_DIR/summary.json" \
    > "$RUN_DIR/sha256sums.txt"
echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> "$RUN_DIR/provenance.txt"
echo "RUN_COMPLETE run_dir=$RUN_DIR rep=$REP sqlite=$SQLITE"
