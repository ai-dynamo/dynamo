#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# End-to-end benchmark runner for TRTLLM MM Agg vs EPD on B200.
#
# Handles salloc, srun, build, JSONL generation, and sweep execution.
# Designed to be invoked once per config (agg, epd-1e, etc.) since each
# full concurrency sweep takes ~4 hours.
#
# Usage:
#   bash run.sh --account <ACCOUNT> --config agg
#   bash run.sh --account <ACCOUNT> --config epd-1e --concurrencies "16 32 64 128 256"
#   bash run.sh --account <ACCOUNT> --config epd-2e --dry-run
#
# The script self-allocates via salloc if not already inside a Slurm job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ─────────────────────────────────────────────────────────
ACCOUNT=""
CONFIG=""
MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
CONCURRENCIES="1 8 16 32 64 128 256"
OSL=500
REQUEST_COUNT=1000
WARMUP_COUNT=5
PARTITION="batch"
TIME="04:00:00"
SQSH="/lustre/fsw/core_dlfw_ci/kprashanth/trtllm_bench_rc7.sqsh"
WORKSPACE="/lustre/fsw/core_dlfw_ci/kprashanth/dynamo"
OUTPUT_DIR="benchmarks/results/qwen3_vl_30b_b200/agg_vs_epd"
DRY_RUN=false
INPUT_FILE="benchmarks/multimodal/jsonl/1000req_1img_200pool_400word_http.jsonl"
TIMEOUT=900

VALID_CONFIGS="agg epd-1e epd-2e epd-4e epd-6e"

# ── Argument Parsing ─────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: bash run.sh --account ACCOUNT --config CONFIG [OPTIONS]

Required:
  --account ACCOUNT       Slurm account
  --config CONFIG         Benchmark config: agg, epd-1e, epd-2e, epd-4e, epd-6e

Options:
  --model MODEL           HuggingFace model name [default: $MODEL]
  --concurrencies "C .."  Space-separated concurrency levels [default: $CONCURRENCIES]
  --osl N                 Output sequence length [default: $OSL]
  --request-count N       Requests per concurrency level [default: $REQUEST_COUNT]
  --warmup-count N        Warmup requests per concurrency level [default: $WARMUP_COUNT]
  --partition PARTITION   Slurm partition [default: $PARTITION]
  --time TIME             Slurm time limit [default: $TIME]
  --sqsh PATH             Container image path [default: $SQSH]
  --workspace PATH        Host path to dynamo repo [default: $WORKSPACE]
  --output-dir DIR        Results directory (relative to workspace) [default: $OUTPUT_DIR]
  --input-file FILE       JSONL input file (relative to workspace) [default: $INPUT_FILE]
  --timeout N             Server startup timeout in seconds [default: $TIMEOUT]
  --dry-run               Quick validation: concurrency=1, 3 requests
  --help                  Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --account)      ACCOUNT="$2"; shift 2 ;;
        --config)       CONFIG="$2"; shift 2 ;;
        --model)        MODEL="$2"; shift 2 ;;
        --concurrencies) CONCURRENCIES="$2"; shift 2 ;;
        --osl)          OSL="$2"; shift 2 ;;
        --request-count) REQUEST_COUNT="$2"; shift 2 ;;
        --warmup-count) WARMUP_COUNT="$2"; shift 2 ;;
        --partition)    PARTITION="$2"; shift 2 ;;
        --time)         TIME="$2"; shift 2 ;;
        --sqsh)         SQSH="$2"; shift 2 ;;
        --workspace)    WORKSPACE="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --input-file)   INPUT_FILE="$2"; shift 2 ;;
        --timeout)      TIMEOUT="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --help)         usage ;;
        *)              echo "Unknown flag: $1"; usage ;;
    esac
done

# ── Validate ─────────────────────────────────────────────────────────
if [[ -z "$ACCOUNT" ]]; then
    echo "ERROR: --account is required"
    exit 1
fi

if [[ -z "$CONFIG" ]]; then
    echo "ERROR: --config is required (one of: $VALID_CONFIGS)"
    exit 1
fi

if ! echo "$VALID_CONFIGS" | grep -qw "$CONFIG"; then
    echo "ERROR: Invalid config '$CONFIG'. Must be one of: $VALID_CONFIGS"
    exit 1
fi

if $DRY_RUN; then
    CONCURRENCIES="1"
    REQUEST_COUNT=3
    WARMUP_COUNT=1
    OSL=50
fi

# Map config name to launch script
case "$CONFIG" in
    agg)     LAUNCH_SCRIPT="benchmarks/multimodal/sweep/experiments/qwen3_vl_30b_b200/launch_agg.sh" ;;
    epd-*)   LAUNCH_SCRIPT="benchmarks/multimodal/sweep/experiments/qwen3_vl_30b_b200/launch_${CONFIG//-/_}.sh" ;;
esac

# ── Self-allocate if not inside Slurm ────────────────────────────────
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "======================================================================="
    echo "  TRTLLM MM Benchmark: $CONFIG"
    echo "======================================================================="
    echo "  Account:       $ACCOUNT"
    echo "  Partition:     $PARTITION"
    echo "  Time:          $TIME"
    echo "  Config:        $CONFIG"
    echo "  Model:         $MODEL"
    echo "  Concurrencies: $CONCURRENCIES"
    echo "  OSL:           $OSL"
    echo "  Requests:      $REQUEST_COUNT"
    echo "  Dry run:       $DRY_RUN"
    echo "  Container:     $SQSH"
    echo ""

    # Rebuild args to forward to salloc
    ARGS=(--account "$ACCOUNT" --config "$CONFIG" --model "$MODEL"
          --concurrencies "$CONCURRENCIES" --osl "$OSL"
          --request-count "$REQUEST_COUNT" --warmup-count "$WARMUP_COUNT"
          --partition "$PARTITION" --time "$TIME" --sqsh "$SQSH"
          --workspace "$WORKSPACE" --output-dir "$OUTPUT_DIR"
          --input-file "$INPUT_FILE" --timeout "$TIMEOUT")
    $DRY_RUN && ARGS+=(--dry-run)

    echo "Allocating node via salloc..."
    exec salloc \
        --partition="$PARTITION" \
        --account="$ACCOUNT" \
        --job-name="${ACCOUNT}-trtllm-bench-${CONFIG}" \
        -t "$TIME" \
        srun --mpi=pmix \
            --container-image="$SQSH" \
            --container-mounts="${WORKSPACE}:/workspace" \
            -A "$ACCOUNT" \
            --pty bash /workspace/benchmarks/multimodal/sweep/experiments/qwen3_vl_30b_b200/run.sh "${ARGS[@]}"
fi

# ── Inside container ─────────────────────────────────────────────────
echo ""
echo "======================================================================="
echo "  Inside container — building and running: $CONFIG"
echo "======================================================================="

cd /workspace
unset SLURM_NODELIST SLURM_JOBID 2>/dev/null || true

# ── Build ────────────────────────────────────────────────────────────
echo ""
echo "[build] Checking out branch..."
git fetch origin && git checkout kprashanth/trtllm-bench

echo "[build] Starting infra services..."
nats-server -js &
etcd --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://0.0.0.0:2379 \
     --data-dir /tmp/etcd &
sleep 2

echo "[build] Building dynamo..."
cd lib/bindings/python && maturin develop --release --uv && cd /workspace
uv pip install -e .

echo "[build] Sanity check..."
deploy/sanity_check.py --runtime-check-only

echo "[build] Installing aiperf..."
uv pip install aiperf

# ── Generate input data if missing ───────────────────────────────────
if [[ ! -f "/workspace/$INPUT_FILE" ]]; then
    echo "[data] Generating JSONL input file..."
    cd /workspace/benchmarks/multimodal/jsonl
    python main.py -n 1000 --images-per-request 1 --images-pool 200 \
        --user-text-tokens 400 --image-mode http
    cd /workspace
else
    echo "[data] Input file exists: $INPUT_FILE"
fi

# ── GPU check ────────────────────────────────────────────────────────
echo ""
echo "[gpu] Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── Generate per-run sweep YAML ──────────────────────────────────────
# Convert "1 8 16 32" -> "[1, 8, 16, 32]"
CONC_YAML="[$(echo "$CONCURRENCIES" | tr ' ' ',')]"

RUN_YAML="/tmp/sweep_${CONFIG}.yaml"
cat > "$RUN_YAML" <<YAML
model: $MODEL
concurrencies: $CONC_YAML
osl: $OSL
request_count: $REQUEST_COUNT
warmup_count: $WARMUP_COUNT
timeout: $TIMEOUT
output_dir: $OUTPUT_DIR

input_files:
  - $INPUT_FILE

configs:
  - label: $CONFIG
    workflow: $LAUNCH_SCRIPT
YAML

echo ""
echo "[sweep] Generated config:"
cat "$RUN_YAML"
echo ""

# ── Run sweep ────────────────────────────────────────────────────────
echo "======================================================================="
echo "  Starting sweep: $CONFIG"
echo "======================================================================="

python -m benchmarks.multimodal.sweep --config "$RUN_YAML"

echo ""
echo "======================================================================="
echo "  Sweep complete: $CONFIG"
echo "  Results: /workspace/$OUTPUT_DIR"
echo "======================================================================="
