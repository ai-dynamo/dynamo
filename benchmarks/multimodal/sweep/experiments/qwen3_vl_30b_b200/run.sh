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
#   bash run.sh --account <ACCOUNT> --config epd-1e --concurrencies "64 32 16"
#   bash run.sh --account <ACCOUNT> --dry-run   # cycles ALL configs, 1 request each
#
# The script self-allocates via salloc if not already inside a Slurm job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ─────────────────────────────────────────────────────────
ACCOUNT=""
CONFIG=""
MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
CONCURRENCIES="1 8 16 32 64"
OSL=500
REQUEST_COUNT=100
WARMUP_COUNT=5
PARTITION="batch"
TIME="04:00:00"
SQSH="/lustre/fsw/core_dlfw_ci/kprashanth/trtllm_bench_rc7_post1.sqsh"
WORKSPACE="/lustre/fsw/core_dlfw_ci/kprashanth/dynamo"
OUTPUT_DIR="benchmarks/results/qwen3_vl_30b_b200/agg_vs_epd"
DRY_RUN=false
INPUT_FILE="benchmarks/multimodal/jsonl/1000req_1img_200pool_400word_http.jsonl"
TIMEOUT=900

VALID_CONFIGS="agg epd-1e epd-2e epd-4e epd-6e"
ALL_CONFIGS=("agg" "epd-1e" "epd-2e" "epd-4e" "epd-6e")

# ── Argument Parsing ─────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: bash run.sh --account ACCOUNT [--config CONFIG | --dry-run] [OPTIONS]

Required:
  --account ACCOUNT       Slurm account

One of:
  --config CONFIG         Benchmark config: agg, epd-1e, epd-2e, epd-4e, epd-6e
  --dry-run               Cycle ALL configs with 1 request each to validate GPU placement

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

if ! $DRY_RUN && [[ -z "$CONFIG" ]]; then
    echo "ERROR: --config is required (one of: $VALID_CONFIGS), or use --dry-run"
    exit 1
fi

if [[ -n "$CONFIG" ]] && ! echo "$VALID_CONFIGS" | grep -qw "$CONFIG"; then
    echo "ERROR: Invalid config '$CONFIG'. Must be one of: $VALID_CONFIGS"
    exit 1
fi

if $DRY_RUN; then
    CONCURRENCIES="1"
    REQUEST_COUNT=1
    WARMUP_COUNT=1
    OSL=50
    OUTPUT_DIR="benchmarks/results/qwen3_vl_30b_b200/dry_run"
fi

# ── Helper: map config name to launch script ─────────────────────────
config_to_script() {
    local cfg="$1"
    case "$cfg" in
        agg)     echo "benchmarks/multimodal/sweep/experiments/qwen3_vl_30b_b200/launch_agg.sh" ;;
        epd-*)   echo "benchmarks/multimodal/sweep/experiments/qwen3_vl_30b_b200/launch_${cfg//-/_}.sh" ;;
    esac
}

# ── Self-allocate if not inside Slurm ────────────────────────────────
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "======================================================================="
    if $DRY_RUN; then
        echo "  TRTLLM MM Benchmark: DRY RUN (all configs)"
    else
        echo "  TRTLLM MM Benchmark: $CONFIG"
    fi
    echo "======================================================================="
    echo "  Account:       $ACCOUNT"
    echo "  Partition:     $PARTITION"
    echo "  Time:          $TIME"
    echo "  Config:        ${CONFIG:-all (dry-run)}"
    echo "  Model:         $MODEL"
    echo "  Concurrencies: $CONCURRENCIES"
    echo "  OSL:           $OSL"
    echo "  Requests:      $REQUEST_COUNT"
    echo "  Dry run:       $DRY_RUN"
    echo "  Container:     $SQSH"
    echo ""

    # Rebuild args to forward to salloc
    ARGS=(--account "$ACCOUNT" --model "$MODEL"
          --concurrencies "$CONCURRENCIES" --osl "$OSL"
          --request-count "$REQUEST_COUNT" --warmup-count "$WARMUP_COUNT"
          --partition "$PARTITION" --time "$TIME" --sqsh "$SQSH"
          --workspace "$WORKSPACE" --output-dir "$OUTPUT_DIR"
          --input-file "$INPUT_FILE" --timeout "$TIMEOUT")
    [[ -n "$CONFIG" ]] && ARGS+=(--config "$CONFIG")
    $DRY_RUN && ARGS+=(--dry-run)

    echo "Allocating node via salloc..."
    exec salloc \
        --partition="$PARTITION" \
        --account="$ACCOUNT" \
        --job-name="${ACCOUNT}-trtllm-bench-${CONFIG:-dryrun}" \
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
if $DRY_RUN; then
    echo "  Inside container — DRY RUN (all configs)"
else
    echo "  Inside container — building and running: $CONFIG"
fi
echo "======================================================================="

cd /workspace
unset SLURM_NODELIST SLURM_JOBID 2>/dev/null || true

# ── Setup ─────────────────────────────────────────────────────────────
echo ""
echo "[setup] Checking out branch..."
git fetch origin && git checkout kprashanth/trtllm-bench

echo "[setup] Starting infra services..."
nats-server -js &
etcd --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://0.0.0.0:2379 \
     --data-dir /tmp/etcd &
sleep 2

echo "[setup] Sanity check..."
deploy/sanity_check.py --runtime-check-only

echo "[setup] Installing benchmark dependencies..."
uv pip install aiperf psutil

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
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

# ── Helper: generate and run a sweep YAML ────────────────────────────
run_sweep_config() {
    local cfg="$1"
    local launch_script
    launch_script=$(config_to_script "$cfg")

    # Convert "1 8 16 32" -> "[1,8,16,32]"
    local conc_yaml="[$(echo "$CONCURRENCIES" | tr ' ' ',')]"

    local run_yaml="/tmp/sweep_${cfg}.yaml"
    cat > "$run_yaml" <<YAML
model: $MODEL
concurrencies: $conc_yaml
osl: $OSL
request_count: $REQUEST_COUNT
warmup_count: $WARMUP_COUNT
timeout: $TIMEOUT
skip_plots: true
output_dir: $OUTPUT_DIR

input_files:
  - $INPUT_FILE

configs:
  - label: $cfg
    workflow: $launch_script
YAML

    echo ""
    echo "[sweep] Config: $cfg"
    cat "$run_yaml"
    echo ""

    python -m benchmarks.multimodal.sweep --config "$run_yaml"

    # Post-run GPU snapshot to verify memory was fully released
    echo ""
    echo "[gpu] Post-cleanup GPU state after $cfg:"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
    echo ""
}

# ── Run ──────────────────────────────────────────────────────────────
if $DRY_RUN; then
    echo "======================================================================="
    echo "  DRY RUN: cycling through all configs"
    echo "======================================================================="

    for cfg in "${ALL_CONFIGS[@]}"; do
        echo ""
        echo "--- DRY RUN: $cfg ---"
        run_sweep_config "$cfg"
    done

    echo ""
    echo "======================================================================="
    echo "  DRY RUN complete — all configs validated"
    echo "  Results: /workspace/$OUTPUT_DIR"
    echo "======================================================================="
else
    echo "======================================================================="
    echo "  Starting sweep: $CONFIG"
    echo "======================================================================="

    run_sweep_config "$CONFIG"

    echo ""
    echo "======================================================================="
    echo "  Sweep complete: $CONFIG"
    echo "  Results: /workspace/$OUTPUT_DIR"
    echo "======================================================================="
fi
