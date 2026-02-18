#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# DIS-1381: Benchmark vllm processor frontend support
#
# Entry point: handles salloc allocation + orchestrates 3 benchmark scenarios.
# Usage:
#   bash bench/run.sh --account <ACCOUNT> [--partition batch] [--dry-run] [--scenario all]
#
# Scenarios:
#   1 - vllm serve (baseline)
#   2 - dynamo frontend (rust preprocessing) + dynamo vllm
#   3 - dynamo frontend --chat-processor vllm + dynamo vllm

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
SQSH="${SQSH:-/lustre/fsw/core_dlfw_ci/kprashanth/dynamo_vllm_bench.sqsh}"
MODEL="Qwen/Qwen3-8B"
ISL=1024
OSL=1024
CONCURRENCIES="1 2 4 8 16 32 64 128 256 512 1024"
REQ_MULTIPLIER=10
SCENARIOS="1 2 3"
DRY_RUN=false
ACCOUNT=""
PARTITION="batch"
TIME_LIMIT="06:00:00"
OUTPUT_DIR=""
RUN_NAME=""
EXTRA_SRUN_FLAGS="${EXTRA_SRUN_FLAGS:-}"
COOLDOWN_SECS=30

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# Scenario name mapping
# ---------------------------------------------------------------------------
scenario_name() {
    case "$1" in
        1) echo "scenario_1_vllm_serve" ;;
        2) echo "scenario_2_dynamo_rust" ;;
        3) echo "scenario_3_dynamo_vllm_processor" ;;
        *) echo "scenario_unknown" ;;
    esac
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
show_help() {
    cat <<'EOF'
Usage: bash bench/run.sh [OPTIONS]

Required:
  --account ACCOUNT         Slurm account for salloc

Options:
  --model MODEL             HF model name (default: Qwen/Qwen3-8B)
  --isl N                   Input sequence length (default: 1024)
  --osl N                   Output sequence length (default: 1024)
  --req-multiplier N        Requests per concurrency = C * N (default: 10)
  --name NAME               Run name (default: run_<timestamp>)
  --partition PARTITION     Slurm partition (default: batch)
  --time TIME               Slurm time limit (default: 06:00:00)
  --output-dir DIR          Results output directory (default: $PWD/bench_results)
  --scenario {1,2,3,all}    Which scenario(s) to run (default: all)
  --sqsh PATH               Container sqsh image path
  --dry-run                 Lightweight validation run (small model, concurrency 1)
  --help                    Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --account)      ACCOUNT="$2"; shift 2 ;;
        --model)        MODEL="$2"; shift 2 ;;
        --isl)          ISL="$2"; shift 2 ;;
        --osl)          OSL="$2"; shift 2 ;;
        --req-multiplier) REQ_MULTIPLIER="$2"; shift 2 ;;
        --name)         RUN_NAME="$2"; shift 2 ;;
        --partition)    PARTITION="$2"; shift 2 ;;
        --time)         TIME_LIMIT="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --sqsh)         SQSH="$2"; shift 2 ;;
        --scenario)
            if [[ "$2" == "all" ]]; then
                SCENARIOS="1 2 3"
            else
                SCENARIOS="$2"
            fi
            shift 2
            ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --help)         show_help ;;
        *)              echo "ERROR: Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$ACCOUNT" ]]; then
    echo "ERROR: --account is required" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Dry-run overrides
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" == "true" ]]; then
    MODEL="Qwen/Qwen3-0.6B"
    CONCURRENCIES="1"
    echo "=== DRY RUN MODE ==="
    echo "  Model:        $MODEL"
    echo "  Concurrencies: $CONCURRENCIES"
    echo "  (lightweight validation -- small model, minimal requests)"
    echo "===================="
fi

# ---------------------------------------------------------------------------
# Self-re-invoke under salloc if not already in an allocation
# ---------------------------------------------------------------------------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "No SLURM allocation detected. Requesting 2 nodes via salloc..."
    echo "  Account:   $ACCOUNT"
    echo "  Partition:  $PARTITION"
    echo "  Time limit: $TIME_LIMIT"

    # Rebuild the argument list for re-invocation
    RERUN_ARGS=(--account "$ACCOUNT" --model "$MODEL" --isl "$ISL" --osl "$OSL" --req-multiplier "$REQ_MULTIPLIER" --partition "$PARTITION" --time "$TIME_LIMIT")
    [[ -n "$RUN_NAME" ]] && RERUN_ARGS+=(--name "$RUN_NAME")
    [[ -n "$OUTPUT_DIR" ]] && RERUN_ARGS+=(--output-dir "$OUTPUT_DIR")
    RERUN_ARGS+=(--sqsh "$SQSH")
    RERUN_ARGS+=(--scenario "$SCENARIOS")
    [[ "$DRY_RUN" == "true" ]] && RERUN_ARGS+=(--dry-run)

    exec salloc \
        --nodes=2 \
        --partition="$PARTITION" \
        --account="$ACCOUNT" \
        --job-name="${ACCOUNT}-dis1381-bench" \
        -t "$TIME_LIMIT" \
        bash "$SCRIPT_DIR/run.sh" "${RERUN_ARGS[@]}"
fi

# ---------------------------------------------------------------------------
# We are inside a SLURM allocation from here on
# ---------------------------------------------------------------------------
echo "=== SLURM allocation active ==="
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node list: $SLURM_JOB_NODELIST"

# Parse node list into an array
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ ${#NODES[@]} -lt 2 ]]; then
    echo "ERROR: Need at least 2 nodes, got ${#NODES[@]}" >&2
    exit 1
fi
SERVER_NODE="${NODES[0]}"
CLIENT_NODE="${NODES[1]}"
echo "  Server node: $SERVER_NODE"
echo "  Client node: $CLIENT_NODE"

# ---------------------------------------------------------------------------
# Set up output directory
# ---------------------------------------------------------------------------
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$PWD/bench_results"
fi
if [[ -n "$RUN_NAME" ]]; then
    RUN_DIR="${OUTPUT_DIR}/${RUN_NAME}"
else
    RUN_DIR="${OUTPUT_DIR}/run_${TIMESTAMP}"
fi
mkdir -p "$RUN_DIR"
echo "  Results dir: $RUN_DIR"

# Dump run configuration
cat > "${RUN_DIR}/config.json" <<CONFIGEOF
{
  "slurm_job_id": "${SLURM_JOB_ID}",
  "server_node": "${SERVER_NODE}",
  "client_node": "${CLIENT_NODE}",
  "sqsh": "${SQSH}",
  "model": "${MODEL}",
  "isl": ${ISL},
  "osl": ${OSL},
  "req_multiplier": ${REQ_MULTIPLIER},
  "concurrencies": "$(echo $CONCURRENCIES | tr ' ' ',')",
  "scenarios": "$(echo $SCENARIOS | tr ' ' ',')",
  "dry_run": ${DRY_RUN},
  "timestamp": "${TIMESTAMP}"
}
CONFIGEOF

echo "  Config written to ${RUN_DIR}/config.json"

# ---------------------------------------------------------------------------
# Common srun prefix (array-based to preserve quoting)
# ---------------------------------------------------------------------------
SRUN_BASE=(
    srun --overlap --nodes=1 --ntasks=1
    --container-image="$SQSH"
    --container-mounts="${RUN_DIR}:${RUN_DIR},${SCRIPT_DIR}:${SCRIPT_DIR},${WORKSPACE_DIR}:/workspace"
)
# Append any user-supplied extra srun flags (word-split intentional here)
if [[ -n "$EXTRA_SRUN_FLAGS" ]]; then
    read -ra _extra <<< "$EXTRA_SRUN_FLAGS"
    SRUN_BASE+=("${_extra[@]}")
fi

# ---------------------------------------------------------------------------
# Run scenarios
# ---------------------------------------------------------------------------
for SCENARIO in $SCENARIOS; do
    SCENARIO_DIR="${RUN_DIR}/$(scenario_name "$SCENARIO")"
    LOG_DIR="${SCENARIO_DIR}/logs"
    METRICS_DIR="${SCENARIO_DIR}/metrics"
    AIPERF_DIR="${SCENARIO_DIR}/aiperf"
    mkdir -p "$LOG_DIR" "$METRICS_DIR" "$AIPERF_DIR"

    echo ""
    echo "============================================================"
    echo "  Scenario $SCENARIO: $(scenario_name "$SCENARIO")"
    echo "============================================================"

    # --- Launch server on node 0 (background) ---
    SERVER_CMD=(
        bash "${SCRIPT_DIR}/scripts/server.sh"
        --scenario "$SCENARIO"
        --model "$MODEL"
        --log-dir "$LOG_DIR"
    )
    [[ "$DRY_RUN" == "true" ]] && SERVER_CMD+=(--dry-run)

    echo "[run.sh] Starting server on $SERVER_NODE ..."
    # Use process substitution (not a pipe) so $! captures srun's PID.
    # A pipe would make $! point to tee, leaving srun unkillable.
    "${SRUN_BASE[@]}" --nodelist="$SERVER_NODE" "${SERVER_CMD[@]}" \
        > >(tee "${LOG_DIR}/server_srun.log") 2>&1 &
    SERVER_PID=$!
    echo "[run.sh] Server srun PID: $SERVER_PID"

    # --- Launch client on node 1 (blocking) ---
    CLIENT_CMD=(
        bash "${SCRIPT_DIR}/scripts/client.sh"
        --server-host "$SERVER_NODE"
        --scenario "$SCENARIO"
        --model "$MODEL"
        --isl "$ISL"
        --osl "$OSL"
        --concurrencies "$CONCURRENCIES"
        --req-multiplier "$REQ_MULTIPLIER"
        --output-dir "$SCENARIO_DIR"
    )
    [[ "$DRY_RUN" == "true" ]] && CLIENT_CMD+=(--dry-run)

    echo "[run.sh] Starting client on $CLIENT_NODE ..."
    # Disable errexit so we always clean up the server even if client fails.
    # Use PIPESTATUS[0] to capture srun's exit code (not tee's).
    set +e
    "${SRUN_BASE[@]}" --nodelist="$CLIENT_NODE" "${CLIENT_CMD[@]}" \
        2>&1 | tee "${LOG_DIR}/client.log"
    CLIENT_EXIT=${PIPESTATUS[0]}
    set -e

    echo "[run.sh] Client finished (exit=$CLIENT_EXIT). Stopping server..."

    # --- Stop server ---
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    echo "[run.sh] Server stopped."

    if [[ "$CLIENT_EXIT" -ne 0 ]]; then
        echo "WARNING: Client exited with code $CLIENT_EXIT for scenario $SCENARIO" >&2
    fi

    # --- Cooldown between scenarios ---
    if [[ "$SCENARIO" != "$(echo $SCENARIOS | awk '{print $NF}')" ]]; then
        echo "[run.sh] Cooldown ${COOLDOWN_SECS}s before next scenario..."
        sleep "$COOLDOWN_SECS"
    fi
done

echo ""
echo "============================================================"
echo "  All scenarios complete. Results in: $RUN_DIR"
echo "============================================================"
