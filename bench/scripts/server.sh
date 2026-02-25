#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# DIS-1381: Server-side deployment launcher.
# Runs inside the container on the server node (node 0).
#
# Launches the appropriate server process(es) for a given scenario and blocks
# until SIGTERM is received (from run.sh killing this srun).

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
SCENARIO=""
MODEL=""
LOG_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario)  SCENARIO="$2"; shift 2 ;;
        --model)     MODEL="$2"; shift 2 ;;
        --log-dir)   LOG_DIR="$2"; shift 2 ;;
        --dry-run)   shift ;;  # accepted but ignored
        *)           echo "ERROR: server.sh: Unknown option: $1" >&2; exit 1 ;;
    esac
done

for var in SCENARIO MODEL LOG_DIR; do
    if [[ -z "${!var}" ]]; then
        echo "ERROR: server.sh: --$(echo $var | tr '[:upper:]' '[:lower:]' | tr '_' '-') is required" >&2
        exit 1
    fi
done

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Cleanup trap -- kill all child processes on exit
# ---------------------------------------------------------------------------
CHILD_PIDS=()

cleanup() {
    echo "[server.sh] Cleaning up (scenario=$SCENARIO)..."
    for pid in "${CHILD_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "[server.sh] Cleanup done."
}
trap cleanup EXIT SIGTERM SIGINT

# ---------------------------------------------------------------------------
# CPU snapshot collector (background)
# ---------------------------------------------------------------------------
start_cpu_snapshots() {
    local outfile="$LOG_DIR/cpu_snapshots.log"
    (
        while true; do
            echo "=== $(date -Iseconds) ===" >> "$outfile"
            top -bcn1 -o %CPU | head -30 >> "$outfile" 2>/dev/null || true
            sleep 30
        done
    ) &
    CHILD_PIDS+=($!)
    echo "[server.sh] CPU snapshot collector started (PID ${CHILD_PIDS[-1]})"
}

# ---------------------------------------------------------------------------
# Helper: wait for a local TCP port to be listening
# ---------------------------------------------------------------------------
wait_for_port() {
    local port="$1"
    local label="$2"
    local max_attempts=60
    for ((i=1; i<=max_attempts; i++)); do
        if bash -c "echo >/dev/tcp/127.0.0.1/$port" 2>/dev/null; then
            echo "[server.sh] $label is listening on port $port (attempt $i)"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: $label did not start on port $port after ${max_attempts}s" >&2
    return 1
}

# ---------------------------------------------------------------------------
# Scenario launchers
# ---------------------------------------------------------------------------
launch_scenario_1() {
    echo "[server.sh] Scenario 1: vllm serve $MODEL --port 8000"
    vllm serve "$MODEL" --port 8000 &
    CHILD_PIDS+=($!)
    echo "[server.sh] vllm serve PID: ${CHILD_PIDS[-1]}"
}

launch_scenario_2() {
    echo "[server.sh] Scenario 2: dynamo frontend (rust) + dynamo vllm"

    # Start supporting services
    nats-server -js &
    CHILD_PIDS+=($!)
    echo "[server.sh] nats-server PID: ${CHILD_PIDS[-1]}"

    etcd \
        --listen-client-urls http://0.0.0.0:2379 \
        --advertise-client-urls http://0.0.0.0:2379 \
        --data-dir /tmp/etcd \
        > "$LOG_DIR/etcd.log" 2>&1 &
    CHILD_PIDS+=($!)
    echo "[server.sh] etcd PID: ${CHILD_PIDS[-1]}"

    # Wait for etcd + nats before launching dynamo
    wait_for_port 2379 "etcd"
    wait_for_port 4222 "nats-server"

    # Start dynamo frontend
    python -m dynamo.frontend &
    CHILD_PIDS+=($!)
    echo "[server.sh] dynamo.frontend PID: ${CHILD_PIDS[-1]}"

    # Start dynamo vllm worker
    DYN_SYSTEM_PORT=8081 python -m dynamo.vllm \
        --model "$MODEL" --connector none &
    CHILD_PIDS+=($!)
    echo "[server.sh] dynamo.vllm PID: ${CHILD_PIDS[-1]}"
}

launch_scenario_3() {
    echo "[server.sh] Scenario 3: dynamo frontend --chat-processor vllm + dynamo vllm"

    # Start supporting services
    nats-server -js &
    CHILD_PIDS+=($!)
    echo "[server.sh] nats-server PID: ${CHILD_PIDS[-1]}"

    etcd \
        --listen-client-urls http://0.0.0.0:2379 \
        --advertise-client-urls http://0.0.0.0:2379 \
        --data-dir /tmp/etcd \
        > "$LOG_DIR/etcd.log" 2>&1 &
    CHILD_PIDS+=($!)
    echo "[server.sh] etcd PID: ${CHILD_PIDS[-1]}"

    # Wait for etcd + nats before launching dynamo
    wait_for_port 2379 "etcd"
    wait_for_port 4222 "nats-server"

    # Start dynamo frontend with vllm processor
    python -m dynamo.frontend \
        --dyn-chat-processor vllm \
        --dyn-preprocess-workers 32 \
        --model "$MODEL" &
    CHILD_PIDS+=($!)
    echo "[server.sh] dynamo.frontend (vllm processor) PID: ${CHILD_PIDS[-1]}"

    # Start dynamo vllm worker
    DYN_SYSTEM_PORT=8081 python -m dynamo.vllm \
        --model "$MODEL" --connector none &
    CHILD_PIDS+=($!)
    echo "[server.sh] dynamo.vllm PID: ${CHILD_PIDS[-1]}"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "[server.sh] Starting scenario $SCENARIO"
echo "[server.sh] Model: $MODEL"
echo "[server.sh] Log dir: $LOG_DIR"

# ---------------------------------------------------------------------------
# Install mounted workspace version of dynamo
# ---------------------------------------------------------------------------
echo "[server.sh] Installing mounted dynamo from /workspace ..."
(
    cd /workspace/lib/bindings/python && maturin develop --release --uv
    cd /workspace && uv pip install -e .
)
echo "[server.sh] Dynamo install complete."

# Clean up any leftover etcd data from previous runs
rm -rf /tmp/etcd

start_cpu_snapshots

case "$SCENARIO" in
    1) launch_scenario_1 ;;
    2) launch_scenario_2 ;;
    3) launch_scenario_3 ;;
    *) echo "ERROR: Unknown scenario: $SCENARIO" >&2; exit 1 ;;
esac

echo "[server.sh] All processes launched. Waiting for termination signal..."

# Wait for any child to exit (or for SIGTERM from run.sh)
# If any critical process dies, we exit so run.sh can detect it.
wait -n "${CHILD_PIDS[@]}" 2>/dev/null || true
echo "[server.sh] A child process exited. Shutting down."
