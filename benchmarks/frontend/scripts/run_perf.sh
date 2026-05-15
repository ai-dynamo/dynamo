#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unified observability capture — starts services, collects ALL observability
# data (nsys, perf, BPF, system stats, Prometheus), runs aiperf load, then
# shuts down and exports results.
#
# Usage:
#   ./run_perf.sh --model Qwen/Qwen3-0.6B --concurrency 64 --num-requests 4096
#   ./run_perf.sh --skip-bpf --skip-nsys   # opt-out of heavy tools
#   sudo ./run_perf.sh --skip-nsys --skip-perf \       # run bpf tracing as root
#        --model Qwen/Qwen3-0.6B --concurrency 64 --num-requests 4096
#
# Output: artifacts/obs_YYYYMMDD_HHMMSS/ with subdirs for each data source.
#
# Prerequisites:
#   - dynamo.mocker and dynamo.frontend installed
#   - aiperf installed
#   - Optional: nsys, perf, bpftrace, flamegraph tools (auto-detected)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../../.."

# Raise file descriptor limit — high concurrency runs open many sockets
# (aiperf connections + frontend + mocker workers + /proc polling + captures).
# The default 1024 is insufficient at concurrency >= 100.
ulimit -n 65536 2>/dev/null || ulimit -n 8192 2>/dev/null || true

# When running under sudo, preserve the invoking user's environment:
# - HF cache (so models downloaded as the regular user are visible)
# - PATH (so aiperf, python, etc. are found)
if [[ -n "${SUDO_USER:-}" ]]; then
    REAL_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
    export HF_HOME="${HF_HOME:-${REAL_HOME}/.cache/huggingface}"
    # Add common user binary paths that sudo strips
    for p in "${REAL_HOME}/.local/bin" "${REPO_ROOT}/dynamo/bin"; do
        [[ -d "$p" ]] && [[ ":$PATH:" != *":$p:"* ]] && export PATH="$p:$PATH"
    done
fi

# ─── Defaults ───────────────────────────────────────────────────────────────
MODEL="${MODEL:-nvidia/Llama-3.1-8B-Instruct-FP8}"
MODEL_NAME=""
NUM_WORKERS="${NUM_WORKERS:-2}"
SPEEDUP_RATIO="${SPEEDUP_RATIO:-1.0}"
CONCURRENCY="${CONCURRENCY:-64}"
NUM_REQUESTS="${NUM_REQUESTS:-}"
ISL="${ISL:-1024}"
OSL="${OSL:-256}"
REQUEST_PLANE="${REQUEST_PLANE:-tcp}"
EVENT_PLANE="${EVENT_PLANE:-nats}"
CAPTURE_DURATION="${CAPTURE_DURATION:-60}"
OUTPUT_DIR=""
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
FRONTEND_PORT="${FRONTEND_PORT:-8000}"
PLANNER_PROFILE="${PLANNER_PROFILE:-}"
TOKENIZER_BACKEND="${TOKENIZER_BACKEND:-}"
NUM_MODELS="${NUM_MODELS:-1}"                 # Number of model instances (each gets NUM_WORKERS workers)
AIPERF_TARGETS="${AIPERF_TARGETS:-first}"     # "first" = model-1 only, "all" = one aiperf run per model
BENCHMARK_DURATION="${BENCHMARK_DURATION:-}"  # aiperf --benchmark-duration (seconds)
REQUEST_RATE="${REQUEST_RATE:-}"              # aiperf --request-rate (requests/sec)
WARMUP_DURATION="${WARMUP_DURATION:-}"        # aiperf --warmup-duration (seconds)
WARMUP_COUNT="${WARMUP_COUNT:-}"              # aiperf --warmup-request-count

# Opt-out flags
SKIP_BPF=false
SKIP_NSYS=false
SKIP_FLAMEGRAPH=false
SKIP_PERF=false

# Optional captures
ENABLE_TCPDUMP=false
TCPDUMP_PORT=""

# Custom tool paths
NSYS_PATH=""

# ─── Argument parsing ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)                MODEL="$2"; shift 2 ;;
        --model-name)           MODEL_NAME="$2"; shift 2 ;;
        --workers)              NUM_WORKERS="$2"; shift 2 ;;
        --speedup-ratio)        SPEEDUP_RATIO="$2"; shift 2 ;;
        --concurrency)          CONCURRENCY="$2"; shift 2 ;;
        --num-requests)         NUM_REQUESTS="$2"; shift 2 ;;
        --isl)                  ISL="$2"; shift 2 ;;
        --osl)                  OSL="$2"; shift 2 ;;
        --request-plane)        REQUEST_PLANE="$2"; shift 2 ;;
        --event-plane)          EVENT_PLANE="$2"; shift 2 ;;
        --capture-duration)     CAPTURE_DURATION="$2"; shift 2 ;;
        --output-dir)           OUTPUT_DIR="$2"; shift 2 ;;
        --data-parallel-size)   DATA_PARALLEL_SIZE="$2"; shift 2 ;;
        --frontend-port)        FRONTEND_PORT="$2"; shift 2 ;;
        --planner-profile)      PLANNER_PROFILE="$2"; shift 2 ;;
        --tokenizer-backend)    TOKENIZER_BACKEND="$2"; shift 2 ;;
        --fast-tokens)          TOKENIZER_BACKEND="fast"; shift ;;
        --num-models)           NUM_MODELS="$2"; shift 2 ;;
        --aiperf-targets)       AIPERF_TARGETS="$2"; shift 2 ;;
        --benchmark-duration)   BENCHMARK_DURATION="$2"; shift 2 ;;
        --request-rate)         REQUEST_RATE="$2"; shift 2 ;;
        --warmup-duration)      WARMUP_DURATION="$2"; shift 2 ;;
        --warmup-count)         WARMUP_COUNT="$2"; shift 2 ;;
        --skip-bpf)             SKIP_BPF=true; shift ;;
        --skip-nsys)            SKIP_NSYS=true; shift ;;
        --skip-flamegraph)      SKIP_FLAMEGRAPH=true; shift ;;
        --skip-perf)            SKIP_PERF=true; shift ;;
        --tcpdump)              ENABLE_TCPDUMP=true; shift ;;
        --tcpdump-port)         TCPDUMP_PORT="$2"; shift 2 ;;
        --nsys-path)            NSYS_PATH="$2"; shift 2 ;;
        -h|--help)
            cat <<'USAGE'
Usage: run_perf.sh [OPTIONS]

Starts mocker + frontend, runs parallel observability capture, then aiperf load.

Service Options:
  --model PATH              Model path (default: nvidia/Llama-3.1-8B-Instruct-FP8)
  --model-name NAME         Served model name (default: same as --model)
  --workers N               Number of mocker workers (default: 2)
  --speedup-ratio RATIO     Mocker speedup ratio (default: 1.0; use large value for near-instant)
  --data-parallel-size N    Mocker DP workers (default: 1)
  --request-plane PLANE     nats|http|tcp (default: tcp)
  --event-plane PLANE       nats|zmq (default: nats)
  --frontend-port PORT      Frontend HTTP port (default: 8000)
  --planner-profile PATH    Planner profile data path
  --tokenizer-backend NAME  Tokenizer backend: "fast" or "hf" (default: unset, uses HF)
  --fast-tokens             Shorthand for --tokenizer-backend fast
  --num-models N            Number of model instances (default: 1). Each gets --workers workers
                            with names model-1, model-2, ...
  --aiperf-targets MODE     "first" (default): aiperf targets model-1 only.
                            "all": run aiperf sequentially for each model.
  --benchmark-duration N    aiperf run duration in seconds (default: use --num-requests)
  --request-rate N          Target requests per second (aiperf --request-rate)
  --warmup-duration N       aiperf warmup phase duration in seconds
  --warmup-count N          aiperf warmup request count (default: concurrency)

Load Options:
  --concurrency N           aiperf concurrency (default: 64)
  --num-requests N          Total requests (default: 640)
  --isl N                   Input sequence length (default: 1024)
  --osl N                   Output sequence length (default: 256)

Capture Options:
  --capture-duration N      Duration for parallel captures in seconds (default: 60)
  --output-dir DIR          Output directory (default: auto timestamped)
  --skip-bpf                Skip BPF tracing
  --skip-nsys               Skip Nsight Systems profiling
  --skip-flamegraph         Skip flamegraph generation
  --skip-perf               Skip perf record/stat
  --tcpdump                 Enable packet capture via tcpdump
  --tcpdump-port PORT       Port filter for tcpdump (default: --frontend-port)
  --nsys-path PATH          Path to nsys binary (default: auto-detected from PATH)
USAGE
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Default model-name to model if not set
[[ -z "$MODEL_NAME" ]] && MODEL_NAME="$MODEL"

# For multi-model: build array of model names.
# Each model group registers under its own served name (model-1, model-2, ...)
# but uses the same --model-path for weights. The mocker publishes source_path
# (the real HF model ID) in the ModelDeploymentCard so the frontend can resolve
# tokenizer/config even when the served name differs.
MODEL_NAMES=()
if [[ "$NUM_MODELS" -le 1 ]]; then
    MODEL_NAMES=("$MODEL_NAME")
else
    for m in $(seq 1 "$NUM_MODELS"); do
        MODEL_NAMES+=("model-${m}")
    done
fi

# Default tcpdump port to frontend port
[[ -z "$TCPDUMP_PORT" ]] && TCPDUMP_PORT="$FRONTEND_PORT"

# Auto-sync capture-duration to cover the full benchmark window.
# If benchmark-duration is set and capture-duration wasn't explicitly overridden,
# extend captures to benchmark-duration + warmup headroom + buffer.
if [[ -n "$BENCHMARK_DURATION" ]] && [[ "$CAPTURE_DURATION" -eq 60 ]]; then
    _WARMUP_HEADROOM=${WARMUP_DURATION:-10}
    CAPTURE_DURATION=$(( BENCHMARK_DURATION + _WARMUP_HEADROOM + 5 ))
    echo "  Auto-adjusted capture-duration to ${CAPTURE_DURATION}s (benchmark=${BENCHMARK_DURATION}s + warmup + 5s buffer)"
fi

# Resolve nsys binary path
if [[ -n "$NSYS_PATH" ]]; then
    if [[ ! -x "$NSYS_PATH" ]]; then
        echo "ERROR: --nsys-path '$NSYS_PATH' is not executable"; exit 1
    fi
    NSYS_CMD="$NSYS_PATH"
else
    NSYS_CMD="nsys"
fi

# ─── Output directory ────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${REPO_ROOT}/artifacts/obs_${TIMESTAMP}"
fi
mkdir -p "$OUTPUT_DIR"/{aiperf,nsys,perf,bpf,system,prometheus,logs}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Unified Observability Capture                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Output:    $OUTPUT_DIR"
echo "Tokenizer: ${TOKENIZER_BACKEND:-hf (default)}"
echo ""

# ─── Pre-flight: detect available tools ──────────────────────────────────────
echo "--- Pre-flight checks ---"

HAS_NSYS=false
HAS_PERF=false
HAS_BPF=false
HAS_FLAMEGRAPH_RENDERER=false

if command -v "$NSYS_CMD" &>/dev/null && [[ "$SKIP_NSYS" == false ]]; then
    HAS_NSYS=true
    echo "  nsys:       $("$NSYS_CMD" --version 2>/dev/null | head -1)"
else
    echo "  nsys:       SKIP"
fi

if command -v perf &>/dev/null && [[ "$SKIP_PERF" == false ]]; then
    HAS_PERF=true
    echo "  perf:       $(perf version 2>/dev/null | head -1)"
else
    echo "  perf:       SKIP"
fi

if command -v bpftrace &>/dev/null && [[ "$SKIP_BPF" == false ]]; then
    # BPF needs root or both CAP_BPF and CAP_PERFMON
    if [[ $(id -u) -eq 0 ]]; then
        HAS_BPF=true
        echo "  bpftrace:   available (root)"
    elif command -v capsh &>/dev/null \
         && capsh --print 2>/dev/null | grep '^Current:' | grep -q cap_bpf \
         && capsh --print 2>/dev/null | grep '^Current:' | grep -q cap_perfmon; then
        HAS_BPF=true
        echo "  bpftrace:   available (CAP_BPF + CAP_PERFMON)"
    else
        echo "  bpftrace:   SKIP (needs root or CAP_BPF + CAP_PERFMON)"
    fi
else
    echo "  bpftrace:   SKIP"
fi

if command -v flamegraph.pl &>/dev/null || command -v inferno-flamegraph &>/dev/null; then
    HAS_FLAMEGRAPH_RENDERER=true
    echo "  flamegraph: available"
else
    echo "  flamegraph: SKIP (install inferno: cargo install inferno)"
fi

if ! command -v jq &>/dev/null; then
    echo "ERROR: jq not found. Install: apt-get install -y jq (or brew install jq)"
    exit 1
fi
echo "  jq:         available"

if ! command -v aiperf &>/dev/null && ! python3 -c "import aiperf" 2>/dev/null; then
    echo "ERROR: aiperf not found. Install: pip install git+https://github.com/ai-dynamo/aiperf.git"
    exit 1
fi
echo "  aiperf:     available"
echo ""

# ─── Tracked PIDs for cleanup ───────────────────────────────────────────────
ALL_PIDS=()      # everything we need to kill on exit
CAPTURE_PIDS=()  # capture processes we wait for
ETCD_PID=""      # etcd PID if we started it
ETCD_DATA_DIR=""
NATS_PID=""      # nats-server PID if we started it

cleanup() {
    echo ""
    echo "--- Cleaning up ---"
    # Stop capture processes with SIGINT first (bpftrace needs INT to flush maps)
    for pid in "${CAPTURE_PIDS[@]}"; do
        kill -INT "$pid" 2>/dev/null || true
    done
    # Give bpftrace a moment to flush aggregation output
    sleep 2
    # SIGTERM any captures still running
    for pid in "${CAPTURE_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    # Then service processes
    for pid in "${ALL_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    # Give processes a moment to exit, then force kill
    sleep 1
    for pid in "${ALL_PIDS[@]}" "${CAPTURE_PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
    # Stop infrastructure daemons if we started them
    if [[ -n "$ETCD_PID" ]]; then
        kill "$ETCD_PID" 2>/dev/null || true
        sleep 1
        kill -9 "$ETCD_PID" 2>/dev/null || true
        [[ -n "$ETCD_DATA_DIR" ]] && rm -rf "$ETCD_DATA_DIR"
    fi
    if [[ -n "$NATS_PID" ]]; then
        kill "$NATS_PID" 2>/dev/null || true
        sleep 1
        kill -9 "$NATS_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    # Wait for the frontend port to be fully released (TIME_WAIT, etc.)
    _port_wait=0
    while ss -tlnp 2>/dev/null | grep -q ":${FRONTEND_PORT} "; do
        sleep 1
        _port_wait=$((_port_wait + 1))
        if [[ $_port_wait -ge 15 ]]; then
            break
        fi
    done
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# ─── Step 0: Ensure etcd + NATS are running ──────────────────────────────────
echo "--- Checking infrastructure (etcd + NATS) ---"

# etcd
if ! curl -sf http://localhost:2379/health >/dev/null 2>&1; then
    if command -v etcd &>/dev/null; then
        echo "  etcd not running — starting it..."
        ETCD_DATA_DIR=$(mktemp -d)
        etcd --data-dir="$ETCD_DATA_DIR" \
             --listen-client-urls=http://localhost:2379 \
             --advertise-client-urls=http://localhost:2379 \
             --listen-peer-urls=http://localhost:2380 \
             --initial-advertise-peer-urls=http://localhost:2380 \
             --initial-cluster=default=http://localhost:2380 \
             > /dev/null 2>&1 &
        ETCD_PID=$!
        for i in $(seq 1 30); do
            if curl -sf http://localhost:2379/health >/dev/null 2>&1; then
                echo "  etcd ready (PID $ETCD_PID)"
                break
            fi
            sleep 1
            if [[ $i -eq 30 ]]; then
                echo "ERROR: etcd failed to start after 30s"; exit 1
            fi
        done
    else
        echo "ERROR: etcd not running and not found in PATH."; exit 1
    fi
else
    echo "  etcd already running"
fi

# NATS
if ! nc -z localhost 4222 2>/dev/null; then
    if command -v nats-server &>/dev/null; then
        echo "  nats-server not running — starting it..."
        nats-server > /dev/null 2>&1 &
        NATS_PID=$!
        for i in $(seq 1 30); do
            if nc -z localhost 4222 2>/dev/null; then
                echo "  nats-server ready (PID $NATS_PID)"
                break
            fi
            sleep 1
            if [[ $i -eq 30 ]]; then
                echo "ERROR: nats-server failed to start after 30s"; exit 1
            fi
        done
    else
        echo "ERROR: nats-server not running and not found in PATH."; exit 1
    fi
else
    echo "  nats-server already running"
fi
echo ""

# ─── Step 1: Start mocker workers ───────────────────────────────────────────
TOTAL_WORKERS=$(( ${#MODEL_NAMES[@]} * NUM_WORKERS ))
echo "--- Starting $TOTAL_WORKERS mocker worker(s) (${#MODEL_NAMES[@]} model(s) x $NUM_WORKERS worker(s)) ---"
BASE_SYSTEM_PORT=8081
WORKER_IDX=0
for MN in "${MODEL_NAMES[@]}"; do
    for i in $(seq 1 "$NUM_WORKERS"); do
        WORKER_IDX=$((WORKER_IDX + 1))
        WORKER_PORT=$((BASE_SYSTEM_PORT + WORKER_IDX - 1))
        MOCKER_ARGS=(
            --model-path "$MODEL"
            --model-name "$MN"
            --speedup-ratio "$SPEEDUP_RATIO"
            --request-plane "$REQUEST_PLANE"
        )
        if [[ "$DATA_PARALLEL_SIZE" -gt 1 ]]; then
            MOCKER_ARGS+=(--data-parallel-size "$DATA_PARALLEL_SIZE")
        fi
        if [[ -n "$PLANNER_PROFILE" ]]; then
            MOCKER_ARGS+=(--planner-profile-data "$PLANNER_PROFILE")
        fi

        MN_SAFE="${MN//\//_}"
        HF_HUB_OFFLINE=1 DYN_SYSTEM_PORT=$WORKER_PORT DYN_EVENT_PLANE="$EVENT_PLANE" python -m dynamo.mocker "${MOCKER_ARGS[@]}" \
            > "$OUTPUT_DIR/logs/mocker_${MN_SAFE}_${i}.log" 2>&1 &
        ALL_PIDS+=($!)
        echo "  Worker $WORKER_IDX ($MN #$i): PID ${ALL_PIDS[-1]}, port $WORKER_PORT"
    done
done
# Update NUM_WORKERS to total for Prometheus scraping later
NUM_WORKERS_TOTAL=$WORKER_IDX

# ─── Step 2: Start frontend (optionally under nsys) ─────────────────────────
echo ""
echo "--- Starting frontend ---"

FRONTEND_ENV=(
    HF_HUB_OFFLINE=1
    DYN_HTTP_PORT="$FRONTEND_PORT"
    DYN_PERF_DIAG=1
    DYN_ENABLE_NVTX=1
    DYN_REQUEST_PLANE="$REQUEST_PLANE"
    DYN_EVENT_PLANE="$EVENT_PLANE"
)

if [[ -n "$TOKENIZER_BACKEND" ]]; then
    # Map human-readable CLI values to DYN_TOKENIZER env var values
    # (Rust model_card.rs reads DYN_TOKENIZER, expects "fastokens" or "default")
    case "$TOKENIZER_BACKEND" in
        fast|fastokens) _DYN_TOK_VAL="fastokens" ;;
        hf|default|"")  _DYN_TOK_VAL="default" ;;
        *)               _DYN_TOK_VAL="$TOKENIZER_BACKEND" ;;
    esac
    FRONTEND_ENV+=(DYN_TOKENIZER="$_DYN_TOK_VAL")
    echo "  Tokenizer backend: $TOKENIZER_BACKEND (DYN_TOKENIZER=$_DYN_TOK_VAL)"
fi

# Enable Rust NVTX annotations when nsys profiling is active.
# The Rust NVTX subsystem (lib/runtime/src/nvtx.rs) requires both the
# compile-time "nvtx" feature AND this runtime env var. We only set it
# when nsys is active to avoid the ~50ns/annotation overhead during
# clean throughput runs.
if [[ "$HAS_NSYS" == true ]]; then
    FRONTEND_ENV+=(DYN_ENABLE_RUST_NVTX=1)
    echo "  Rust NVTX annotations: enabled"
fi

if [[ "$HAS_NSYS" == true ]]; then
    echo "  (under nsys profiling)"
    env "${FRONTEND_ENV[@]}" \
        "$NSYS_CMD" profile \
        --trace=osrt,nvtx \
        --sample=cpu \
        --cpuctxsw=none \
        --output="${OUTPUT_DIR}/nsys/frontend" \
        --force-overwrite=true \
        python -m dynamo.frontend \
        > "$OUTPUT_DIR/logs/frontend.log" 2>&1 &
    NSYS_WRAPPER_PID=$!
    ALL_PIDS+=($NSYS_WRAPPER_PID)
    # Resolve the actual python child PID for perf stat / proc polling.
    # nsys spawns python as a child; we need the real PID for --pid attachments.
    FRONTEND_PID=""
    for _try in $(seq 1 30); do
        sleep 1
        _child=$(pgrep -P "$NSYS_WRAPPER_PID" -f "python.*dynamo.frontend" 2>/dev/null | head -1 || true)
        if [[ -n "$_child" ]]; then
            FRONTEND_PID="$_child"
            break
        fi
    done
    if [[ -z "$FRONTEND_PID" ]]; then
        echo "  WARNING: could not resolve python child PID under nsys; using nsys wrapper PID"
        echo "  (perf stat and /proc polling will attach to nsys, not the frontend process)"
        FRONTEND_PID="$NSYS_WRAPPER_PID"
    fi
    echo "  nsys wrapper PID: $NSYS_WRAPPER_PID"
    echo "  Frontend PID: $FRONTEND_PID"
else
    env "${FRONTEND_ENV[@]}" python -m dynamo.frontend \
        > "$OUTPUT_DIR/logs/frontend.log" 2>&1 &
    FRONTEND_PID=$!
    ALL_PIDS+=($FRONTEND_PID)
    echo "  Frontend PID: $FRONTEND_PID"
fi

# ─── Step 3: Wait for readiness ─────────────────────────────────────────────
echo ""
echo "Waiting for ${#MODEL_NAMES[@]} model(s) to be ready: ${MODEL_NAMES[*]}..."
MAX_WAIT=180
WAITED=0
_all_models_ready=false
while [[ "$_all_models_ready" == false ]]; do
    _all_models_ready=true
    for _mn in "${MODEL_NAMES[@]}"; do
        if ! curl -s --max-time 5 "http://127.0.0.1:$FRONTEND_PORT/v1/models" 2>/dev/null | \
              jq -e --arg model "$_mn" '.data[]? | select(.id == $model)' >/dev/null 2>&1; then
            _all_models_ready=false
            break
        fi
    done
    if [[ "$_all_models_ready" == true ]]; then
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    if [[ $WAITED -ge $MAX_WAIT ]]; then
        echo "ERROR: Models not ready after ${MAX_WAIT}s."
        echo "Last 20 lines of frontend log:"
        tail -20 "$OUTPUT_DIR/logs/frontend.log" 2>/dev/null || true
        exit 1
    fi
    if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo "ERROR: Frontend process died."
        echo "Last 20 lines of frontend log:"
        tail -20 "$OUTPUT_DIR/logs/frontend.log" 2>/dev/null || true
        exit 1
    fi
done
echo "All ${#MODEL_NAMES[@]} model(s) ready (waited ${WAITED}s)"

# Capture initial Prometheus snapshot (baseline for histogram delta analysis)
echo "  Capturing initial Prometheus snapshot..."
{
    curl -s --max-time 5 "http://127.0.0.1:$FRONTEND_PORT/metrics" 2>/dev/null || true
    for wi in $(seq 1 "$NUM_WORKERS_TOTAL"); do
        WPORT=$((BASE_SYSTEM_PORT + wi - 1))
        curl -s --max-time 5 "http://127.0.0.1:$WPORT/metrics" 2>/dev/null || true
    done
} > "$OUTPUT_DIR/prometheus/initial_snapshot.txt"

# ─── Step 4: Start parallel captures ────────────────────────────────────────
echo ""
echo "--- Starting parallel captures (${CAPTURE_DURATION}s) ---"

# 4a. perf stat — with NMI watchdog fallback
if [[ "$HAS_PERF" == true ]]; then
    NMI_WATCHDOG=$(cat /proc/sys/kernel/nmi_watchdog 2>/dev/null || echo "0")
    if [[ "$NMI_WATCHDOG" == "1" ]]; then
        # NMI watchdog holds a HW PMU counter, causing <not counted> for
        # most hardware events. Fall back to software counters + userspace-
        # qualified cycles/instructions which work on the remaining PMUs.
        echo "  [perf stat] pid=$FRONTEND_PID (NMI watchdog active — using software + :u counters)"
        echo "  TIP: disable watchdog for full HW counters: sudo sysctl kernel.nmi_watchdog=0"
        perf stat --pid "$FRONTEND_PID" \
            -e task-clock,context-switches,cpu-migrations,page-faults,cycles:u,instructions:u \
            -o "$OUTPUT_DIR/perf/perf_stat.txt" \
            -- sleep "$CAPTURE_DURATION" &
        CAPTURE_PIDS+=($!)
    else
        echo "  [perf stat] pid=$FRONTEND_PID"
        perf stat --pid "$FRONTEND_PID" \
            -o "$OUTPUT_DIR/perf/perf_stat.txt" \
            -- sleep "$CAPTURE_DURATION" &
        CAPTURE_PIDS+=($!)
    fi
fi

# 4b. BPF scripts (if available) — delegate to bpf/run.sh
# run.sh --batch waits for its children internally and forwards TERM→INT
# so bpftrace flushes aggregation maps before exiting.
if [[ "$HAS_BPF" == true ]]; then
    "${SCRIPT_DIR}/bpf/run.sh" --batch \
        --pid "$FRONTEND_PID" \
        --output-dir "$OUTPUT_DIR/bpf" \
        --duration "$CAPTURE_DURATION" &
    CAPTURE_PIDS+=($!)
fi

# 4e. Flamegraph captures — delegate to flamegraph/ scripts
if [[ "$SKIP_FLAMEGRAPH" == false ]]; then
    if command -v perf &>/dev/null || command -v samply &>/dev/null; then
        echo "  [flamegraph] CPU, pid=$FRONTEND_PID"
        "${SCRIPT_DIR}/flamegraph/cpu_flamegraph.sh" \
            --pid "$FRONTEND_PID" \
            --duration "$CAPTURE_DURATION" \
            --output-dir "$OUTPUT_DIR/perf" \
            --output cpu_flamegraph &
        CAPTURE_PIDS+=($!)
    else
        echo "  [flamegraph] SKIP CPU — install perf or samply:"
        echo "    apt install linux-tools-\$(uname -r)  OR  cargo install samply"
    fi
    if [[ "$HAS_BPF" == true ]]; then
        echo "  [flamegraph] Off-CPU, pid=$FRONTEND_PID"
        "${SCRIPT_DIR}/flamegraph/offcpu_flamegraph.sh" \
            --pid "$FRONTEND_PID" \
            --duration "$CAPTURE_DURATION" \
            --output-dir "$OUTPUT_DIR/perf" \
            --output offcpu_flamegraph &
        CAPTURE_PIDS+=($!)
    else
        echo "  [flamegraph] SKIP Off-CPU — needs bpftrace with root or CAP_BPF"
    fi
fi

# 4c. System capture (/proc polling, thread/fd counts, socket stats)
echo "  [system] /proc stats, thread/fd count, ss"
(
    INTERVAL=1
    for _i in $(seq 1 "$CAPTURE_DURATION"); do
        TS=$(date -Iseconds)

        # /proc status
        echo "--- $TS ---" >> "$OUTPUT_DIR/system/proc_status.txt"
        cat "/proc/$FRONTEND_PID/status" >> "$OUTPUT_DIR/system/proc_status.txt" 2>/dev/null || true

        # /proc stat — raw scheduler/CPU time info
        echo "--- $TS ---" >> "$OUTPUT_DIR/system/proc_stat.txt"
        cat "/proc/$FRONTEND_PID/stat" >> "$OUTPUT_DIR/system/proc_stat.txt" 2>/dev/null || true

        # /proc statm — page-level memory info
        echo "--- $TS ---" >> "$OUTPUT_DIR/system/proc_statm.txt"
        cat "/proc/$FRONTEND_PID/statm" >> "$OUTPUT_DIR/system/proc_statm.txt" 2>/dev/null || true

        # Thread count
        THREADS=$(ls -1 "/proc/$FRONTEND_PID/task/" 2>/dev/null | wc -l)
        echo "$TS threads=$THREADS" >> "$OUTPUT_DIR/system/thread_count.txt"

        # FD count
        FDS=$(ls -1 "/proc/$FRONTEND_PID/fd/" 2>/dev/null | wc -l)
        echo "$TS fds=$FDS" >> "$OUTPUT_DIR/system/fd_count.txt"

        # Socket stats
        echo "--- $TS ---" >> "$OUTPUT_DIR/system/ss_stats.txt"
        ss -tin >> "$OUTPUT_DIR/system/ss_stats.txt" 2>/dev/null || true

        sleep "$INTERVAL"
    done
) &
CAPTURE_PIDS+=($!)

# 4c-2. tcpdump packet capture (optional)
if [[ "$ENABLE_TCPDUMP" == true ]]; then
    if command -v tcpdump &>/dev/null; then
        echo "  [tcpdump] port=$TCPDUMP_PORT"
        timeout "$CAPTURE_DURATION" tcpdump -i any \
            -w "$OUTPUT_DIR/system/capture.pcap" \
            "port $TCPDUMP_PORT" \
            -s 96 -c 100000 &>/dev/null &
        CAPTURE_PIDS+=($!)
    else
        echo "  [tcpdump] SKIP — tcpdump not found (apt install tcpdump)"
    fi
fi

# 4d. Periodic Prometheus /metrics scraping (frontend + all mocker workers)
echo "  [prometheus] scraping every 1s"
(
    for _i in $(seq 1 "$CAPTURE_DURATION"); do
        METRICS=$(curl -s --max-time 3 "http://127.0.0.1:$FRONTEND_PORT/metrics" 2>/dev/null || echo "")
        # Append mocker worker metrics (ports BASE_SYSTEM_PORT .. BASE_SYSTEM_PORT+NUM_WORKERS-1)
        for wi in $(seq 1 "$NUM_WORKERS_TOTAL"); do
            WPORT=$((BASE_SYSTEM_PORT + wi - 1))
            WMETRICS=$(curl -s --max-time 3 "http://127.0.0.1:$WPORT/metrics" 2>/dev/null || echo "")
            if [[ -n "$WMETRICS" ]]; then
                METRICS="${METRICS}"$'\n'"${WMETRICS}"
            fi
        done
        if [[ -n "$METRICS" ]]; then
            # JSONL: one line per scrape with timestamp
            TS=$(date -Iseconds)
            printf '{"ts":"%s","metrics":%s}\n' "$TS" "$(echo "$METRICS" | python3 -c '
import sys, json
lines = sys.stdin.read().strip().split("\n")
out = {}
for line in lines:
    if line.startswith("#") or not line.strip():
        continue
    parts = line.split()
    if len(parts) >= 2:
        out[parts[0]] = parts[1]
print(json.dumps(out))
' 2>/dev/null || echo '"{}"')" >> "$OUTPUT_DIR/prometheus/timeseries.jsonl"
        fi
        sleep 1
    done
) &
CAPTURE_PIDS+=($!)

# ─── Step 5: Run aiperf load ────────────────────────────────────────────────
echo ""
echo "--- Running aiperf load ---"
echo "  concurrency=$CONCURRENCY  requests=${NUM_REQUESTS:-auto}  isl=$ISL  osl=$OSL"
[[ -n "$BENCHMARK_DURATION" ]] && echo "  benchmark-duration=${BENCHMARK_DURATION}s"
[[ -n "$REQUEST_RATE" ]] && echo "  request-rate=${REQUEST_RATE} req/s"

# Build load-control args: prefer --benchmark-duration (time-based) over
# --request-count (count-based) so each run gets a consistent measurement
# window regardless of throughput.  This mirrors k8s_run_aiperf() in sweep.sh.
_LOAD_ARGS=()
_EFFECTIVE_REQUESTS="null"
if [[ -n "$BENCHMARK_DURATION" ]]; then
    _LOAD_ARGS+=(--benchmark-duration "$BENCHMARK_DURATION")
fi
if [[ -n "$REQUEST_RATE" ]]; then
    _LOAD_ARGS+=(--request-rate "$REQUEST_RATE")
fi
if [[ -n "$NUM_REQUESTS" ]]; then
    _LOAD_ARGS+=(--request-count "$NUM_REQUESTS")
    _EFFECTIVE_REQUESTS="$NUM_REQUESTS"
fi
# If neither was specified, default to concurrency * 20 (min 640)
if [[ ${#_LOAD_ARGS[@]} -eq 0 ]]; then
    _AUTO=$(( CONCURRENCY * 20 ))
    [[ "$_AUTO" -lt 640 ]] && _AUTO=640
    _LOAD_ARGS+=(--request-count "$_AUTO")
    _EFFECTIVE_REQUESTS="$_AUTO"
fi

_WARMUP_ARGS=()
if [[ -n "$WARMUP_DURATION" ]]; then
    _WARMUP_ARGS+=(--warmup-duration "$WARMUP_DURATION")
elif [[ -n "$WARMUP_COUNT" ]]; then
    _WARMUP_ARGS+=(--warmup-request-count "$WARMUP_COUNT")
else
    _WARMUP_ARGS+=(--warmup-request-count "$CONCURRENCY")
fi

# Build the list of models to target
_AIPERF_MODELS=()
if [[ "$AIPERF_TARGETS" == "all" && ${#MODEL_NAMES[@]} -gt 1 ]]; then
    _AIPERF_MODELS=("${MODEL_NAMES[@]}")
else
    _AIPERF_MODELS=("${MODEL_NAMES[0]}")
fi
echo "  aiperf targets: ${_AIPERF_MODELS[*]} (mode=$AIPERF_TARGETS)"

for _AIPERF_MODEL in "${_AIPERF_MODELS[@]}"; do
    if [[ ${#_AIPERF_MODELS[@]} -gt 1 ]]; then
        AIPERF_ARTIFACT_DIR="$OUTPUT_DIR/aiperf/${_AIPERF_MODEL}"
        echo ""
        echo "  --- aiperf: model=${_AIPERF_MODEL} ---"
    else
        AIPERF_ARTIFACT_DIR="$OUTPUT_DIR/aiperf"
    fi
    mkdir -p "$AIPERF_ARTIFACT_DIR"

    # When the served model name differs from the HF model path (multi-model),
    # tell aiperf where to find the tokenizer.
    _AIPERF_TOK_ARGS=()
    if [[ "$_AIPERF_MODEL" != "$MODEL" ]]; then
        _AIPERF_TOK_ARGS=(--tokenizer "$MODEL")
    fi

    HF_HUB_OFFLINE=1 aiperf profile --artifact-dir "$AIPERF_ARTIFACT_DIR" \
        --model "$_AIPERF_MODEL" \
        "${_AIPERF_TOK_ARGS[@]}" \
        --endpoint-type chat \
        --endpoint /v1/chat/completions \
        --streaming \
        --url "http://127.0.0.1:$FRONTEND_PORT" \
        --synthetic-input-tokens-mean "$ISL" \
        --synthetic-input-tokens-stddev 0 \
        --output-tokens-mean "$OSL" \
        --output-tokens-stddev 0 \
        --extra-inputs max_tokens:"$OSL" \
        --extra-inputs min_tokens:"$OSL" \
        --extra-inputs ignore_eos:true \
        --extra-inputs repetition_penalty:1.0 \
        --extra-inputs temperature:0.0 \
        --concurrency "$CONCURRENCY" \
        "${_LOAD_ARGS[@]}" \
        "${_WARMUP_ARGS[@]}" \
        --num-dataset-entries 12800 \
        --random-seed 100 \
        --workers-max "$CONCURRENCY" \
        --record-processors 32 \
        --ui simple || echo "WARNING: aiperf failed for model ${_AIPERF_MODEL}"
done

# Check for server_metrics_export.json in the primary aiperf dir
_PRIMARY_AIPERF_DIR="$OUTPUT_DIR/aiperf"
[[ ${#_AIPERF_MODELS[@]} -gt 1 ]] && _PRIMARY_AIPERF_DIR="$OUTPUT_DIR/aiperf/${_AIPERF_MODELS[0]}"
if [[ -f "$_PRIMARY_AIPERF_DIR/server_metrics_export.json" ]]; then
    echo "  Found server_metrics_export.json"
fi

# ─── Step 6: Wait for captures to finish ─────────────────────────────────────
echo ""
echo "--- Waiting for captures to finish ---"

# Give captures CAPTURE_DURATION + 15s grace period, then force-kill stragglers.
# Profiler processes (perf record, flamegraph, bpftrace) can hang if the target
# PID becomes idle or exits, or if hardware counters can't be released cleanly.
_CAPTURE_DEADLINE=$(( CAPTURE_DURATION + 15 ))
_capture_waited=0
_all_done=false
while [[ "$_all_done" == false && $_capture_waited -lt $_CAPTURE_DEADLINE ]]; do
    _all_done=true
    for pid in "${CAPTURE_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            _all_done=false
            break
        fi
    done
    if [[ "$_all_done" == false ]]; then
        sleep 1
        _capture_waited=$((_capture_waited + 1))
    fi
done

if [[ "$_all_done" == false ]]; then
    echo "  WARNING: captures still running after ${_CAPTURE_DEADLINE}s — sending SIGTERM"
    for pid in "${CAPTURE_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    sleep 2
    # Force kill any remaining
    for pid in "${CAPTURE_PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
fi
# Reap all child statuses
for pid in "${CAPTURE_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

# Capture final Prometheus snapshot AFTER all aiperf requests have completed.
# This must happen after aiperf (Step 5) and after captures (Step 6) to ensure
# all request metrics are reflected in the snapshot.
echo "  Capturing final Prometheus snapshot..."
{
    curl -s --max-time 5 "http://127.0.0.1:$FRONTEND_PORT/metrics" 2>/dev/null || true
    for wi in $(seq 1 "$NUM_WORKERS_TOTAL"); do
        WPORT=$((BASE_SYSTEM_PORT + wi - 1))
        curl -s --max-time 5 "http://127.0.0.1:$WPORT/metrics" 2>/dev/null || true
    done
} > "$OUTPUT_DIR/prometheus/final_snapshot.txt"
# Save individual worker snapshots for debugging
for wi in $(seq 1 "$NUM_WORKERS_TOTAL"); do
    WPORT=$((BASE_SYSTEM_PORT + wi - 1))
    curl -s --max-time 5 "http://127.0.0.1:$WPORT/metrics" \
        > "$OUTPUT_DIR/prometheus/mocker_${wi}_snapshot.txt" 2>/dev/null || true
done

# ─── Step 7: Stop frontend and post-process nsys ─────────────────────────────
echo ""
echo "--- Post-processing ---"

# 7a. Stop frontend so nsys can finalize the .nsys-rep file.
# When running under nsys, the .nsys-rep is only written after the wrapped
# process exits. We must stop the frontend before attempting the export.
if [[ "$HAS_NSYS" == true ]]; then
    echo "  Stopping frontend (PID $FRONTEND_PID) so nsys can finalize..."
    kill -INT "$FRONTEND_PID" 2>/dev/null || true
    # Wait for the nsys-wrapped frontend to exit and write the .nsys-rep
    NSYS_WAIT=0
    NSYS_WAIT_MAX=30
    while kill -0 "$FRONTEND_PID" 2>/dev/null; do
        sleep 1
        NSYS_WAIT=$((NSYS_WAIT + 1))
        if [[ $NSYS_WAIT -ge $NSYS_WAIT_MAX ]]; then
            echo "  WARNING: frontend did not exit after ${NSYS_WAIT_MAX}s, sending SIGTERM"
            kill "$FRONTEND_PID" 2>/dev/null || true
            sleep 2
            break
        fi
    done
    # Remove frontend from ALL_PIDS so cleanup() doesn't try to kill it again
    NEW_ALL_PIDS=()
    for pid in "${ALL_PIDS[@]}"; do
        [[ "$pid" != "$FRONTEND_PID" ]] && NEW_ALL_PIDS+=("$pid")
    done
    ALL_PIDS=("${NEW_ALL_PIDS[@]}")

    if [[ -f "$OUTPUT_DIR/nsys/frontend.nsys-rep" ]]; then
        echo "  Exporting nsys to SQLite..."
        "$NSYS_CMD" export --type sqlite \
            --output "$OUTPUT_DIR/nsys/frontend.sqlite" \
            "$OUTPUT_DIR/nsys/frontend.nsys-rep" 2>/dev/null || \
            echo "  WARNING: nsys sqlite export failed"
        if [[ -f "$OUTPUT_DIR/nsys/frontend.sqlite" ]]; then
            echo "  nsys SQLite: $OUTPUT_DIR/nsys/frontend.sqlite"
        fi
    else
        echo "  WARNING: $OUTPUT_DIR/nsys/frontend.nsys-rep not found after frontend exit"
    fi
else
    # No nsys — stop frontend normally (cleanup will handle it, but stop early
    # to avoid holding ports during config save)
    kill -INT "$FRONTEND_PID" 2>/dev/null || true
fi

# ─── Step 8: Save config ────────────────────────────────────────────────────
cat > "$OUTPUT_DIR/config.json" <<EOF
{
  "timestamp": "$TIMESTAMP",
  "model": "$MODEL",
  "model_name": "$MODEL_NAME",
  "model_names": $(printf '%s\n' "${MODEL_NAMES[@]}" | jq -R . | jq -s .),
  "num_models": $NUM_MODELS,
  "num_workers_per_model": $NUM_WORKERS,
  "num_workers_total": $NUM_WORKERS_TOTAL,
  "speedup_ratio": "$SPEEDUP_RATIO",
  "data_parallel_size": $DATA_PARALLEL_SIZE,
  "request_plane": "$REQUEST_PLANE",
  "event_plane": "$EVENT_PLANE",
  "frontend_port": $FRONTEND_PORT,
  "tokenizer_backend": "${TOKENIZER_BACKEND:-hf}",
  "concurrency": $CONCURRENCY,
  "num_requests": ${_EFFECTIVE_REQUESTS:-null},
  "benchmark_duration": ${BENCHMARK_DURATION:-null},
  "request_rate": ${REQUEST_RATE:-null},
  "isl": $ISL,
  "osl": $OSL,
  "capture_duration": $CAPTURE_DURATION,
  "frontend_pid": $FRONTEND_PID,
  "has_nsys": $HAS_NSYS,
  "has_perf": $HAS_PERF,
  "has_bpf": $HAS_BPF,
  "skip_bpf": $SKIP_BPF,
  "skip_nsys": $SKIP_NSYS,
  "skip_flamegraph": $SKIP_FLAMEGRAPH,
  "tcpdump": $ENABLE_TCPDUMP,
  "tcpdump_port": $TCPDUMP_PORT,
  "nsys_path": "${NSYS_PATH:-auto}"
}
EOF

# ─── Done ────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Capture Complete                                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Contents:"
find "$OUTPUT_DIR" -type f | sort | while read -r f; do
    SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
    REL="${f#$OUTPUT_DIR/}"
    printf "  %-50s %s bytes\n" "$REL" "$SIZE"
done
echo ""
echo "Run analysis:"
echo "  python3 ${SCRIPT_DIR}/analysis/create_report.py analyze $OUTPUT_DIR"
