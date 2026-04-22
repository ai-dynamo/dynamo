#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launches dynamo.frontend + dynamo.vllm for the benchmark sweep.
#
# Optional nsys capture, controlled by env vars set by the sweep orchestrator:
#   DYN_BACKEND_CAPTURE_OUT   — path; wraps dynamo.vllm in nsys with
#                                --capture-range=cudaProfilerApi. The orchestrator
#                                POSTs /engine/start_profile on DYN_SYSTEM_PORT
#                                around aiperf; AsyncLLM broadcasts cudaProfilerStart
#                                to all TP workers. nsys 2026.1+ traces the full
#                                process tree by default, so one .nsys-rep covers
#                                every worker.
#   DYN_FRONTEND_CAPTURE_OUT  — path; wraps dynamo.frontend in nsys full-lifetime
#                                (no trigger; Rust is low-noise and the existing
#                                per-request NVTX ranges suffice).
#   DYN_SYSTEM_PORT           — exposed on the backend so /engine/* routes are
#                                reachable (required when backend capture is on;
#                                the backend system-status server is off by default).
#
# Prereqs: etcd + nats-server running locally (standard Dynamo dev setup).

set -euo pipefail

MODEL=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required" >&2
    exit 1
fi

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
SYSTEM_PORT="${DYN_SYSTEM_PORT:-}"
BACKEND_CAPTURE_OUT="${DYN_BACKEND_CAPTURE_OUT:-}"
FRONTEND_CAPTURE_OUT="${DYN_FRONTEND_CAPTURE_OUT:-}"

# ─── Preflight nsys if any capture is requested ─────────────────────────────
NSYS_BIN=""
if [[ -n "$BACKEND_CAPTURE_OUT$FRONTEND_CAPTURE_OUT" ]]; then
    if ! NSYS_BIN="$(command -v nsys)"; then
        echo "ERROR: capture requested but nsys not found in PATH" >&2
        exit 1
    fi
    NSYS_VERSION="$("$NSYS_BIN" --version | awk '{print $NF}')"
    echo "nsys: $NSYS_BIN ($NSYS_VERSION)"
    # Single-file multi-process capture relies on nsys 2026.1+'s default
    # process-tree scope (--cuda-trace-scope=process-tree, --sample=process-tree).
    # Earlier versions don't trace TP workers, so the resulting trace is near-empty.
    case "$NSYS_VERSION" in
        2026.*|2027.*|2028.*|2029.*) ;;
        *)
            echo "ERROR: nsys $NSYS_VERSION is too old. Single-file TP capture needs 2026.1+ (process-tree default)." >&2
            exit 1 ;;
    esac
fi

if [[ -n "$BACKEND_CAPTURE_OUT" && -z "$SYSTEM_PORT" ]]; then
    echo "ERROR: DYN_BACKEND_CAPTURE_OUT is set but DYN_SYSTEM_PORT is not." >&2
    echo "       The orchestrator needs /engine/start_profile on that port." >&2
    exit 1
fi

FRONTEND_PID=""
BACKEND_PID=""

cleanup() {
    # Forward SIGINT (not SIGTERM) so nsys drains queued events and writes
    # .nsys-rep instead of leaving a .qdstrm. See project/code.md
    # "Profiler wrappers must forward SIGINT/SIGTERM".
    for pid in "$BACKEND_PID" "$FRONTEND_PID"; do
        [[ -n "$pid" ]] && kill -INT "$pid" 2>/dev/null || true
    done
    # Give nsys up to 120s to finalize.
    for _ in $(seq 1 120); do
        local alive=0
        for pid in "$BACKEND_PID" "$FRONTEND_PID"; do
            [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && alive=1
        done
        [[ $alive -eq 0 ]] && break
        sleep 1
    done
    # Escalate to TERM → KILL only if something is still alive.
    for pid in "$BACKEND_PID" "$FRONTEND_PID"; do
        [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "$BACKEND_PID" "$FRONTEND_PID"; do
        [[ -n "$pid" ]] && kill -KILL "$pid" 2>/dev/null || true
    done
}
trap cleanup INT TERM

echo "==========================================="
echo " dynamo_serve.sh"
echo "==========================================="
echo "  Model:              $MODEL"
echo "  Frontend:           http://localhost:$HTTP_PORT"
echo "  System port:        ${SYSTEM_PORT:-(unset)}"
echo "  Backend capture:    ${BACKEND_CAPTURE_OUT:-(off)}"
echo "  Frontend capture:   ${FRONTEND_CAPTURE_OUT:-(off)}"
echo "  Extra vllm args:    ${EXTRA_ARGS[*]:-(none)}"
echo "==========================================="

# ─── Start frontend ─────────────────────────────────────────────────────────
FRONTEND_ENV=(DYN_HTTP_PORT="$HTTP_PORT")

if [[ -n "$FRONTEND_CAPTURE_OUT" ]]; then
    mkdir -p "$(dirname "$FRONTEND_CAPTURE_OUT")"
    FRONTEND_ENV+=(DYN_ENABLE_RUST_NVTX=1)
    # --trace must include "cuda" even though the frontend has no CUDA: nsys
    # drops NVTX events from processes that never issue a CUDA API call, so
    # without "cuda" in --trace the frontend trace is empty of NVTX ranges.
    # See research/profiling.md.
    env "${FRONTEND_ENV[@]}" "$NSYS_BIN" profile \
        --trace=nvtx,cuda,osrt \
        --output="$FRONTEND_CAPTURE_OUT" \
        --force-overwrite=true \
        python -m dynamo.frontend &
else
    env "${FRONTEND_ENV[@]}" python -m dynamo.frontend &
fi
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

# Wait for frontend HTTP port to respond. /v1/models returns 200 with an empty
# list while the backend hasn't registered yet — that's fine, we just need the
# port to be open before the backend comes up.
for i in $(seq 1 60); do
    if curl -sf "http://localhost:$HTTP_PORT/v1/models" > /dev/null 2>&1; then
        echo "Frontend up."
        break
    fi
    if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo "ERROR: frontend exited during startup." >&2
        exit 1
    fi
    sleep 1
done

# ─── Start backend ──────────────────────────────────────────────────────────
BACKEND_ENV=()
if [[ -n "$SYSTEM_PORT" ]]; then
    BACKEND_ENV+=(DYN_SYSTEM_PORT="$SYSTEM_PORT")
fi
# Pin multiprocessing start method so vLLM workers spawn cleanly under nsys
# (fork + CUDA is unsafe anyway; spawn is the CUDA-safe default in V1).
BACKEND_ENV+=(VLLM_WORKER_MULTIPROC_METHOD=spawn)

BACKEND_CMD=(python -m dynamo.vllm --model "$MODEL" "${EXTRA_ARGS[@]}")

if [[ -n "$BACKEND_CAPTURE_OUT" ]]; then
    mkdir -p "$(dirname "$BACKEND_CAPTURE_OUT")"
    # capture-range=cudaProfilerApi: start/stop gated by engine routes.
    # capture-range-end=stop-shutdown --kill=sigterm: on stop_profile, nsys
    # finalizes the .nsys-rep then SIGTERMs the tree. The sweep orchestrator
    # restarts the server between sweep points (enforced by config validator).
    env "${BACKEND_ENV[@]}" "$NSYS_BIN" profile \
        --trace=cuda,nvtx,cublas,cudnn \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop-shutdown \
        --kill=sigterm \
        --output="$BACKEND_CAPTURE_OUT" \
        --force-overwrite=true \
        "${BACKEND_CMD[@]}" &
else
    env "${BACKEND_ENV[@]}" "${BACKEND_CMD[@]}" &
fi
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Block until either dies; the sweep orchestrator drives lifecycle via SIGTERM.
wait
