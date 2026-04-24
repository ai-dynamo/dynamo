#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dynamo serve workflow: frontend + backend, each wrapped in its own
# nsys instance. Prerequisites: etcd + nats-server started inline below
# if not already running.
#
# Usage (by orchestrator): bash dynamo_serve.sh --model <model> [extra vllm args...]

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

# NVTX gates:
# - DYN_NVTX=1 enables Python `mm:*` markers in the worker.
# - DYN_ENABLE_RUST_NVTX=1 enables the Rust nvtx markers (preprocess.*,
#   codec.*, detokenize). Requires libnvToolsExt.so.1 at runtime —
#   supplied by `apt install libnvtoolsext1` (not shipped by the
#   container's cuda-nvtx-12-9 package, which is NVTX3 headers only).
# - VLLM_MM_CACHE_PROBE=1 activates v3 [mmproc]/[mmcache] probes in
#   the vllm-overlay site-packages.
export DYN_NVTX=1
export DYN_ENABLE_RUST_NVTX=1
export VLLM_NVTX_SCOPES_FOR_PROFILING=1
export VLLM_MM_CACHE_PROBE=1

# nsys wrapping: long duration + --kill=sigterm. On orchestrator shutdown
# (SIGINT from server.py:stop()), cleanup() forwards SIGINT to BOTH nsys
# instances; nsys stops tracing, kills its subtree, finalizes the .nsys-rep.
# Both finalize in parallel → total wall time = max(fe, be), fits in the
# orchestrator's 180 s SIGKILL budget. Mirrors vllm_serve.sh's proven pattern.
NSYS_BIN="${DYN_NSYS_BIN:-/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys}"
NSYS_DIR="${DYN_NSYS_DIR:-/dynamo-tmp/logs/sweep_nsys_dynamo}"
NSYS_DELAY_S="${DYN_NSYS_DELAY_S:-60}"      # skip only nsys-agent startup noise
NSYS_DURATION_S="${DYN_NSYS_DURATION_S:-1500}"  # covers 2B smoke + 397B runs

# Redirect nsys's intermediate staging (.qdstrm) from /tmp/nsys-<uid>/ to
# scratch. Container /tmp is a Pyxis overlay and evaporates when the SLURM
# job ends — so an orphaned .qdstrm (from a finalize that got SIGKILL'd)
# is unrecoverable if it stays there. With TMPDIR on /dynamo-tmp the .qdstrm
# survives container teardown and can be recovered later via a newer nsys
# that ships `qdstrm-importer`.
export TMPDIR="${DYN_NSYS_TMPDIR:-/dynamo-tmp/nsys-staging}"
mkdir -p "$TMPDIR"

mkdir -p "$NSYS_DIR"
TS=$(date +%Y%m%d_%H%M%S)
FRONTEND_OUT="$NSYS_DIR/frontend_$TS.nsys-rep"
BACKEND_OUT="$NSYS_DIR/backend_$TS.nsys-rep"

# DYN_DISABLE_NSYS=1 — run frontend + backend without nsys wrapping (perf sweeps).
# Skip the install check entirely; FE_PREFIX / BE_PREFIX are left empty below so
# the python -m invocations run directly.
DISABLE_NSYS="${DYN_DISABLE_NSYS:-0}"
if [[ "$DISABLE_NSYS" == "1" ]]; then
    echo "[nsys] DYN_DISABLE_NSYS=1 — running frontend+backend without nsys wrap" >&2
fi

# Auto-install nsys from the mounted .deb if missing. Fail fast if the .deb
# isn't available or the install fails — mirrors vllm_serve.sh:78-101. A
# silent "run without nsys" fallback would leave the orchestrator timing out
# on readiness with no trace artifact, which is the footgun we're avoiding.
if [[ "$DISABLE_NSYS" != "1" && ! -x "$NSYS_BIN" ]]; then
    NSYS_DEB=$(ls /nsys/NsightSystems-linux-cli-public-*.deb 2>/dev/null | head -1)
    if [[ -n "$NSYS_DEB" ]]; then
        echo "[nsys] $NSYS_BIN missing; installing from $NSYS_DEB" >&2
        DEBIAN_FRONTEND=noninteractive apt-get update >&2 || true
        DEBIAN_FRONTEND=noninteractive apt-get install -y "$NSYS_DEB" >&2 || {
            echo "[nsys] apt install $NSYS_DEB failed; falling back to dpkg -i + apt --fix-broken" >&2
            dpkg -i "$NSYS_DEB" >&2 || true
            DEBIAN_FRONTEND=noninteractive apt-get -y --fix-broken install >&2 || {
                echo "[nsys] FATAL: all install paths failed; refusing to run without profiling" >&2
                exit 1
            }
        }
    else
        echo "[nsys] FATAL: $NSYS_BIN not executable and no .deb at /nsys/ — refusing to run without profiling" >&2
        exit 1
    fi
fi

if [[ "$DISABLE_NSYS" == "1" ]]; then
    FE_PREFIX=()
    BE_PREFIX=()
else
    NSYS_COMMON=(
        "$NSYS_BIN" profile
        --trace=nvtx,cuda
        --sample=none
        --cpuctxsw=none
        --delay="$NSYS_DELAY_S"
        --duration="$NSYS_DURATION_S"
        # --kill=sigterm: on SIGINT, nsys STOPS tracing, sends SIGTERM to its
        # children (incl. re-parented TP workers on the backend side), waits
        # for them to exit, then finalizes.
        --kill=sigterm
        --force-overwrite=true
    )
    echo "[nsys] frontend -> $FRONTEND_OUT" >&2
    echo "[nsys] backend  -> $BACKEND_OUT" >&2
    FE_PREFIX=("${NSYS_COMMON[@]}" -o "$FRONTEND_OUT")
    BE_PREFIX=("${NSYS_COMMON[@]}" -o "$BACKEND_OUT")
fi

# Infra (etcd + nats) is expected to be already running on the allocation
# (cloud-session starts them in a separate tmux window on container
# attach). If they're not, a fallback start runs here — but without
# deleting /tmp/etcd or /tmp/nats so a live infra isn't wiped out.
if ! curl -sf http://localhost:2379/health > /dev/null 2>&1; then
    echo "[dynamo_serve] etcd not responding — starting fallback" >&2
    etcd --listen-client-urls http://0.0.0.0:2379 \
         --advertise-client-urls http://0.0.0.0:2379 \
         --data-dir /tmp/etcd > /dev/null 2>&1 &
    nats-server -js -m 8222 -p 4222 -sd /tmp/nats > /dev/null 2>&1 &
    sleep 2
fi

FE_NSYS_PID=0
BE_NSYS_PID=0

# Install the trap BEFORE launching children so a fast orchestrator SIGINT
# during the startup window is still forwarded to nsys. cleanup() is a safe
# no-op while PIDs are still 0.
cleanup() {
    echo "[dynamo-serve] signal received; SIGINT -> nsys (FE=$FE_NSYS_PID BE=$BE_NSYS_PID) for graceful finalize" >&2
    for pid in "$FE_NSYS_PID" "$BE_NSYS_PID"; do
        [[ -n "$pid" && "$pid" -gt 0 ]] && kill -INT "$pid" 2>/dev/null || true
    done
    # Shared 150 s deadline — both nsys instances finalize in parallel, so
    # wall time = max(fe, be), under the orchestrator's 180 s SIGKILL budget.
    for _ in $(seq 1 150); do
        any_alive=0
        for pid in "$FE_NSYS_PID" "$BE_NSYS_PID"; do
            [[ -n "$pid" && "$pid" -gt 0 ]] && kill -0 "$pid" 2>/dev/null && any_alive=1
        done
        [[ "$any_alive" -eq 0 ]] && break
        sleep 1
    done
    exit 0
}
trap cleanup INT TERM

"${FE_PREFIX[@]}" python -m dynamo.frontend &
FE_NSYS_PID=$!

"${BE_PREFIX[@]}" python -m dynamo.vllm \
    --model "$MODEL" \
    "${EXTRA_ARGS[@]}" &
BE_NSYS_PID=$!

# Wait for both nsys instances.
# - If orchestrator signals us first, cleanup() forwards and exits.
# - If --duration=$NSYS_DURATION_S elapses, both nsys auto-stop. We drop
#   into the idle loop so the orchestrator's eventual SIGINT still has a
#   well-defined target (this bash).
# - If either nsys exits early with an error (e.g., engine init failure),
#   we surface that to the orchestrator instead of idling forever — lets
#   ServerManager.start() observe startup failures rather than timing out
#   on readiness.
wait "$FE_NSYS_PID"; FE_EXIT=$?
wait "$BE_NSYS_PID"; BE_EXIT=$?
if [[ "$FE_EXIT" -ne 0 || "$BE_EXIT" -ne 0 ]]; then
    WALL=$(date +%s)
    echo "[dynamo-serve] early exit: FE_EXIT=$FE_EXIT BE_EXIT=$BE_EXIT wall=$WALL" >&2
    # If we exited this early, nsys likely didn't run its full --duration
    # window. Report non-zero so the orchestrator doesn't wait until its
    # readiness timeout for a diagnosis.
    exit 1
fi
while sleep 60; do :; done
