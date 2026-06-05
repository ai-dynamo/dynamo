#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Scenario 1: Remote-search uniform TP=2.
#
#   Hub          --features indexer,p2p
#   Instance A   port 8000   GPUs 0,1   NUMA 0   TP=2
#   Instance B   port 8001   GPUs 2,3   NUMA 1   TP=2
#
# Smoke: R -> A (warm + golden), then R -> B (cold G2). B must remote-search
# the hub, pull A's blocks via P2P, onboard, decode, and produce greedy
# output matching A's golden.
#
# Usage: bash scenario-remote-search-uniform-tp2.sh
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/env.sh"
. "$SCRIPT_DIR/numa-lib.sh"

LABEL=${KVBM_EXPERIMENT_LABEL:-remote-search-tp2-uniform}
ROOT=$(bash "$KVBM_REPO/.claude/skills/disagg-bringup/new-experiment.sh" "$LABEL")
echo "EXP=$ROOT"

strip_ansi() { sed 's/\x1b\[[0-9;]*m//g'; }
cleanup() {
  # Teardown on any exit (success or failure) unless explicitly kept alive.
  [ "${KVBM_KEEP_ALIVE:-0}" = "1" ] && { echo "[scenario-1] KVBM_KEEP_ALIVE=1 — leaving hub + vLLMs running"; return; }
  echo "[scenario-1] cleanup: spinning down hub + vLLMs"
  pkill -f "vllm.entrypoints.openai" 2>/dev/null || true
  sleep 2
  pkill -9 -f kvbm_hub 2>/dev/null || true
  rm -f /tmp/velo-kvbm-*.sock 2>/dev/null || true
}
trap cleanup EXIT
fail() {
  echo "FATAL: $1" >&2
  [ -n "${2:-}" ] && [ -f "$2" ] && { echo "--- tail $2 ---" >&2; tail -n 30 "$2" | strip_ansi >&2; }
  exit 1
}

echo "[scenario-1] venv import check"
"$KVBM_VENV/bin/python3" -c "import vllm,kvbm; print('vllm', vllm.__version__, '; kvbm OK')" \
  || fail "venv $KVBM_VENV is missing vllm or kvbm"

echo "[scenario-1] verify GPU NUMA topology"
verify_numa_topology || fail "GPU NUMA topology mismatch -- bailing"

echo "[scenario-1] build kvbm_hub + kvbmctl"
bash "$SCRIPT_DIR/build-deps.sh" "$ROOT/build.log" || fail "build failed" "$ROOT/build.log"

# --- teardown stale -------------------------------------------------------
pkill -f "vllm.entrypoints.openai" 2>/dev/null || true
sleep 3
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' || true)
[ -n "$PIDS" ] && echo "$PIDS" | xargs -r kill -9 2>/dev/null || true
sleep 2
pkill -9 -f kvbm_hub 2>/dev/null || true
rm -f /tmp/velo-kvbm-*.sock 2>/dev/null || true
sleep 1

# --- hub ------------------------------------------------------------------
echo "[scenario-1] starting hub (indexer,p2p)"
bash "$SCRIPT_DIR/start-hub-remote-search.sh" "$ROOT/hub.log" &
HUB_PID=$!
deadline=$(( $(date +%s) + KVBM_HUB_READY_TIMEOUT ))
until curl -fsS -m 5 "http://127.0.0.1:$KVBM_HUB_CONTROL_PORT/health" >/dev/null 2>&1; do
  kill -0 "$HUB_PID" 2>/dev/null || fail "hub exited before ready" "$ROOT/hub.log"
  [ "$(date +%s)" -ge "$deadline" ] && fail "hub not ready after ${KVBM_HUB_READY_TIMEOUT}s" "$ROOT/hub.log"
  sleep 2
done
echo "[scenario-1] HUB UP"

# --- instance A (port 8000, GPUs 0,1, NUMA 0, TP=2) -----------------------
export RUST_LOG=${RUST_LOG:-info,kvbm_connector=debug,kvbm_engine=debug,kvbm_audit=info}
echo "[scenario-1] launching A (:8000 GPUs 0,1 NUMA 0 TP=2)"
PORT=8000 GPUS=0,1 NUMA=0 TP=2 FEATURES=indexer,p2p ROLE= \
  bash "$SCRIPT_DIR/launch-instance.sh" > "$ROOT/instance_a.log" 2>&1 &
A_PID=$!
deadline=$(( $(date +%s) + KVBM_VLLM_READY_TIMEOUT ))
until curl -fsS -m 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; do
  kill -0 "$A_PID" 2>/dev/null || fail "A exited before ready" "$ROOT/instance_a.log"
  [ "$(date +%s)" -ge "$deadline" ] && fail "A not ready after ${KVBM_VLLM_READY_TIMEOUT}s" "$ROOT/instance_a.log"
  sleep 5
done
echo "[scenario-1] A UP"

# --- instance B (port 8001, GPUs 2,3, NUMA 1, TP=2) -----------------------
echo "[scenario-1] launching B (:8001 GPUs 2,3 NUMA 1 TP=2)"
PORT=8001 GPUS=2,3 NUMA=1 TP=2 FEATURES=indexer,p2p ROLE= \
  bash "$SCRIPT_DIR/launch-instance.sh" > "$ROOT/instance_b.log" 2>&1 &
B_PID=$!
deadline=$(( $(date +%s) + KVBM_VLLM_READY_TIMEOUT ))
until curl -fsS -m 5 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; do
  kill -0 "$B_PID" 2>/dev/null || fail "B exited before ready" "$ROOT/instance_b.log"
  [ "$(date +%s)" -ge "$deadline" ] && fail "B not ready after ${KVBM_VLLM_READY_TIMEOUT}s" "$ROOT/instance_b.log"
  sleep 5
done
echo "[scenario-1] B UP"

# --- run smoke ------------------------------------------------------------
bash "$SCRIPT_DIR/remote-search-smoke.sh" "$ROOT"
