#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Scenario 3: Conditional-disagg asymmetric (Prefill TP=1 + Decode TP=2).
#
#   Hub          --features disagg (auto-expands to disagg+p2p)
#                +--prefill-vllm-url http://127.0.0.1:8000
#   Prefill      port 8000   GPU  2     NUMA 1   TP=1   role=prefill
#   Decode       port 8001   GPUs 0,1   NUMA 0   TP=2   role=decode
#
# Why this layout: keeps the disagg port convention (prefill=8000, decode=8001)
# and the existing TP=2-decode pattern from disagg-bringup/launch-decode-tp2.sh,
# while putting the big workload (decode TP=2) on NUMA 0. Asymmetric pull
# behavior is the headline thing to exercise here.
#
# Smoke: same two-request golden as the uniform run.
#
# Universal block layout is REQUIRED for asymmetric TP.
# The hub's P2P layout-compat in Operational mode enforces baseline.tp_size ==
# candidate.tp_size on the second registration (lib/kvbm-protocols/src/control/
# layout_compat.rs:161-166). Universal mode compares canonical shape only, so
# TP=1 and TP=2 peers can coexist; G1 stays native (Operational HND, vLLM-
# owned), G2 is Universal (canonical, transfer plane).
#
# Usage: bash scenario-disagg-asymmetric.sh
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Force universal layout before sourcing env.sh — otherwise the operational
# default would let the hub reject the second-to-register side.
export KVBM_BLOCK_LAYOUT=${KVBM_BLOCK_LAYOUT:-universal}
. "$SCRIPT_DIR/env.sh"
. "$SCRIPT_DIR/numa-lib.sh"

LABEL=${KVBM_EXPERIMENT_LABEL:-disagg-asymmetric-p1-d2}
ROOT=$(bash "$KVBM_REPO/.claude/skills/disagg-bringup/new-experiment.sh" "$LABEL")
echo "EXP=$ROOT"

strip_ansi() { sed 's/\x1b\[[0-9;]*m//g'; }
cleanup() {
  [ "${KVBM_KEEP_ALIVE:-0}" = "1" ] && { echo "[scenario-3] KVBM_KEEP_ALIVE=1 — leaving hub + vLLMs running"; return; }
  echo "[scenario-3] cleanup: spinning down hub + vLLMs"
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

echo "[scenario-3] venv import check"
"$KVBM_VENV/bin/python3" -c "import vllm,kvbm; print('vllm', vllm.__version__, '; kvbm OK')" \
  || fail "venv $KVBM_VENV is missing vllm or kvbm"

echo "[scenario-3] verify GPU NUMA topology"
verify_numa_topology || fail "GPU NUMA topology mismatch -- bailing"

echo "[scenario-3] build kvbm_hub + kvbmctl"
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
echo "[scenario-3] starting hub (disagg)"
PREFILL_PORT=8000 bash "$SCRIPT_DIR/start-hub-disagg.sh" "$ROOT/hub.log" &
HUB_PID=$!
deadline=$(( $(date +%s) + KVBM_HUB_READY_TIMEOUT ))
until curl -fsS -m 5 "http://127.0.0.1:$KVBM_HUB_CONTROL_PORT/health" >/dev/null 2>&1; do
  kill -0 "$HUB_PID" 2>/dev/null || fail "hub exited before ready" "$ROOT/hub.log"
  [ "$(date +%s)" -ge "$deadline" ] && fail "hub not ready after ${KVBM_HUB_READY_TIMEOUT}s" "$ROOT/hub.log"
  sleep 2
done
echo "[scenario-3] HUB UP"

# --- prefill (port 8000, GPU 2, NUMA 1, TP=1, role=prefill) ---------------
# Launch + wait sequentially so hub registration is one-at-a-time. Even
# with universal layout this avoids a wire_p2p race on the second-to-register.
export RUST_LOG=${RUST_LOG:-info,kvbm_connector=debug,kvbm_audit=info}
echo "[scenario-3] launching Prefill (:8000 GPU 2 NUMA 1 TP=1)"
PORT=8000 GPUS=2 NUMA=1 TP=1 FEATURES=disagg ROLE=prefill \
  bash "$SCRIPT_DIR/launch-instance.sh" > "$ROOT/prefill.log" 2>&1 &
PREFILL_PID=$!
deadline=$(( $(date +%s) + KVBM_VLLM_READY_TIMEOUT ))
until curl -fsS -m 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; do
  kill -0 "$PREFILL_PID" 2>/dev/null || fail "Prefill exited before ready" "$ROOT/prefill.log"
  [ "$(date +%s)" -ge "$deadline" ] && fail "Prefill not ready after ${KVBM_VLLM_READY_TIMEOUT}s" "$ROOT/prefill.log"
  sleep 5
done
echo "[scenario-3] Prefill UP"

# --- decode (port 8001, GPUs 0,1, NUMA 0, TP=2, role=decode) --------------
echo "[scenario-3] launching Decode (:8001 GPUs 0,1 NUMA 0 TP=2)"
PORT=8001 GPUS=0,1 NUMA=0 TP=2 FEATURES=disagg ROLE=decode \
  bash "$SCRIPT_DIR/launch-instance.sh" > "$ROOT/decode.log" 2>&1 &
DECODE_PID=$!
deadline=$(( $(date +%s) + KVBM_VLLM_READY_TIMEOUT ))
until curl -fsS -m 5 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; do
  kill -0 "$PREFILL_PID" 2>/dev/null || fail "Prefill died during Decode bringup" "$ROOT/prefill.log"
  kill -0 "$DECODE_PID"  2>/dev/null || fail "Decode exited before ready"        "$ROOT/decode.log"
  [ "$(date +%s)" -ge "$deadline" ] && fail "Decode not ready after ${KVBM_VLLM_READY_TIMEOUT}s" "$ROOT/decode.log"
  sleep 5
done
echo "[scenario-3] BOTH UP"

# --- run smoke ------------------------------------------------------------
bash "$SCRIPT_DIR/two-request-smoke.sh" "$ROOT"
