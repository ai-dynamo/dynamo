#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Scenario 2: Conditional-disagg uniform TP=2.
#
#   Hub          --features disagg (auto-expands to disagg+p2p)
#                +--prefill-vllm-url http://127.0.0.1:8000
#   Prefill      port 8000   GPUs 0,1   NUMA 0   TP=2   role=prefill
#   Decode       port 8001   GPUs 2,3   NUMA 1   TP=2   role=decode
#
# Smoke: two-request golden (R1 cold + reset prefill G2 + R2 warm).
#
# Usage: bash scenario-disagg-uniform-tp2.sh
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/env.sh"
. "$SCRIPT_DIR/numa-lib.sh"

LABEL=${KVBM_EXPERIMENT_LABEL:-disagg-tp2-uniform}
ROOT=$(bash "$KVBM_REPO/.claude/skills/disagg-bringup/new-experiment.sh" "$LABEL")
echo "EXP=$ROOT"

strip_ansi() { sed 's/\x1b\[[0-9;]*m//g'; }
cleanup() {
  [ "${KVBM_KEEP_ALIVE:-0}" = "1" ] && { echo "[scenario-2] KVBM_KEEP_ALIVE=1 — leaving hub + vLLMs running"; return; }
  echo "[scenario-2] cleanup: spinning down hub + vLLMs"
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

echo "[scenario-2] venv import check"
"$KVBM_VENV/bin/python3" -c "import vllm,kvbm; print('vllm', vllm.__version__, '; kvbm OK')" \
  || fail "venv $KVBM_VENV is missing vllm or kvbm"

echo "[scenario-2] verify GPU NUMA topology"
verify_numa_topology || fail "GPU NUMA topology mismatch -- bailing"

echo "[scenario-2] build kvbm_hub + kvbmctl"
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
echo "[scenario-2] starting hub (disagg)"
PREFILL_PORT=8000 bash "$SCRIPT_DIR/start-hub-disagg.sh" "$ROOT/hub.log" &
HUB_PID=$!
deadline=$(( $(date +%s) + KVBM_HUB_READY_TIMEOUT ))
until curl -fsS -m 5 "http://127.0.0.1:$KVBM_HUB_CONTROL_PORT/health" >/dev/null 2>&1; do
  kill -0 "$HUB_PID" 2>/dev/null || fail "hub exited before ready" "$ROOT/hub.log"
  [ "$(date +%s)" -ge "$deadline" ] && fail "hub not ready after ${KVBM_HUB_READY_TIMEOUT}s" "$ROOT/hub.log"
  sleep 2
done
echo "[scenario-2] HUB UP"

# --- prefill (port 8000, GPUs 0,1, NUMA 0, TP=2, role=prefill) ------------
# Launch + wait sequentially (prefill THEN decode) so the hub sees one
# registration at a time. Removes the wire_p2p race; parallel launches racing
# layout_compat is what makes asymmetric TP fail in Operational mode.
export RUST_LOG=${RUST_LOG:-info,kvbm_connector=debug,kvbm_audit=info}
echo "[scenario-2] launching Prefill (:8000 GPUs 0,1 NUMA 0 TP=2)"
PORT=8000 GPUS=0,1 NUMA=0 TP=2 FEATURES=disagg ROLE=prefill \
  bash "$SCRIPT_DIR/launch-instance.sh" > "$ROOT/prefill.log" 2>&1 &
PREFILL_PID=$!
deadline=$(( $(date +%s) + KVBM_VLLM_READY_TIMEOUT ))
until curl -fsS -m 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; do
  kill -0 "$PREFILL_PID" 2>/dev/null || fail "Prefill exited before ready" "$ROOT/prefill.log"
  [ "$(date +%s)" -ge "$deadline" ] && fail "Prefill not ready after ${KVBM_VLLM_READY_TIMEOUT}s" "$ROOT/prefill.log"
  sleep 5
done
echo "[scenario-2] Prefill UP"

# --- decode (port 8001, GPUs 2,3, NUMA 1, TP=2, role=decode) --------------
echo "[scenario-2] launching Decode (:8001 GPUs 2,3 NUMA 1 TP=2)"
PORT=8001 GPUS=2,3 NUMA=1 TP=2 FEATURES=disagg ROLE=decode \
  bash "$SCRIPT_DIR/launch-instance.sh" > "$ROOT/decode.log" 2>&1 &
DECODE_PID=$!
deadline=$(( $(date +%s) + KVBM_VLLM_READY_TIMEOUT ))
until curl -fsS -m 5 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; do
  kill -0 "$PREFILL_PID" 2>/dev/null || fail "Prefill died during Decode bringup" "$ROOT/prefill.log"
  kill -0 "$DECODE_PID"  2>/dev/null || fail "Decode exited before ready"        "$ROOT/decode.log"
  [ "$(date +%s)" -ge "$deadline" ] && fail "Decode not ready after ${KVBM_VLLM_READY_TIMEOUT}s" "$ROOT/decode.log"
  sleep 5
done
echo "[scenario-2] BOTH UP"

# --- run smoke ------------------------------------------------------------
bash "$SCRIPT_DIR/two-request-smoke.sh" "$ROOT"
