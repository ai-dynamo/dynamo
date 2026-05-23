#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Remote-search smoke: two plain aggregated instances + a hub running indexer +
# p2p. Proves the leader-driven, out-of-band remote search: B's connector, on a
# cold-cache request, asks the hub indexer who holds the prompt's blocks (A),
# opens a transfer session on A, and RDMA-pulls the blocks into B's local G2.
#
# Flow:
#   1. Start hub (indexer,p2p). Wait /health.
#   2. Launch A (:8000) and B (:8001). Both wire the index publisher AND register
#      Feature::P2P; both have remote_search enabled.
#   3. Issue R to A. A computes + offloads to G2 → blocks indexed on the hub.
#   4. Confirm A's blocks are in the index (so discovery has a holder to find).
#   5. Issue R to B (same prompt, B's G2 cold). B's GNMT kicks off a remote
#      search → open_session on A → pull_from_session into B's G2.
#   6. Assert via the kvbm_audit stream: B emits transfer_pull_started/
#      transfer_pull_completed; A emits transfer_session_opened. That is the
#      block transfer A→B, driven internally by B's leader.
#
# Usage: bash remote-search-smoke.sh [logs_dir]
set -eu

SMOKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=${KVBM_REPO:-$(cd "$SMOKE_DIR/../../.." && pwd)}
BRINGUP=$REPO/.claude/skills/disagg-bringup
export KVBM_VENV=${KVBM_VENV:-$REPO/.sandbox}
export KVBM_REPO=$REPO

HUB_DISC=${KVBM_HUB_DISCOVERY_PORT:-1337}
HUB_CTRL=${KVBM_HUB_CONTROL_PORT:-8337}
HUB_BASE="http://127.0.0.1:$HUB_DISC"
KVI="$HUB_BASE/v1/features/indexer"
HUB_READY_TIMEOUT=${KVBM_HUB_READY_TIMEOUT:-300}
VLLM_READY_TIMEOUT=${KVBM_VLLM_READY_TIMEOUT:-300}
# Give B's background remote search (discover → open → RDMA pull → close) time.
PULL_WAIT_TIMEOUT=${KVBM_PULL_WAIT_TIMEOUT:-60}

# Surface the leader-side transfer audit events (they run in the EngineCore
# subprocess but inherit its stdout → the instance log).
export RUST_LOG=${RUST_LOG:-info,kvbm_connector=debug,kvbm_engine=debug,kvbm_audit=info}

LABEL=${KVBM_EXPERIMENT_LABEL:-remote-search}
ROOT=${1:-$(bash "$BRINGUP/new-experiment.sh" "$LABEL")}
echo "EXP=$ROOT"

strip_ansi() { sed 's/\x1b\[[0-9;]*m//g'; }
fail() {
  echo "FATAL: $1" >&2
  [ -n "${2:-}" ] && [ -f "$2" ] && { echo "--- tail $2 ---" >&2; tail -n 30 "$2" | strip_ansi >&2; }
  pkill -f "vllm.entrypoints.openai" 2>/dev/null || true
  pkill -9 -f kvbm_hub 2>/dev/null || true
  exit 1
}

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
bash "$SMOKE_DIR/start-hub.sh" "$ROOT/hub.log" &
HUB_PID=$!
echo "waiting for hub /health (timeout ${HUB_READY_TIMEOUT}s)..."
deadline=$(( $(date +%s) + HUB_READY_TIMEOUT ))
until curl -fsS -m 5 "http://127.0.0.1:$HUB_CTRL/health" >/dev/null 2>&1; do
  kill -0 "$HUB_PID" 2>/dev/null || fail "hub exited before ready" "$ROOT/hub.log"
  [ "$(date +%s)" -ge "$deadline" ] && fail "hub not ready" "$ROOT/hub.log"
  sleep 2
done
echo "HUB UP (features: indexer,p2p)."

# --- two plain instances (sequential; vLLM profiler races on unified mem) -
echo "launching instance A (:8000)…"
KVBM_INSTANCE_PORT=8000 bash "$SMOKE_DIR/launch-instance.sh" > "$ROOT/instance_a.log" 2>&1 &
A_PID=$!
deadline=$(( $(date +%s) + VLLM_READY_TIMEOUT ))
until curl -fsS -m 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; do
  kill -0 "$A_PID" 2>/dev/null || fail "instance A exited before ready" "$ROOT/instance_a.log"
  [ "$(date +%s)" -ge "$deadline" ] && fail "instance A not ready" "$ROOT/instance_a.log"
  sleep 5
done
echo "A UP."

echo "launching instance B (:8001)…"
KVBM_INSTANCE_PORT=8001 bash "$SMOKE_DIR/launch-instance.sh" > "$ROOT/instance_b.log" 2>&1 &
B_PID=$!
deadline=$(( $(date +%s) + VLLM_READY_TIMEOUT ))
until curl -fsS -m 5 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; do
  kill -0 "$B_PID" 2>/dev/null || fail "instance B exited before ready" "$ROOT/instance_b.log"
  [ "$(date +%s)" -ge "$deadline" ] && fail "instance B not ready" "$ROOT/instance_b.log"
  sleep 5
done
echo "B UP."

. "$BRINGUP/hardware-profiles.sh"; kvbm_apply_disagg_bringup_profile

# Long enough to span several full 16-token blocks so B has remote blocks to pull.
PROMPT='The quick brown fox jumps over the lazy dog and then keeps on running through the meadow until it reaches the river where it finally stops to drink some water and rest for a while before continuing its journey home through the forest path under a bright sky.'
issue() {  # issue <port> — fire-and-forget (warming)
  M="$KVBM_MODEL" P="$PROMPT" python3 -c 'import json,os;print(json.dumps({"model":os.environ["M"],"prompt":os.environ["P"],"max_tokens":8,"temperature":0}))' \
    | curl -m 90 -sS -X POST "http://127.0.0.1:$1/v1/completions" -H 'Content-Type: application/json' -d @- >/dev/null
}

issue_text() {  # issue <port> — blocking; echoes the decoded completion text
  local resp
  resp=$(M="$KVBM_MODEL" P="$PROMPT" python3 -c 'import json,os;print(json.dumps({"model":os.environ["M"],"prompt":os.environ["P"],"max_tokens":8,"temperature":0}))' \
    | curl -m 120 -sS -X POST "http://127.0.0.1:$1/v1/completions" -H 'Content-Type: application/json' -d @- || true)
  echo "$resp" | python3 -c 'import json,sys
try:
  print(json.load(sys.stdin)["choices"][0]["text"])
except Exception: pass' 2>/dev/null || true
}

scan_index() {  # echo "<pos> <hash_u128> <instances>" for first non-empty bucket
  for pos in $(seq 0 63); do
    out=$(curl -fsS "$KVI/hashes/by_position/$pos" 2>/dev/null \
      | python3 -c 'import json,sys
b=json.load(sys.stdin)
e=b.get("entries") or []
if e: print(b["position"], e[0]["hash_u128"], ",".join(e[0]["instances"]))' 2>/dev/null || true)
    [ -n "$out" ] && { echo "$out"; return 0; }
  done
  return 1
}

# --- R -> A (1st): warm A and confirm its blocks land in the index --------
echo "=== R -> A (:8000) #1: warm cache + index ==="
READ=""
for i in $(seq 1 20); do
  issue 8000
  sleep 2
  READ=$(scan_index || true)
  [ -n "$READ" ] && break
  echo "  …no index entries yet (attempt $i)"
done
[ -n "$READ" ] || fail "no blocks indexed after requests to A (publisher not wired?)" "$ROOT/instance_a.log"
A_ID=$(echo "$READ" | awk '{print $3}')
echo "OK: A's blocks indexed (A=$A_ID, sample pos=$(echo "$READ"|awk '{print $1}'))"

# --- R -> A (2nd): warm path — A already holds every block, so it must NOT
#     remote-search (local_covers_all → Ready). Capture its output as the
#     golden completion to validate B against. ---------------------------------
echo "=== R -> A (:8000) #2: warm path (must skip remote search) ==="
A_TEXT=$(issue_text 8000)
echo "A golden text: $(printf %q "$A_TEXT")"
[ -n "$A_TEXT" ] || fail "A produced no completion on the warm request" "$ROOT/instance_a.log"
# A holds the blocks locally: it must never open a pull as the *initiator*.
# (transfer_session_opened on A is the HOLDER side, expected when B pulls — not
# checked here. transfer_pull_started is the PULLER/initiator side.)
A_SELF_PULL=$(grep -acE 'kvbm_audit.*event="transfer_pull_started"' "$ROOT/instance_a.log" 2>/dev/null || true)
A_SELF_PULL=${A_SELF_PULL:-0}
[ "$A_SELF_PULL" -eq 0 ] || fail "A initiated a remote pull ($A_SELF_PULL) but it holds all blocks locally — it must skip remote search (self-candidate must be filtered)" "$ROOT/instance_a.log"
echo "OK: A did not initiate any remote pull (skipped remote search on the warm path)."

# --- R -> B (same prompt, cold G2): one blocking request --------------------
# The completion call blocks until B's leader has: stalled GNMT on the in-flight
# remote search → pulled A's blocks → onboarded G2→G1 → prefilled the remaining
# tail → decoded. So when curl returns, the whole pull→prefill→decode chain is
# done and every audit marker has fired.
echo "=== R -> B (:8001): cold cache → remote search → pull → prefill → decode ==="
B_TEXT=$(issue_text 8001)
echo "B decoded text: $(printf %q "$B_TEXT")"
[ -n "$B_TEXT" ] || fail "B produced no completion (decode did not run)" "$ROOT/instance_b.log"

# Correctness: greedy (temp=0) decode of the same prompt must produce the same
# text whether computed locally (A) or served from A's pulled KV (B). A mismatch
# means the pulled cache is wrong (layout/order/corruption), not just absent.
if [ "$B_TEXT" = "$A_TEXT" ]; then
  echo "OK: B's output matches A's golden output (pulled KV is correct)."
else
  fail "B output != A golden output — pulled KV may be incorrect.
        A: $(printf %q "$A_TEXT")
        B: $(printf %q "$B_TEXT")" "$ROOT/instance_b.log"
fi

# request_finished fires just after the response; give it a beat to land.
deadline=$(( $(date +%s) + 15 ))
until grep -qaE 'kvbm_audit.*event="request_finished"' "$ROOT/instance_b.log" 2>/dev/null; do
  [ "$(date +%s)" -ge "$deadline" ] && break; sleep 1
done

# --- collect the per-request lifecycle markers + transfer events ----------
# `grep -c` prints "0" AND exits 1 on no match; mask the exit with `|| true`
# (a bare `|| echo 0` would append a second "0" and break integer tests).
count_ev() { local c; c=$(grep -acE "kvbm_audit.*event=\"$2\"" "$1" 2>/dev/null || true); echo "${c:-0}"; }
ev_b() { count_ev "$ROOT/instance_b.log" "$1"; }
ev_a() { count_ev "$ROOT/instance_a.log" "$1"; }
B_PENDING=$(ev_b gnmt_pending); B_MATCHED=$(ev_b gnmt_matched)
B_PULL_START=$(ev_b transfer_pull_started); B_PULL_DONE=$(ev_b transfer_pull_completed)
B_ONBOARD=$(ev_b onboard_complete); B_FIN=$(ev_b request_finished)
A_OPEN=$(ev_a transfer_session_opened)

echo
echo "================================================================"
echo " B request lifecycle (counts): gnmt_pending=$B_PENDING gnmt_matched=$B_MATCHED"
echo "   transfer_pull_started=$B_PULL_START transfer_pull_completed=$B_PULL_DONE"
echo "   onboard_complete=$B_ONBOARD request_finished=$B_FIN"
echo " A holder: transfer_session_opened=$A_OPEN"
echo "----------------------------------------------------------------"
echo " ordered B timeline (kvbm_audit):"
grep -aE 'kvbm_audit.*event="(gnmt_pending|gnmt_matched|transfer_pull_started|transfer_pull_completed|onboard_start|onboard_complete|request_finished)"' \
  "$ROOT/instance_b.log" 2>/dev/null | strip_ansi \
  | sed -E 's/.*([0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+).*event="([a-z_]+)"(.*)/  \1  \2 \3/' | head -25
echo "================================================================"

[ "$B_PULL_DONE" -ge 1 ] || fail "B never completed a remote pull (transfer_pull_completed)" "$ROOT/instance_b.log"
[ "$A_OPEN" -ge 1 ]     || fail "A never opened a transfer session for B (transfer_session_opened)" "$ROOT/instance_a.log"
[ "$B_MATCHED" -ge 1 ]  || fail "B GNMT never resolved to an external match (gnmt_matched)" "$ROOT/instance_b.log"
[ "$B_ONBOARD" -ge 1 ]  || fail "B never completed onboarding G2→G1 (onboard_complete)" "$ROOT/instance_b.log"
[ "$B_FIN" -ge 1 ]      || fail "B request never finished (request_finished) — decode may not have run" "$ROOT/instance_b.log"

# --- render the trace.html (RDMA pull → prefill → decode timeline) --------
TRACE_PY="$REPO/.claude/skills/disagg-trace/p2p-trace.py"
if [ -f "$TRACE_PY" ]; then
  python3 "$TRACE_PY" "$ROOT/" && echo "trace rendered: $ROOT/trace.html"
fi

echo
echo "  remote-search smoke PASSED"
echo "  A=$A_ID  B stalled on the remote search, pulled A's blocks, then prefilled+decoded."
echo "  A skipped remote search on its warm request (held all blocks locally)."
echo "  golden A: $(printf %q "$A_TEXT")"
echo "  B output: $(printf %q "$B_TEXT")  (== A)"
echo "  logs + trace: $ROOT"
