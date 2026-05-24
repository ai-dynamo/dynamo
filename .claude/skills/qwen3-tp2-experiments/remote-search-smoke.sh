#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Remote-search smoke for the qwen3-tp2-experiments bundle. ASSUMES hub +
# both instances are already up (the orchestrator does bringup + readiness).
# Validates the same invariants as .claude/skills/remote-search-smoke/:
#
#   1. R -> A indexes blocks on the hub (A is a holder).
#   2. R -> A again (warm) returns a golden completion and A does NOT initiate
#      a remote pull (it holds everything locally).
#   3. R -> B (cold G2) stalls on GNMT, hits the hub /query for holders,
#      opens a session on A, RDMA-pulls blocks into B's G2, onboards them,
#      and decodes. The pulled-KV path must yield bit-identical greedy decode
#      to the golden.
#
# Prompt is ~280 tokens (vs ~70 in the dev smoke) so multiple bs=64 blocks
# land in A's G2 → B has real blocks to remote-search for.
#
# Usage: bash remote-search-smoke.sh <experiment_root>
set -eu

SMOKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SMOKE_DIR/env.sh"

ROOT=${1:?"usage: $0 <experiment_root>"}
HUB_BASE="http://127.0.0.1:$KVBM_HUB_DISCOVERY_PORT"
KVI="$HUB_BASE/v1/features/indexer"
PULL_WAIT_TIMEOUT=${KVBM_PULL_WAIT_TIMEOUT:-120}

strip_ansi() { sed 's/\x1b\[[0-9;]*m//g'; }
fail() {
  echo "FATAL: $1" >&2
  [ -n "${2:-}" ] && [ -f "$2" ] && { echo "--- tail $2 ---" >&2; tail -n 30 "$2" | strip_ansi >&2; }
  exit 1
}

# A long-enough prompt that several 64-token blocks land in G2. ~280 tokens.
PROMPT='The kvbm hub mediates the discovery of where each block of the prompt is currently held in the cluster. When a connector on a remote instance receives a request whose tokens it has not seen before, it asks the hub which other instances already hold the matching blocks, then opens a session on one of those holders and pulls the blocks via RDMA. The blocks are placed in G2 host memory, onboarded into the G1 GPU cache, and the engine then resumes the regular prefill or decode path. This avoids recomputing prefill on every cache miss when another instance already has the work cached, and is the substrate that conditional disaggregation builds on. Pulling is layout-checked at session-open time, so the holder and the puller must agree on block size, head count, and layout mode.'

issue() {  # issue <port> -- fire-and-forget warm
  M="$KVBM_MODEL" P="$PROMPT" python3 -c 'import json,os;print(json.dumps({"model":os.environ["M"],"prompt":os.environ["P"],"max_tokens":8,"temperature":0}))' \
    | curl -m 300 -sS -X POST "http://127.0.0.1:$1/v1/completions" -H 'Content-Type: application/json' -d @- >/dev/null
}

issue_text() {  # issue <port> -- blocking; echoes decoded completion
  local resp
  resp=$(M="$KVBM_MODEL" P="$PROMPT" python3 -c 'import json,os;print(json.dumps({"model":os.environ["M"],"prompt":os.environ["P"],"max_tokens":8,"temperature":0}))' \
    | curl -m 300 -sS -X POST "http://127.0.0.1:$1/v1/completions" -H 'Content-Type: application/json' -d @- || true)
  echo "$resp" | python3 -c 'import json,sys
try:
  print(json.load(sys.stdin)["choices"][0]["text"])
except Exception: pass' 2>/dev/null || true
}

scan_index() {  # echo "<pos> <hash_u128> <instances>" for first non-empty bucket
  local num_positions
  num_positions=$(curl -fsS "$KVI/config" 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("num_positions", 64))' 2>/dev/null || echo 64)
  for pos in $(seq 0 $((num_positions - 1))); do
    out=$(curl -fsS "$KVI/hashes/by_position/$pos" 2>/dev/null \
      | python3 -c 'import json,sys
b=json.load(sys.stdin)
e=b.get("entries") or []
if e: print(b["position"], e[0]["hash_u128"], ",".join(e[0]["instances"]))' 2>/dev/null || true)
    [ -n "$out" ] && { echo "$out"; return 0; }
  done
  return 1
}

# --- R -> A (1st): warm A; confirm blocks land in the index ---------------
echo "=== R -> A (:8000) #1: warm A's cache and index ==="
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
SAMPLE_POS=$(echo "$READ" | awk '{print $1}')
echo "OK: A's blocks indexed (A=$A_ID, sample pos=$SAMPLE_POS)"

# --- R -> A (2nd): warm path -- A must NOT initiate a remote pull ----------
echo "=== R -> A (:8000) #2: warm path (must skip remote search) ==="
A_TEXT=$(issue_text 8000)
echo "A golden text: $(printf %q "$A_TEXT")"
[ -n "$A_TEXT" ] || fail "A produced no completion on the warm request" "$ROOT/instance_a.log"
A_SELF_PULL=$(grep -acE 'kvbm_audit.*event="transfer_pull_started"' "$ROOT/instance_a.log" 2>/dev/null || true)
A_SELF_PULL=${A_SELF_PULL:-0}
[ "$A_SELF_PULL" -eq 0 ] || fail "A initiated $A_SELF_PULL remote pull(s) but it holds all blocks locally" "$ROOT/instance_a.log"
echo "OK: A did not initiate any remote pull on the warm path."

# --- R -> B (same prompt, cold G2): blocking completion --------------------
echo "=== R -> B (:8001): cold cache -> remote search -> pull -> prefill -> decode ==="
B_TEXT=$(issue_text 8001)
echo "B decoded text: $(printf %q "$B_TEXT")"
[ -n "$B_TEXT" ] || fail "B produced no completion (decode did not run)" "$ROOT/instance_b.log"

if [ "$B_TEXT" = "$A_TEXT" ]; then
  echo "OK: B's output matches A's golden output (pulled KV is correct)."
else
  fail "B output != A golden output -- pulled KV may be incorrect.
        A: $(printf %q "$A_TEXT")
        B: $(printf %q "$B_TEXT")" "$ROOT/instance_b.log"
fi

# request_finished fires just after the response; give it a beat.
deadline=$(( $(date +%s) + 15 ))
until grep -qaE 'kvbm_audit.*event="request_finished"' "$ROOT/instance_b.log" 2>/dev/null; do
  [ "$(date +%s)" -ge "$deadline" ] && break; sleep 1
done

# Audit-event counts.
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
[ "$B_ONBOARD" -ge 1 ]  || fail "B never completed onboarding G2->G1 (onboard_complete)" "$ROOT/instance_b.log"
[ "$B_FIN" -ge 1 ]      || fail "B request never finished (request_finished)" "$ROOT/instance_b.log"

echo
echo "  remote-search smoke PASSED"
echo "  A=$A_ID  B stalled on the remote search, pulled A's blocks, then prefilled+decoded."
echo "  A skipped remote search on its warm request (held all blocks locally)."
echo "  golden A: $(printf %q "$A_TEXT")"
echo "  B output: $(printf %q "$B_TEXT")  (== A)"
echo "  logs: $ROOT"
