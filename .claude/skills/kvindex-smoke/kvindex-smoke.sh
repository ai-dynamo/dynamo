#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# KV-index smoke: two plain aggregated instances + a hub running the KV
# indexer feature. No P/D specialization, no NATS, no consolidator.
#
# Flow:
#   1. Start hub with KVBM_HUB_FEATURES=indexer. Wait for /health.
#   2. Launch instance A (:8000) and B (:8001), both publishing G2 block
#      events to the hub. Wait for /v1/models on each.
#   3. Confirm each connector wired its KV-index publisher (parse logs for
#      the instance_id it stamped).
#   4. Issue request R to A. Its G2 offload registers blocks → events → hub.
#      Assert a block appears in the index mapped to A only.
#   5. POST /query with that block's hash → hit maps to A (the query API).
#   6. Issue the SAME request R to B. Assert the same block now maps to A AND B.
#
# Usage: bash kvindex-smoke.sh [logs_dir]
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

LABEL=${KVBM_EXPERIMENT_LABEL:-kvindex}
ROOT=${1:-$(bash "$BRINGUP/new-experiment.sh" "$LABEL")}
echo "EXP=$ROOT"

strip_ansi() { sed 's/\x1b\[[0-9;]*m//g'; }

fail() {
  echo "FATAL: $1" >&2
  [ -n "${2:-}" ] && [ -f "$2" ] && { echo "--- tail $2 ---" >&2; tail -n 25 "$2" | strip_ansi >&2; }
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
echo "HUB UP. config:"; curl -fsS "$KVI/config"; echo

# --- two plain instances (sequential; vLLM profiler races on unified mem) -
echo "launching instance A (:8000)…"
KVBM_INSTANCE_PORT=8000 RUST_LOG=${RUST_LOG:-info,kvbm_connector=debug} \
  bash "$SMOKE_DIR/launch-instance.sh" > "$ROOT/instance_a.log" 2>&1 &
A_PID=$!
deadline=$(( $(date +%s) + VLLM_READY_TIMEOUT ))
until curl -fsS -m 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; do
  kill -0 "$A_PID" 2>/dev/null || fail "instance A exited before ready" "$ROOT/instance_a.log"
  [ "$(date +%s)" -ge "$deadline" ] && fail "instance A not ready" "$ROOT/instance_a.log"
  sleep 5
done
echo "A UP."

echo "launching instance B (:8001)…"
KVBM_INSTANCE_PORT=8001 RUST_LOG=${RUST_LOG:-info,kvbm_connector=debug} \
  bash "$SMOKE_DIR/launch-instance.sh" > "$ROOT/instance_b.log" 2>&1 &
B_PID=$!
deadline=$(( $(date +%s) + VLLM_READY_TIMEOUT ))
until curl -fsS -m 5 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; do
  kill -0 "$B_PID" 2>/dev/null || fail "instance B exited before ready" "$ROOT/instance_b.log"
  [ "$(date +%s)" -ge "$deadline" ] && fail "instance B not ready" "$ROOT/instance_b.log"
  sleep 5
done
echo "B UP."

# Instance ids are derived from the index data itself (the connector defers
# init to the first request, and its Rust logs may not surface from the
# EngineCore subprocess — so we don't parse logs for ids).

# --- request prompt (≥ a few full 16-token blocks) ------------------------
# Resolve the model name the instances launched with (from the profile) — set
# before any request so the curl body carries the right "model".
. "$BRINGUP/hardware-profiles.sh"; kvbm_apply_disagg_bringup_profile

PROMPT='The quick brown fox jumps over the lazy dog and then keeps on running through the meadow until it reaches the river where it finally stops to drink some water and rest for a while before continuing its journey home through the forest path.'
issue() {  # issue <port> — model in $KVBM_MODEL, prompt in $PROMPT
  M="$KVBM_MODEL" P="$PROMPT" python3 -c 'import json,os;print(json.dumps({"model":os.environ["M"],"prompt":os.environ["P"],"max_tokens":8,"temperature":0}))' \
    | curl -m 90 -sS -X POST "http://127.0.0.1:$1/v1/completions" -H 'Content-Type: application/json' -d @- >/dev/null
}

# Scan position buckets; echo "<pos> <hash_u128> <comma-instances>" for the
# first non-empty bucket.
scan_index() {
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

# Given a hash_u128, ask the hub /query who holds it -> comma-joined instances.
# PLH serializes as serialize_bytes(be_bytes(u128)) -> JSON array of 16 bytes.
query_owners() {
  HASH="$1" python3 - "$KVI" <<'PY'
import json,os,sys,urllib.request
kvi=sys.argv[1]
plh=list(int(os.environ["HASH"]).to_bytes(16,"big"))
body=json.dumps({"hashes":[plh]}).encode()
req=urllib.request.Request(kvi+"/query",data=body,headers={"Content-Type":"application/json"})
hit=json.load(urllib.request.urlopen(req,timeout=10)).get("hit")
print(",".join(hit["instances"]) if hit else "")
PY
}
ncsv() { echo "$1" | tr ',' '\n' | grep -c . ; }   # count comma-list members

# --- R -> A: block appears in the index, owned by A only ------------------
echo "=== R -> A (:8000) ==="
READ=""
for i in $(seq 1 20); do
  issue 8000
  sleep 2   # G2 offload + event batch flush
  READ=$(scan_index || true)
  [ -n "$READ" ] && break
  echo "  …no index entries yet (attempt $i)"
done
[ -n "$READ" ] || fail "no blocks indexed after requests to A (publisher not wired? check config injection)" "$ROOT/instance_a.log"
POS=$(echo "$READ" | awk '{print $1}')
HASH=$(echo "$READ" | awk '{print $2}')
OWNERS_A=$(echo "$READ" | awk '{print $3}')
echo "indexed block: position=$POS hash_u128=$HASH owners=[$OWNERS_A]"
[ "$(ncsv "$OWNERS_A")" -eq 1 ] || fail "expected exactly one owner after A, got [$OWNERS_A]"
A_ID="$OWNERS_A"
echo "OK: block at position $POS owned solely by A ($A_ID)"

# --- query API: given the hash, who has it? (expect A) --------------------
Q_A=$(query_owners "$HASH")
echo "query owners after A: [$Q_A]"
echo "$Q_A" | tr ',' '\n' | grep -qx "$A_ID" || fail "/query did not return A ($A_ID) for hash $HASH"
echo "OK: /query maps the hash to A"

# --- R -> B (same prompt): same block now owned by A AND B ----------------
echo "=== R -> B (:8001) (same prompt) ==="
Q_AB=""
for i in $(seq 1 20); do
  issue 8001
  sleep 2
  Q_AB=$(query_owners "$HASH" || true)
  [ "$(ncsv "$Q_AB")" -ge 2 ] && break
  echo "  …only [$Q_AB] so far (attempt $i)"
done
echo "query owners after B: [$Q_AB]"
echo "$Q_AB" | tr ',' '\n' | grep -qx "$A_ID" || fail "after B, /query lost A ($A_ID); owners=[$Q_AB]"
B_ID=$(echo "$Q_AB" | tr ',' '\n' | grep -vx "$A_ID" | head -1)
[ -n "$B_ID" ] || fail "after B, /query shows no second owner; owners=[$Q_AB]"
echo "OK: /query maps the hash to BOTH A ($A_ID) and B ($B_ID)"

echo
echo "================================================================"
echo "  kvindex smoke PASSED"
echo "  position=$POS hash=$HASH"
echo "  A=$A_ID  B=$B_ID"
echo "  owners after A: [$Q_A]   owners after B: [$Q_AB]"
echo "  logs: $ROOT"
echo "================================================================"
