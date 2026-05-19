#!/bin/bash
# p2p-smoke.sh — exercise hub-mediated P2P G2 block transfer.
#
# Topology:  hub  +  instance_a (port 8000)  +  instance_b (port 8002)
#                            ^                       ^
#                            \__ both Qwen3-0.6B, both registered with hub
#
# Flow:
#   R1 → instance_a            (warms G2 on A; emits offload_register_complete
#                               audit event with the ISL block hashes)
#   scrape audit log on A      (extract those hashes)
#   POST /open_session   on A  (hub forwards to A via velo)
#   POST /pull_from_session on B (hub forwards to B, B connects to A's session
#                               via velo and pulls the blocks into B's G2)
#   POST /close_session  on A  (cleanup)
#   R2 → instance_b            (same prompt; expect G2 cache hit in B's audit)
#
# Usage:  bash p2p-smoke.sh [logs_dir]
#   If logs_dir not given, mints $KVBM_EXPERIMENTS_DIR/<ts>-p2p-smoke/.
#
# Env vars:
#   KVBM_REPO  (default: worktree root inferred from script location)
#   P2P_GMU    (default: 0.15)   — per-instance GPU memory util
#   P2P_BLOCKS (default: 16)     — desired ISL block count (Qwen3-0.6B
#                                  block_size=16 → 16 blocks ≈ 256 tokens)
set -eu

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DYNAMO=${KVBM_REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
SKILL_BRINGUP=$DYNAMO/.claude/skills/disagg-bringup
SKILL_TRACE=$DYNAMO/.claude/skills/disagg-trace
LABEL=${KVBM_EXPERIMENT_LABEL:-p2p-smoke}
P2P_GMU=${P2P_GMU:-0.15}
P2P_BLOCKS=${P2P_BLOCKS:-16}
export P2P_GMU

ROOT=${1:-$(bash "$SKILL_BRINGUP/new-experiment.sh" "$LABEL")}
echo "EXP=$ROOT"
echo "$ROOT" > /tmp/p2p-trace-current-exp

# ----------------------------------------------------------------------
# 0. Teardown stale processes (the GB10 reports memory as Not Supported,
#    so we kill by process name rather than by GPU process list).
# ----------------------------------------------------------------------
pkill -f "vllm.entrypoints.openai" 2>/dev/null || true
pkill -9 -f kvbm_hub 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
sleep 3

# ----------------------------------------------------------------------
# 1. Hub up.
# ----------------------------------------------------------------------
bash "$SKILL_BRINGUP/start-hub.sh" "$ROOT/hub.log" &
disown
sleep 2

# Wait for hub HTTP
until curl -fsS http://127.0.0.1:8337/health >/dev/null 2>&1; do sleep 1; done
echo "HUB UP"

# ----------------------------------------------------------------------
# 2. Launch instances SEQUENTIALLY on ports 8000 and 8002.
#    role=prefill for A, role=decode for B — purely cosmetic (transfer
#    endpoints work on either; we never use the CD prefill queue).
#
#    Sequential is load-bearing on the Spark's unified-memory GB10:
#    vLLM's GPU-memory profiler measures free memory at startup and
#    allocates GMU * total. Launching in parallel makes both instances
#    see the same "free", so the second one can't fit its KV cache and
#    errors with "No available memory for the cache blocks". Waiting
#    for A's /v1/models means A's GPU allocation is settled before B
#    profiles.
# ----------------------------------------------------------------------
RUST_LOG=${RUST_LOG:-info,kvbm_connector=debug,kvbm_engine=info,kvbm_audit=info}
export RUST_LOG
P2P_STARTUP_TIMEOUT=${P2P_STARTUP_TIMEOUT:-300}

wait_for_models() {
  local port=$1
  local timeout=$2
  local deadline=$(( $(date +%s) + timeout ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 3
  done
  return 1
}

echo "launching instance A (port 8000, role=prefill)..."
P2P_PORT=8000 P2P_ROLE=prefill bash "$SCRIPT_DIR/launch-instance.sh" \
  > "$ROOT/instance_a.log" 2>&1 &
disown
if ! wait_for_models 8000 "$P2P_STARTUP_TIMEOUT"; then
  echo "FAIL: instance A did not respond on /v1/models within ${P2P_STARTUP_TIMEOUT}s" >&2
  echo "       tail of instance_a.log:" >&2
  tail -30 "$ROOT/instance_a.log" >&2
  exit 4
fi
echo "instance A up $(date)"

echo "launching instance B (port 8002, role=decode)..."
P2P_PORT=8002 P2P_ROLE=decode  bash "$SCRIPT_DIR/launch-instance.sh" \
  > "$ROOT/instance_b.log" 2>&1 &
disown
if ! wait_for_models 8002 "$P2P_STARTUP_TIMEOUT"; then
  echo "FAIL: instance B did not respond on /v1/models within ${P2P_STARTUP_TIMEOUT}s" >&2
  echo "       tail of instance_b.log:" >&2
  tail -30 "$ROOT/instance_b.log" >&2
  exit 4
fi
echo "BOTH UP $(date)"

# ----------------------------------------------------------------------
# 3. Discover instance IDs from hub registry.
#    /v1/peers (no /instance/{id}) returns the full peer list.
# ----------------------------------------------------------------------
sleep 3  # let registrations settle
PEERS=$(curl -sS http://127.0.0.1:8337/v1/peers)
# Hub keys peers by instance_id (UUID). Both instances are CD-registered
# (prefill + decode roles), so we can use the CD instances endpoint to
# pick them out by role.
CD=$(curl -sS http://127.0.0.1:8337/v1/features/conditional-disagg/instances)
INSTANCE_A=$(echo "$CD" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["prefill"][0])')
INSTANCE_B=$(echo "$CD" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["decode"][0])')
echo "INSTANCE_A=$INSTANCE_A  (port 8000, role=prefill)"
echo "INSTANCE_B=$INSTANCE_B  (port 8002, role=decode)"

# ----------------------------------------------------------------------
# 3.5. Pre-warm velo peer relationships via the hub.
#
#      pull_from_session opens a velo connection from B to A. Velo's
#      streaming transport requires `messenger.register_peer(peer)`
#      ahead of time. With the hub wired as velo's PeerDiscovery (via
#      `seed_leader_builder_with_hub_discovery` in the leader runtime),
#      this is just a POST to `core/register_leader` — the leader looks
#      the peer up via the hub and registers it on its own velo.
# ----------------------------------------------------------------------
echo "=== pre-warm velo peer relationships via hub discovery ==="
for src_dst in "$INSTANCE_B:$INSTANCE_A" "$INSTANCE_A:$INSTANCE_B"; do
  src=${src_dst%:*}
  dst=${src_dst#*:}
  resp=$(curl -m 30 -sS -X POST \
    "http://127.0.0.1:8337/v1/instances/$src/control/core/register_leader" \
    -H 'content-type: application/json' \
    -d "{\"instance_id\":\"$dst\"}")
  echo "  $src learns peer $dst -> $resp"
done

# ----------------------------------------------------------------------
# 4. R1 — issue prompt to instance A.
#    Prompt needs to produce at least P2P_BLOCKS full G2 blocks. Each
#    block holds block_size=16 tokens; so we need ≥ P2P_BLOCKS*16 tokens
#    of ISL. Padding to ~320 tokens to clear 16 full blocks comfortably.
# ----------------------------------------------------------------------
P1="$(python3 -c '
words = [
  "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
  "while", "the", "sun", "sets", "behind", "the", "ancient", "mountains",
  "casting", "long", "shadows", "across", "the", "meadow", "where", "wildflowers",
  "bloom", "in", "vibrant", "colors", "of", "red", "orange", "yellow", "and",
  "purple", "as", "the", "evening", "breeze", "carries", "the", "scent", "of",
  "pine", "and", "wet", "earth", "through", "the", "valley", "below", "where",
  "a", "stream", "winds", "its", "way", "between", "moss", "covered", "stones",
  "and", "fallen", "logs", "creating", "small", "pools", "of", "clear", "water",
  "that", "reflect", "the", "fading", "light", "of", "the", "departing", "day",
  "and", "as", "night", "begins", "to", "fall", "the", "stars", "appear", "one",
  "by", "one", "in", "the", "deepening", "blue", "sky"
] * 4
print(" ".join(words))
')"

echo "=== R1: POST /v1/completions to instance A (port 8000) ==="
R1=$(python3 -c '
import json, sys
print(json.dumps({"model":"Qwen/Qwen3-0.6B","prompt":sys.argv[1],"max_tokens":8,"temperature":0}))
' "$P1" \
  | curl -m 120 -sS -X POST http://127.0.0.1:8000/v1/completions \
      -H "Content-Type: application/json" -d @-)
echo "$R1" | head -c 300; echo

# ----------------------------------------------------------------------
# 5. Wait for offload and scrape audit log on A.
#    The kvbm_engine offload pipeline emits
#      kvbm_audit: event="offload_register_complete" src=... dst=...G2... sequence_hashes=...
#    on every batch register into G2. The full sequence_hashes line is a
#    comma-separated list of stringified PositionalLineageHash values.
# ----------------------------------------------------------------------
sleep 3  # let offload finish + audit lines flush

# Find audit events where dst is G2. Strip ANSI escapes first because
# tracing_subscriber colorizes by default and that interleaves bytes
# between literal field characters, breaking naive greps.
HASH_LINE=$(sed 's/\x1b\[[0-9;]*m//g' "$ROOT/instance_a.log" \
            | grep -a 'kvbm_audit.*event="offload_register_complete"' \
            | grep -aE 'dst="[^"]*::G2"' \
            | tail -1 || true)

if [ -z "$HASH_LINE" ]; then
  echo "FAIL: no offload_register_complete audit event with dst=G2 found in instance_a.log" >&2
  echo "       (this means R1 didn't trigger G2 offload, or the audit emit didn't fire)" >&2
  echo "       grep heads up:" >&2
  sed 's/\x1b\[[0-9;]*m//g' "$ROOT/instance_a.log" | grep -a 'kvbm_audit' | head -10 >&2 || true
  exit 2
fi
echo "audit line: $HASH_LINE" | head -c 500; echo

# Extract sequence_hashes_hex — 32-hex-char u128s separated by commas
# (audit emit at lib/kvbm-engine/src/offload/pipeline.rs uses
# {:032x} format). The field is quoted because tracing renders String
# fields via Debug by default.
HASHES_RAW=$(echo "$HASH_LINE" | sed -nE 's/.*sequence_hashes_hex="([^"]+)".*/\1/p')
if [ -z "$HASHES_RAW" ]; then
  echo "FAIL: could not parse sequence_hashes_hex field from audit line" >&2
  exit 2
fi
HASH_COUNT=$(echo "$HASHES_RAW" | tr ',' '\n' | wc -l | tr -d ' ')
echo "extracted $HASH_COUNT hash(es) from R1 G2 offload"

if [ "$HASH_COUNT" -lt 1 ]; then
  echo "FAIL: zero hashes extracted" >&2
  exit 2
fi

# Build JSON payload of hashes for open_session. Each hex string is the
# big-endian u128 representation; the hub deserializes SequenceHash from
# a 16-element u8 sequence (see lib/tokens/src/lib.rs:140 — u128::from_be_bytes).
# Decode hex → 16 bytes → JSON array of u8.
HASHES_JSON=$(echo "$HASHES_RAW" | python3 -c '
import json, sys
parts = [p.strip() for p in sys.stdin.read().split(",") if p.strip()]
arrs = []
for p in parts:
    if len(p) != 32:
        raise SystemExit(f"bad hash hex length: {len(p)} (want 32) in {p!r}")
    arrs.append(list(bytes.fromhex(p)))
print(json.dumps(arrs))
')
echo "hashes JSON sample (first hash): $(echo "$HASHES_JSON" | python3 -c 'import json,sys;print(json.load(sys.stdin)[0])')"

# ----------------------------------------------------------------------
# 6. open_session on A.  find_mode=sync so we get the matched set inline.
# ----------------------------------------------------------------------
echo "=== open_session on A ==="
OPEN_REQ=$(python3 -c '
import json, sys
print(json.dumps({
  "sequence_hashes": json.loads(sys.argv[1]),
  "search_mode": "prefix",
  "find_mode":   "sync"
}))
' "$HASHES_JSON")
OPEN_RESP=$(curl -m 30 -sS -X POST \
  "http://127.0.0.1:8337/v1/instances/$INSTANCE_A/control/transfer/open_session" \
  -H 'content-type: application/json' \
  -d "$OPEN_REQ")
echo "open_session response: $(echo "$OPEN_RESP" | head -c 500)"

RESULT=$(echo "$OPEN_RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("result","?"))')
if [ "$RESULT" = "no_blocks_found" ]; then
  echo "FAIL: open_session reports no_blocks_found — A's G2 doesn't have the hashes we extracted from its own audit log" >&2
  exit 3
fi

# Extract capability + committed list.
SESSION_ID=$(echo "$OPEN_RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["capability"]["session_id"])')
COMMITTED_COUNT=$(echo "$OPEN_RESP" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d.get("committed", [])))')
echo "session_id=$SESSION_ID  committed=$COMMITTED_COUNT block(s)"

# ----------------------------------------------------------------------
# 7. pull_from_session on B.  Pass capability.endpoint through so B
#    doesn't need a separate hub lookup.
# ----------------------------------------------------------------------
echo "=== pull_from_session on B ==="
PULL_REQ=$(echo "$OPEN_RESP" | python3 -c '
import json, sys
d = json.load(sys.stdin)
cap = d["capability"]
print(json.dumps({
  "session_id":         cap["session_id"],
  "source_instance_id": cap["instance_id"],
  "endpoint":           cap["endpoint"]
}))
')
PULL_RESP=$(curl -m 120 -sS -X POST \
  "http://127.0.0.1:8337/v1/instances/$INSTANCE_B/control/transfer/pull_from_session" \
  -H 'content-type: application/json' \
  -d "$PULL_REQ")
echo "pull_from_session response: $(echo "$PULL_RESP" | head -c 500)"

PULLED_COUNT=$(echo "$PULL_RESP" | python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("pulled", [])))')
echo "pulled $PULLED_COUNT block(s) into B's G2"

# Hard assertion: the pull must have transferred something. An empty
# `pulled` list means the session failed (handshake, attach, or transport)
# even if curl returned 200.
if [ "$PULLED_COUNT" -lt 1 ]; then
  echo "FAIL: pull_from_session returned 0 blocks (expected at least 1)" >&2
  echo "       full pull response:" >&2
  echo "$PULL_RESP" >&2
  exit 4
fi

# Stricter assertion: every committed block should have been pulled. A
# partial pull means the session ended early or the puller dropped blocks.
if [ "$PULLED_COUNT" -ne "$COMMITTED_COUNT" ]; then
  echo "FAIL: partial pull — pulled=$PULLED_COUNT but committed=$COMMITTED_COUNT" >&2
  exit 4
fi

# ----------------------------------------------------------------------
# 8. close_session on A.
# ----------------------------------------------------------------------
echo "=== close_session on A ==="
CLOSE_RESP=$(curl -m 30 -sS -X POST \
  "http://127.0.0.1:8337/v1/instances/$INSTANCE_A/control/transfer/close_session" \
  -H 'content-type: application/json' \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"session_id": sys.argv[1]}))' "$SESSION_ID")")
echo "close_session response: $CLOSE_RESP"

# ----------------------------------------------------------------------
# 9. R2 — same prompt to instance B. Expect G2 cache hit.
# ----------------------------------------------------------------------
echo "=== R2: POST /v1/completions to instance B (port 8002) ==="
R2=$(python3 -c '
import json, sys
print(json.dumps({"model":"Qwen/Qwen3-0.6B","prompt":sys.argv[1],"max_tokens":8,"temperature":0}))
' "$P1" \
  | curl -m 120 -sS -X POST http://127.0.0.1:8002/v1/completions \
      -H "Content-Type: application/json" -d @-)
echo "$R2" | head -c 300; echo

sleep 3

# ----------------------------------------------------------------------
# 10. Validation report.
# ----------------------------------------------------------------------
echo
echo "================================================================"
echo "  P2P smoke validation report  (trace at $ROOT/trace.html)"
echo "================================================================"
echo
echo "-- R1 audit: G1→G2 offload register on A --"
sed 's/\x1b\[[0-9;]*m//g' "$ROOT/instance_a.log" \
  | grep -a 'kvbm_audit.*event="offload_register_complete"' | head -5
echo
echo "-- pull confirmation: B's audit for incoming G2 register --"
sed 's/\x1b\[[0-9;]*m//g' "$ROOT/instance_b.log" \
  | grep -a 'kvbm_audit.*event="offload_register_complete"' | head -5
echo
echo "-- R2 cache hit on B (expect Host rate > 0%) --"
R2_CACHE_HIT_LINE=$(sed 's/\x1b\[[0-9;]*m//g' "$ROOT/instance_b.log" \
  | grep -aE "Cache Hit Rates - Host: " | tail -1 || true)
echo "$R2_CACHE_HIT_LINE"

# Hard assertion: R2 must show a non-zero Host (G2) cache hit. The
# whole point of the smoke is to prove the pulled blocks are USABLE —
# not just present. Parse `Host: XX.X%` and require > 0.
HOST_HIT_PCT=$(echo "$R2_CACHE_HIT_LINE" \
  | sed -nE 's/.*Host:[[:space:]]*([0-9.]+)%.*/\1/p' \
  | head -1)
if [ -z "$HOST_HIT_PCT" ]; then
  echo "FAIL: could not parse R2 Host cache hit rate from instance_b.log" >&2
  echo "       last 20 Cache Hit Rates lines:" >&2
  sed 's/\x1b\[[0-9;]*m//g' "$ROOT/instance_b.log" \
    | grep -aE "Cache Hit Rates" | tail -20 >&2
  exit 5
fi
# bash arithmetic only handles integers; compare via awk.
if awk -v p="$HOST_HIT_PCT" 'BEGIN { exit (p+0 > 0) ? 0 : 1 }'; then
  echo "  -> Host G2 hit rate = ${HOST_HIT_PCT}% (assertion passed)"
else
  echo "FAIL: R2 Host G2 hit rate was 0% — pulled blocks did not serve R2" >&2
  echo "       this means the blocks are in B's G2 metadata but the actual" >&2
  echo "       KV bytes are absent or unusable for inference reuse" >&2
  exit 5
fi

echo
echo "-- ERRORs across logs --"
for s in hub instance_a instance_b; do
  cnt=$(grep -aE 'ERROR' "$ROOT/$s.log" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' \
        | grep -v "kvbm_audit\|UCX\|invalid configuration\|kernel_config" | wc -l)
  echo "  $s.log: $cnt error lines"
done

# Render HTML.
if [ -x "$SKILL_TRACE/p2p-trace.py" ]; then
    python3 "$SKILL_TRACE/p2p-trace.py" "$ROOT"
    echo
    echo "Open: file://$ROOT/trace.html"
fi

echo
echo "p2p-smoke PASS: pulled=$PULLED_COUNT (of $COMMITTED_COUNT committed, $HASH_COUNT requested); R2 Host hit=${HOST_HIT_PCT}%"
