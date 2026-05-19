#!/bin/bash
# kvrouter-smoke.sh — exercise the embedded KV router in dynamo.frontend
# with two dynamo.vllm workers under the same component endpoint.
#
# Flow:
#   teardown stale          (vllm, EngineCore, dynamo, kvbm_hub)
#   validate etcd + nats    (must already be running)
#   mint experiment dir     (reuses disagg-bringup/new-experiment.sh)
#   launch worker A         (port 8001, kv-events tcp://*:5557)
#   wait for A in etcd
#   launch worker B         (port 8002, kv-events tcp://*:5567)
#   wait for B in etcd
#   launch frontend         (--router-mode kv, --http-port 8080)
#   wait for /v1/models
#   R1 → /v1/completions    (cold prompt; record which worker handled it)
#   sleep — let G1 events flush to NATS, router ingests
#   R2 → /v1/completions    (same prompt; record which worker handled it)
#   assert R1_worker == R2_worker
#   assert R2.usage.prompt_tokens_details.cached_tokens > 0.5 * prompt_tokens
#
# Usage: bash kvrouter-smoke.sh [logs_dir]
#   If logs_dir is omitted, mints $KVBM_EXPERIMENTS_DIR/<ts>-kvrouter-smoke/.
#
# Env:
#   KVBM_REPO              (default: worktree root inferred from script location)
#   KVBM_VENV              (default: ryan-velo-messenger/.sandbox)
#   KVR_GMU                (default: 0.15)
#   KVR_PROMPT_WORDS_MULT  (default: 4) — prompt length multiplier
set -eu

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DYNAMO=${KVBM_REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
SKILL_BRINGUP=$DYNAMO/.claude/skills/disagg-bringup
LABEL=${KVBM_EXPERIMENT_LABEL:-kvrouter-smoke}

export KVBM_VENV=${KVBM_VENV:-/home/ryan/repos/dynamo-workspaces/ryan-velo-messenger/.sandbox}
export KVR_GMU=${KVR_GMU:-0.15}

ROOT=${1:-$(bash "$SKILL_BRINGUP/new-experiment.sh" "$LABEL")}
echo "EXP=$ROOT"

# ----------------------------------------------------------------------
# 0. Validate dependencies.
# ----------------------------------------------------------------------
echo "=== validate etcd + nats ==="
if ! curl -fsS http://127.0.0.1:2379/health >/dev/null 2>&1; then
  echo "FAIL: etcd not reachable at http://127.0.0.1:2379/health" >&2
  exit 1
fi
echo "  etcd OK"

# NATS doesn't expose HTTP healthz here, so use a 1-byte TCP probe.
if ! timeout 3 bash -c "exec 3<>/dev/tcp/127.0.0.1/4222" >/dev/null 2>&1; then
  echo "FAIL: NATS not reachable at tcp://127.0.0.1:4222" >&2
  exit 1
fi
echo "  nats OK"

# ----------------------------------------------------------------------
# 1. Teardown stale procs (the GB10 reports memory as Not Supported,
#    so we kill by process name rather than by GPU process list).
#
#    Dynamo registers each worker into etcd with a 10s TTL lease (see
#    lib/runtime/src/transports/etcd.rs:86). When a worker dies, the
#    keep-alive stops and etcd auto-deletes the keys. So we don't
#    need to delete any etcd keys ourselves — we just need to wait
#    long enough for any prior leases to expire before launching new
#    workers. 12s > the 10s TTL gives a safety margin.
# ----------------------------------------------------------------------
echo "=== teardown stale ==="
pkill -f "vllm.entrypoints.openai" 2>/dev/null || true
pkill -f "dynamo.vllm" 2>/dev/null || true
pkill -f "dynamo.frontend" 2>/dev/null || true
pkill -9 -f kvbm_hub 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true

# Belt-and-suspenders: explicitly delete any v1/instances/<namespace>/
# keys still hanging around. Dynamo's etcd key prefix is constant
# `v1/instances` (see lib/runtime/src/discovery/kv_store.rs:19). Prefix
# deletion in etcd v3 uses range_end == prefix with the last byte
# incremented — for `v1/instances/dynamo/` (trailing 0x2f) the range
# end is `v1/instances/dynamo0` (last byte 0x30 = '/'+1).
"$KVBM_VENV/bin/python3" - <<'PY' 2>/dev/null || true
import base64, json, urllib.request
prefix = b"v1/instances/dynamo/"
range_end = bytearray(prefix); range_end[-1] += 1
body = json.dumps({
  "key":       base64.b64encode(prefix).decode(),
  "range_end": base64.b64encode(bytes(range_end)).decode(),
}).encode()
req = urllib.request.Request(
  "http://127.0.0.1:2379/v3/kv/deleterange",
  data=body, headers={"content-type":"application/json"})
with urllib.request.urlopen(req, timeout=3) as r:
    d = json.load(r); print(f"  etcd: cleared {d.get('deleted', 0)} stale key(s)")
PY

# Wait > TTL for any leases we didn't actively delete.
sleep 12

# ----------------------------------------------------------------------
# 3. Launch worker A.
# ----------------------------------------------------------------------
RUST_LOG=${RUST_LOG:-info,kvbm_connector=info,kvbm_engine=info,kvbm_audit=info}
export RUST_LOG
export PATH="$KVBM_VENV/bin:$PATH"

KVR_STARTUP_TIMEOUT=${KVR_STARTUP_TIMEOUT:-300}

wait_for_workers_in_etcd() {
  local expected=$1
  local timeout=$2
  local deadline=$(( $(date +%s) + timeout ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    # etcd v3 range key prefix "instances/dynamo" (we lookup by the
    # rust runtime registry key; this just counts how many instance
    # records exist under the namespace).
    count=$("$KVBM_VENV/bin/python3" - <<'PY' 2>/dev/null || echo 0
import json, sys
try:
    import urllib.request, base64
    # Etcd prefix that Dynamo writes for component instance records:
    # `v1/instances/<namespace>/<component>/<endpoint>/<lease_id>`
    # see lib/runtime/src/discovery/kv_store.rs:19.
    prefix = b"v1/instances/dynamo/backend/"
    range_end = bytearray(prefix); range_end[-1] += 1
    body = json.dumps({
      "key":       base64.b64encode(prefix).decode(),
      "range_end": base64.b64encode(bytes(range_end)).decode(),
      "keys_only": True,
    }).encode()
    req = urllib.request.Request(
      "http://127.0.0.1:2379/v3/kv/range",
      data=body, headers={"content-type":"application/json"})
    with urllib.request.urlopen(req, timeout=3) as r:
        d = json.load(r)
        kvs = d.get("kvs", [])
        print(len(kvs))
except Exception:
    print(0)
PY
)
    if [ "$count" -ge "$expected" ]; then
      return 0
    fi
    sleep 2
  done
  return 1
}

echo "=== launch worker A (kv-events=5557, kvbm-pub=56001 → consolidator=57001) ==="
KVR_KV_EVENT_PORT=5557 KVR_WORKER_LABEL=worker_a \
  DYN_KVBM_LEADER_ZMQ_PUB_PORT=56001 \
  bash "$SCRIPT_DIR/launch-worker.sh" \
  > "$ROOT/worker_a.log" 2>&1 &
WORKER_A_PID=$!
disown
echo "  worker A pid=$WORKER_A_PID"

if ! wait_for_workers_in_etcd 1 "$KVR_STARTUP_TIMEOUT"; then
  echo "FAIL: worker A did not register in etcd within ${KVR_STARTUP_TIMEOUT}s" >&2
  echo "       tail of worker_a.log:" >&2
  tail -40 "$ROOT/worker_a.log" >&2
  exit 4
fi
echo "  worker A registered"

# ----------------------------------------------------------------------
# 4. Launch worker B.
# ----------------------------------------------------------------------
echo "=== launch worker B (kv-events=5567, kvbm-pub=56011 → consolidator=57011) ==="
KVR_KV_EVENT_PORT=5567 KVR_WORKER_LABEL=worker_b \
  DYN_KVBM_LEADER_ZMQ_PUB_PORT=56011 \
  bash "$SCRIPT_DIR/launch-worker.sh" \
  > "$ROOT/worker_b.log" 2>&1 &
WORKER_B_PID=$!
disown
echo "  worker B pid=$WORKER_B_PID"

if ! wait_for_workers_in_etcd 2 "$KVR_STARTUP_TIMEOUT"; then
  echo "FAIL: worker B did not register in etcd within ${KVR_STARTUP_TIMEOUT}s" >&2
  echo "       tail of worker_b.log:" >&2
  tail -40 "$ROOT/worker_b.log" >&2
  exit 4
fi
echo "  worker B registered"

# ----------------------------------------------------------------------
# 5. Launch frontend.
# ----------------------------------------------------------------------
echo "=== launch frontend (HTTP=8080) ==="
KVR_HTTP_PORT=8080 bash "$SCRIPT_DIR/launch-frontend.sh" \
  > "$ROOT/frontend.log" 2>&1 &
FRONTEND_PID=$!
disown
echo "  frontend pid=$FRONTEND_PID"

# Wait for /v1/models.
DEADLINE=$(( $(date +%s) + 120 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if curl -fsS http://127.0.0.1:8080/v1/models 2>/dev/null | grep -q "Qwen3"; then
    break
  fi
  sleep 2
done
if ! curl -fsS http://127.0.0.1:8080/v1/models 2>/dev/null | grep -q "Qwen3"; then
  echo "FAIL: frontend /v1/models did not return Qwen3-0.6B within 120s" >&2
  tail -30 "$ROOT/frontend.log" >&2
  exit 5
fi
echo "  frontend up"

# ----------------------------------------------------------------------
# 6. R1 — cold prompt.
# ----------------------------------------------------------------------
WORDS_MULT=${KVR_PROMPT_WORDS_MULT:-4}
PROMPT="$("$KVBM_VENV/bin/python3" - <<PY
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
] * ${WORDS_MULT}
print(" ".join(words))
PY
)"

req_payload() {
  "$KVBM_VENV/bin/python3" - <<PY
import json,sys
print(json.dumps({
  "model": "Qwen/Qwen3-0.6B",
  "prompt": $(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$PROMPT"),
  "max_tokens": 8,
  "temperature": 0,
  "stream": False
}))
PY
}

echo "=== R1 → POST /v1/completions ==="
R1=$(echo "$(req_payload)" \
  | curl -m 120 -sS -X POST http://127.0.0.1:8080/v1/completions \
      -H "Content-Type: application/json" -d @-)
echo "$R1" | head -c 500; echo

R1_PROMPT_TOK=$(echo "$R1" | "$KVBM_VENV/bin/python3" -c 'import json,sys;d=json.load(sys.stdin);print(d.get("usage",{}).get("prompt_tokens","?"))')
R1_CACHED=$(echo "$R1" | "$KVBM_VENV/bin/python3" -c 'import json,sys;d=json.load(sys.stdin);u=d.get("usage",{});pd=u.get("prompt_tokens_details") or {};print(pd.get("cached_tokens",0))')
echo "  R1: prompt_tokens=$R1_PROMPT_TOK cached_tokens=$R1_CACHED"

# Wait for G1 store events to flush to NATS and the indexer.
sleep 4

# ----------------------------------------------------------------------
# 7. R2 — same prompt; should route to same worker.
# ----------------------------------------------------------------------
echo "=== R2 → POST /v1/completions (same prompt) ==="
R2=$(echo "$(req_payload)" \
  | curl -m 120 -sS -X POST http://127.0.0.1:8080/v1/completions \
      -H "Content-Type: application/json" -d @-)
echo "$R2" | head -c 500; echo

R2_PROMPT_TOK=$(echo "$R2" | "$KVBM_VENV/bin/python3" -c 'import json,sys;d=json.load(sys.stdin);print(d.get("usage",{}).get("prompt_tokens","?"))')
R2_CACHED=$(echo "$R2" | "$KVBM_VENV/bin/python3" -c 'import json,sys;d=json.load(sys.stdin);u=d.get("usage",{});pd=u.get("prompt_tokens_details") or {};print(pd.get("cached_tokens",0))')
echo "  R2: prompt_tokens=$R2_PROMPT_TOK cached_tokens=$R2_CACHED"

# ----------------------------------------------------------------------
# 8. Determine which worker handled each request.
#    The KV router emits a structured log line per routing decision
#    via `dynamo_kv_router::scheduling::selector`:
#
#      Selected worker: worker_type=decode, worker_id=NNN dp_rank=0, ...
#                                                   ^ deterministic
#
#    Each call to /v1/completions produces exactly one such line. The
#    response includes an OpenAI completion id; the frontend tagged the
#    routing log line with `request_id=UUID` from its own http-request
#    span, which is the frontend's per-request UUID — NOT the same as
#    the OpenAI completion id (the frontend issues its own id).
#
#    So we can't match by id. Instead we use the obvious property: the
#    smoke is single-threaded and serial — R1's `Selected worker` line
#    appears before R2's. Take the first two `Selected worker:` lines
#    after we launched the frontend; they're R1 and R2.
#
#    (Per tests/CLAUDE.md guidance: "If a router-internal fact is only
#    exposed as a structured tracing event, parse it through a shared
#    test helper." This is a smoke; we parse inline. If the smoke
#    becomes a fixture, factor the parse out.)
# ----------------------------------------------------------------------
parse_router_pick_field() {
  # Extract one field from each `Selected worker:` line, in order.
  #   $1 — sed-friendly capture pattern (e.g. 'worker_id=([0-9]+)')
  # Output: one capture per line. ANSI-strip first.
  local pat="$1"
  sed 's/\x1b\[[0-9;]*m//g' "$ROOT/frontend.log" 2>/dev/null \
    | grep -aE 'Selected worker:' \
    | sed -nE "s/.*${pat}.*/\\1/p"
}

R1_WORKER=$(parse_router_pick_field 'worker_id=([0-9]+)' | sed -n '1p')
R2_WORKER=$(parse_router_pick_field 'worker_id=([0-9]+)' | sed -n '2p')
R1_LOGIT=$(parse_router_pick_field 'logit:[[:space:]]*([0-9.]+)'  | sed -n '1p')
R2_LOGIT=$(parse_router_pick_field 'logit:[[:space:]]*([0-9.]+)'  | sed -n '2p')

if [ -z "$R1_WORKER" ] || [ -z "$R2_WORKER" ] || [ -z "$R1_LOGIT" ] || [ -z "$R2_LOGIT" ]; then
  echo "FAIL: could not parse 'Selected worker:' lines from frontend.log" >&2
  echo "       (KV router may not be emitting routing decisions — check that --router-mode kv is active)" >&2
  echo "       last 40 frontend.log lines:" >&2
  tail -40 "$ROOT/frontend.log" >&2
  exit 7
fi

echo "  R1 routed to worker_id=$R1_WORKER  logit=$R1_LOGIT"
echo "  R2 routed to worker_id=$R2_WORKER  logit=$R2_LOGIT"

# ----------------------------------------------------------------------
# 9. Validation report + assertions.
# ----------------------------------------------------------------------
echo
echo "================================================================"
echo "  kvrouter-smoke validation report  ($ROOT)"
echo "================================================================"
echo
echo "-- routing decisions (from frontend KV router) --"
echo "  R1 → worker_id=$R1_WORKER  logit=$R1_LOGIT"
echo "  R2 → worker_id=$R2_WORKER  logit=$R2_LOGIT"
echo "-- R1 usage --"
echo "  prompt_tokens=$R1_PROMPT_TOK cached_tokens=$R1_CACHED"
echo "-- R2 usage --"
echo "  prompt_tokens=$R2_PROMPT_TOK cached_tokens=$R2_CACHED"
echo

# --------------------------------------------------------------------
# Assertion A (NECESSARY): R1 and R2 routed to the SAME worker.
#
# Necessary but not sufficient on its own — with no KV signal the
# router can still deterministically pick the same worker for both
# requests via tiebreakers (worker-id sort, load tracker) even
# without any kv-aware logic firing. Pair with Assertion B.
# --------------------------------------------------------------------
if [ "$R1_WORKER" != "$R2_WORKER" ]; then
  echo "FAIL: KV-aware routing did NOT keep R1 and R2 on the same worker" >&2
  echo "       R1 worker_id=$R1_WORKER, R2 worker_id=$R2_WORKER" >&2
  echo "       diagnostic — frontend KV router decision logs:" >&2
  sed 's/\x1b\[[0-9;]*m//g' "$ROOT/frontend.log" 2>/dev/null \
    | grep -aE 'Selected worker|host_pinned|disk blocks' | tail -10 >&2
  echo "       diagnostic — worker A KV-event publisher info:" >&2
  grep -aE 'KV event publisher|kv_events|consolidator|publishing' "$ROOT/worker_a.log" 2>/dev/null | head -5 >&2
  exit 7
fi
echo "  worker-co-location assertion PASSED (R1 == R2 == worker_id=$R1_WORKER)"

# --------------------------------------------------------------------
# Assertion B (PRIMARY): R2 was routed by KV-aware logic, not by a
#                        coincidental tiebreaker.
#
# The router emits a per-worker logit and picks the lowest one
# (lib/kv-router/src/scheduling/selector.rs:302-309). Logit =
# prefill_cost + decode_cost; overlap with the worker's cached prefix
# subtracts blocks from prefill_cost via overlap_score_credit
# (selector.rs:107-168). So:
#
#   R2 logit < R1 logit  ⇔  the router saw overlap blocks for R2
#                            and applied the discount  ⇔  KV-aware
#                            logic fired.
#
# If KV events never reached the indexer (consolidator dead, NATS
# subject mismatch, publisher pointed at wrong port, ...), both R1
# and R2 see zero overlap and have identical logits, and a
# deterministic tiebreaker can still produce R1_WORKER == R2_WORKER
# without any kv-aware logic. That false-positive is exactly what
# this assertion blocks.
# --------------------------------------------------------------------
LOGIT_DROP=$(awk -v a="$R1_LOGIT" -v b="$R2_LOGIT" 'BEGIN { print a - b }')
# bash arithmetic can't do floats; use awk for the comparison.
if ! awk -v a="$R1_LOGIT" -v b="$R2_LOGIT" 'BEGIN { exit (b < a) ? 0 : 1 }'; then
  echo "FAIL: R2 logit ($R2_LOGIT) is NOT lower than R1 logit ($R1_LOGIT)" >&2
  echo "       The router applied no KV-overlap credit on R2 — KV-aware" >&2
  echo "       routing did not fire. R1==R2 alone is not proof; the same" >&2
  echo "       outcome can happen via deterministic tiebreakers when zero" >&2
  echo "       KV events reach the indexer." >&2
  echo "       diagnostic — last 5 router decisions:" >&2
  sed 's/\x1b\[[0-9;]*m//g' "$ROOT/frontend.log" 2>/dev/null \
    | grep -aE 'Selected worker' | tail -5 >&2
  exit 9
fi
echo "  kv-aware-routing assertion PASSED (R2 logit $R2_LOGIT < R1 logit $R1_LOGIT, drop=$LOGIT_DROP)"

# --------------------------------------------------------------------
# Assertion C (CORROBORATING): the chosen worker's prefix cache
# actually served R2's prefix.
#
# vLLM's OpenAI usage object includes `prompt_tokens_details.cached_tokens`
# when prefix caching hits. Independent of any router-internal log —
# this corroborates A+B from the OpenAI response surface. If A and B
# pass but C fails, something is wrong with prefix caching on the
# worker itself (eviction? feature off?).
# --------------------------------------------------------------------
if [ -z "$R2_CACHED" ] || [ "$R2_CACHED" = "?" ]; then
  echo "FAIL: R2 response has no usage.prompt_tokens_details.cached_tokens field" >&2
  echo "       (vLLM 0.16+ should populate this when prefix caching is enabled)" >&2
  exit 8
fi
HALF=$(( R2_PROMPT_TOK / 2 ))
if [ "$R2_CACHED" -le "$HALF" ]; then
  echo "FAIL: R2 cached_tokens=$R2_CACHED is <= half of prompt_tokens=$R2_PROMPT_TOK" >&2
  echo "       routing landed on the right worker (R1 == R2 == $R1_WORKER) but" >&2
  echo "       that worker's prefix cache did not serve R2's prefix" >&2
  exit 8
fi
echo "  prefix-cache-hit assertion PASSED (cached=$R2_CACHED of $R2_PROMPT_TOK prompt tokens)"

# --------------------------------------------------------------------
# Assertion D (CONSOLIDATOR INGRESS): both event streams reach the
#                                     consolidator.
#
# Stage 2 wired the v2 connector to spawn lib/kvbm-consolidator
# in-process. The consolidator multiplexes:
#   - vLLM ZMQ events  → ingress_zmq
#   - kvbm-engine EventsManager events → ingress_kvbm
# and publishes a unified ZMQ stream that KvEventPublisher subscribes
# to.  If either ingress source is silent the smoke is degenerate —
# the router still sees events (because the other ingress is wired)
# but we wouldn't be exercising the consolidator's dedup logic.
#
# We grep both worker logs combined (only one worker handles both
# requests; the other may be idle).  Both `event="ingress_zmq"` AND
# `event="ingress_kvbm"` must appear at least once.
# --------------------------------------------------------------------
ANSI_STRIP="sed s/\\x1b\\[[0-9;]*m//g"
INGRESS_ZMQ=$( (sed 's/\x1b\[[0-9;]*m//g' "$ROOT/worker_a.log" "$ROOT/worker_b.log" 2>/dev/null) | grep -caE 'kvbm_consolidator_audit.*event="ingress_zmq"' )
INGRESS_KVBM=$( (sed 's/\x1b\[[0-9;]*m//g' "$ROOT/worker_a.log" "$ROOT/worker_b.log" 2>/dev/null) | grep -caE 'kvbm_consolidator_audit.*event="ingress_kvbm"' )
EGRESS_COUNT=$( (sed 's/\x1b\[[0-9;]*m//g' "$ROOT/worker_a.log" "$ROOT/worker_b.log" 2>/dev/null) | grep -caE 'kvbm_consolidator_audit.*event="egress"' )
echo "-- consolidator audit counts (across both worker logs) --"
echo "  ingress_zmq=$INGRESS_ZMQ  ingress_kvbm=$INGRESS_KVBM  egress=$EGRESS_COUNT"

if [ "$INGRESS_ZMQ" -lt 1 ] || [ "$INGRESS_KVBM" -lt 1 ]; then
  echo "FAIL: consolidator did not observe both ingress streams" >&2
  echo "       ingress_zmq=$INGRESS_ZMQ, ingress_kvbm=$INGRESS_KVBM (both required > 0)" >&2
  echo "       diagnostic — most recent audit lines per worker:" >&2
  for s in worker_a worker_b; do
    echo "       === $s.log ===" >&2
    sed 's/\x1b\[[0-9;]*m//g' "$ROOT/$s.log" 2>/dev/null \
      | grep -aE 'kvbm_consolidator_audit|Starting in-process consolidator|failed to start in-process consolidator' \
      | tail -5 >&2
  done
  exit 10
fi
echo "  consolidator-ingress assertion PASSED (both vLLM ZMQ + KVBM EventsManager streams observed)"

# --------------------------------------------------------------------
# Assertion E (CONSOLIDATOR EGRESS): consolidator emits downstream.
#
# Egress events fire when the tracker drains a non-empty event set
# and `socket.send` returns Ok.  Zero egress means either (i) the
# tracker dedup'd everything to nothing (degenerate — would only
# happen if ingress was empty too, which D rules out), or (ii) the
# ZMQ socket bind failed (already caught by `failed to start
# in-process consolidator` errors at startup). One positive egress is
# sufficient to prove the publisher pipeline is alive.
# --------------------------------------------------------------------
if [ "$EGRESS_COUNT" -lt 1 ]; then
  echo "FAIL: consolidator did not emit any egress batches" >&2
  echo "       (ingress observed but tracker.drain_events → publisher pipeline is broken)" >&2
  exit 11
fi
echo "  consolidator-egress assertion PASSED (at least one batch published downstream)"

echo
echo "-- ERRORs across logs --"
for s in worker_a worker_b frontend; do
  cnt=$(sed 's/\x1b\[[0-9;]*m//g' "$ROOT/$s.log" 2>/dev/null \
        | grep -aE 'ERROR' \
        | grep -vE "kvbm_audit|UCX|invalid configuration|kernel_config|Failed to load CuPy" \
        | wc -l)
  echo "  $s.log: $cnt error lines"
done

echo
echo "kvrouter-smoke PASS:"
echo "  co-location:        R1==R2 worker_id=$R1_WORKER"
echo "  kv-aware:           R2 logit=$R2_LOGIT < R1 logit=$R1_LOGIT (drop=$LOGIT_DROP)"
echo "  cache-hit:          R2 cached $R2_CACHED of $R2_PROMPT_TOK tokens"
echo "  consolidator-in:    ingress_zmq=$INGRESS_ZMQ ingress_kvbm=$INGRESS_KVBM"
echo "  consolidator-out:   egress batches=$EGRESS_COUNT"
