#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# p2p-smoke.sh — exercise hub-mediated P2P G2 block transfer via the standalone
# P2P feature and the `kvbmctl p2p` verbs.
#
# Topology:  hub (--features p2p)  +  instance_a (8000)  +  instance_b (8002)
#            Both register Feature::P2P (no conditional-disagg role) — the
#            P2.5 standalone-p2p connector path. Each is a remote-controllable
#            block-copy peer.
#
# Flow:
#   R1 → instance_a            (warms G2 on A; indexer publishes G2 hashes)
#   query hub index            (discover A's offloaded hash_u128 values)
#   kvbmctl p2p pin   on A     (open_session over those hashes → session+endpoint)
#   kvbmctl p2p pull  A→B      (B registers A as a peer, then pulls into B's G2)
#   kvbmctl p2p unpin on A     (close_session)
#   verify hub index           (B is now an owner for the pulled blocks)
#   R2 → instance_b            (same prompt, chat streaming)
#
# Usage:  bash p2p-smoke.sh [logs_dir]
# Env:
#   KVBM_REPO            (default: worktree root inferred from script location)
#   P2P_HARDWARE_PROFILE (default: h100-a100; use spark-gb10 on a single GB10)
#   KVBM_VENV            (optional Python environment; no default local sandbox)
#   PYTHON_BIN           (optional explicit Python; otherwise resolved from
#                         KVBM_VENV, runtime image, or PATH)
#   P2P_CLEANUP_STALE_PROCESSES
#                         (default: 1; set 0 in shared sessions to skip broad
#                         stale-process and socket cleanup)
set -eu

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DYNAMO=${KVBM_REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
export KVBM_REPO=$DYNAMO
SKILL_BRINGUP=$DYNAMO/.claude/skills/disagg-bringup
HUB_BRINGUP=$DYNAMO/.claude/skills/kvbm-hub-bringup
SKILL_TRACE=$DYNAMO/.claude/skills/disagg-trace
KVBMCTL=${KVBM_KVBMCTL_BIN:-$DYNAMO/target/debug/kvbmctl}
LABEL=${KVBM_EXPERIMENT_LABEL:-p2p-smoke}
P2P_CLEANUP_STALE_PROCESSES=${P2P_CLEANUP_STALE_PROCESSES:-1}
. "$SKILL_BRINGUP/hardware-profiles.sh"
kvbm_apply_p2p_profile

case "$P2P_CLEANUP_STALE_PROCESSES" in
  1|true|yes|on) P2P_CLEANUP_STALE_PROCESSES=1 ;;
  0|false|no|off) P2P_CLEANUP_STALE_PROCESSES=0 ;;
  *) echo "P2P_CLEANUP_STALE_PROCESSES must be boolean, got $P2P_CLEANUP_STALE_PROCESSES" >&2; exit 2 ;;
esac

resolve_python() {
  if [ -n "${PYTHON_BIN:-}" ]; then
    [ -x "$PYTHON_BIN" ] || { echo "PYTHON_BIN is set but not executable: $PYTHON_BIN" >&2; return 2; }
  elif [ -n "${KVBM_VENV:-}" ] && [ -x "$KVBM_VENV/bin/python3" ]; then
    PYTHON_BIN="$KVBM_VENV/bin/python3"
  elif [ -n "${KVBM_VENV:-}" ] && [ -x "$KVBM_VENV/bin/python" ]; then
    PYTHON_BIN="$KVBM_VENV/bin/python"
  elif [ -x /opt/dynamo/venv/bin/python3 ]; then
    KVBM_VENV=/opt/dynamo/venv
    PYTHON_BIN=/opt/dynamo/venv/bin/python3
  elif [ -x /opt/dynamo/venv/bin/python ]; then
    KVBM_VENV=/opt/dynamo/venv
    PYTHON_BIN=/opt/dynamo/venv/bin/python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python3)
  else
    echo "No usable Python found; set PYTHON_BIN or KVBM_VENV" >&2
    return 2
  fi
  export PYTHON_BIN
  [ -z "${KVBM_VENV:-}" ] || export KVBM_VENV
}
resolve_python

HUB_DISC=${KVBM_HUB_DISCOVERY_PORT:-1337}
HUB_CTRL=${KVBM_HUB_CONTROL_PORT:-8337}
HUB1337="http://127.0.0.1:$HUB_DISC"
HUB8337="http://127.0.0.1:$HUB_CTRL"
export KVBMCTL_HUB="$HUB1337"

ROOT=${1:-$(bash "$SKILL_BRINGUP/new-experiment.sh" "$LABEL")}
echo "EXP=$ROOT"
echo "$ROOT" > /tmp/p2p-trace-current-exp
TRACE_GATE="$ROOT/trace-gate.env"

strip_ansi() { sed 's/\x1b\[[0-9;]*m//g'; }

cleanup_started_processes() {
  set +e
  local pid
  for pid in "${INSTANCE_A_PID:-}" "${INSTANCE_B_PID:-}" "${HUB_PID:-}"; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
  done
  sleep 1
  for pid in "${INSTANCE_A_PID:-}" "${INSTANCE_B_PID:-}" "${HUB_PID:-}"; do
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
  done
  set -e
}

cleanup_stale_processes() {
  if [ "$P2P_CLEANUP_STALE_PROCESSES" != "1" ]; then
    echo "[p2p-smoke] skip broad stale-process cleanup; smoke-owned PIDs are still cleaned on failure"
    return 0
  fi

  pkill -f "vllm.entrypoints.openai" 2>/dev/null || true
  pkill -9 -f kvbm_hub 2>/dev/null || true
  pkill -9 -f "EngineCore" 2>/dev/null || true
  sleep 3
  PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' || true)
  [ -n "$PIDS" ] && echo "$PIDS" | xargs -r kill -9 2>/dev/null || true
  rm -f /tmp/velo-kvbm-*.sock 2>/dev/null || true
  sleep 1
}

fail() {
  echo "FAIL: $1" >&2
  [ -n "${2:-}" ] && [ -f "$2" ] && { echo "--- tail $2 ---" >&2; tail -n 30 "$2" | strip_ansi >&2; }
  cleanup_started_processes
  cleanup_stale_processes
  exit 1
}

write_trace_gate() {
  local useful=$1 reason=$2 rendered=${3:-false} events=${4:-0} bytes=${5:-0}
  {
    echo "trace_useful=$useful"
    echo "trace_rendered=$rendered"
    echo "reason=$reason"
    echo "trace_audit_events=$events"
    echo "trace_html_bytes=$bytes"
    echo "p2p_pull_committed=${COMMITTED:-}"
    echo "p2p_pull_pulled=${PULLED:-}"
    echo "p2p_hash_count=${HASH_COUNT:-}"
  } > "$TRACE_GATE"
}

render_trace() {
  local trace_py="$SKILL_TRACE/p2p-trace.py"
  local trace_log="$ROOT/p2p-trace.log"
  local trace_file="$ROOT/trace.html"
  local trace_output events bytes
  [ -f "$trace_py" ] || { write_trace_gate false trace_renderer_missing false 0 0; fail "p2p trace renderer missing: $trace_py"; }
  if ! trace_output=$("$PYTHON_BIN" "$trace_py" "$ROOT" 2>"$trace_log"); then
    write_trace_gate false trace_render_failed false 0 0
    fail "p2p trace render failed" "$trace_log"
  fi
  printf '%s\n' "$trace_output" >> "$trace_log"
  [ -s "$trace_file" ] || { write_trace_gate false trace_html_missing_or_empty false 0 0; fail "p2p trace.html missing or empty" "$trace_log"; }
  events=$(printf '%s\n' "$trace_output" | sed -n 's/^\([0-9][0-9]*\) audit events.*/\1/p' | tail -n 1)
  events=${events:-0}
  bytes=$(wc -c < "$trace_file" | tr -d ' ')
  if [ "$events" -le 0 ]; then
    write_trace_gate false p2p_trace_has_no_audit_events true "$events" "$bytes"
    fail "p2p trace rendered but contained no audit events" "$trace_log"
  fi
  write_trace_gate true p2p_pin_pull_index_owner_validation_and_trace_rendered true "$events" "$bytes"
  echo "TRACE_DONE trace=$trace_file events=$events bytes=$bytes gate=$TRACE_GATE"
}

# --- 0. teardown stale ----------------------------------------------------
cleanup_stale_processes

# --- 1. hub (--features p2p,indexer) --------------------------------------
# p2p is under test; indexer is the harness hash-discovery mechanism (the
# EngineCore subprocess doesn't surface Rust kvbm_audit logs, so we read the
# holder's offloaded block hashes back from the hub index instead of scraping).
# Size the hub to the instances: block_size 16, max_seq_len = profile max_model_len.
KVBM_HUB_FEATURES=p2p,indexer \
KVBM_HUB_BLOCK_SIZE=16 \
KVBM_HUB_MAX_SEQ_LEN="${P2P_MAX_MODEL_LEN:-2048}" \
KVBM_HUB_G2_MEMORY_GIB="${P2P_CACHE_GB:-2}" \
  bash "$HUB_BRINGUP/start-hub.sh" "$ROOT/hub.log" &
HUB_PID=$!

HUB_READY_TIMEOUT=${KVBM_HUB_READY_TIMEOUT:-300}
hub_deadline=$(( $(date +%s) + HUB_READY_TIMEOUT ))
until curl -fsS -m 5 "$HUB8337/health" >/dev/null 2>&1; do
  kill -0 "$HUB_PID" 2>/dev/null || fail "hub exited before ready (build/bind failure)" "$ROOT/hub.log"
  [ "$(date +%s)" -ge "$hub_deadline" ] && fail "hub not ready after ${HUB_READY_TIMEOUT}s" "$ROOT/hub.log"
  sleep 2
done
echo "HUB UP (features=p2p)"

# --- 2. launch instances (standalone p2p, no role) ------------------------
RUST_LOG=${RUST_LOG:-info,kvbm_connector=debug,kvbm_engine=info,kvbm_audit=info}
export RUST_LOG
P2P_STARTUP_TIMEOUT=${P2P_STARTUP_TIMEOUT:-300}

wait_for_models() {
  local port=$1 timeout=$2 deadline
  deadline=$(( $(date +%s) + timeout ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && return 0
    sleep 3
  done
  return 1
}

# List the instance ids the hub currently knows (PeerInfo.instance_id). NOTE:
# the hub self-registers its own id here too, so discovery must exclude it.
list_instances() {
  curl -sS "$HUB8337/v1/instances" \
    | "$PYTHON_BIN" -c 'import json,sys; d=json.load(sys.stdin); print("\n".join(p["instance_id"] for p in d["instances"]))'
}

# The hub's own self-registered id(s), captured before any connector registers.
BASELINE_IDS=$(list_instances || true)
exclude_baseline() { grep -vxF "${BASELINE_IDS:-__none__}" || true; }

# Poll the registry for a connector id not in $1 (newline-separated excludes).
discover_new() {
  local excludes=$1 deadline id
  deadline=$(( $(date +%s) + 30 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    id=$(list_instances | grep -vxF "${BASELINE_IDS:-__none__}" \
          | { [ -n "$excludes" ] && grep -vxF "$excludes" || cat; } | head -1)
    [ -n "$id" ] && { echo "$id"; return 0; }
    sleep 2
  done
  return 1
}

echo "launching instance A (port 8000)..."
P2P_PORT=8000 P2P_CUDA_VISIBLE_DEVICES="$P2P_A_CUDA_VISIBLE_DEVICES" \
  bash "$SCRIPT_DIR/launch-instance.sh" > "$ROOT/instance_a.log" 2>&1 &
INSTANCE_A_PID=$!
disown
wait_for_models 8000 "$P2P_STARTUP_TIMEOUT" || fail "instance A not ready in ${P2P_STARTUP_TIMEOUT}s" "$ROOT/instance_a.log"
INSTANCE_A=$(discover_new "") || fail "instance A did not register with the hub" "$ROOT/instance_a.log"
echo "instance A up — INSTANCE_A=$INSTANCE_A"

echo "launching instance B (port 8002)..."
P2P_PORT=8002 P2P_CUDA_VISIBLE_DEVICES="$P2P_B_CUDA_VISIBLE_DEVICES" \
  bash "$SCRIPT_DIR/launch-instance.sh" > "$ROOT/instance_b.log" 2>&1 &
INSTANCE_B_PID=$!
disown
wait_for_models 8002 "$P2P_STARTUP_TIMEOUT" || fail "instance B not ready in ${P2P_STARTUP_TIMEOUT}s" "$ROOT/instance_b.log"
INSTANCE_B=$(discover_new "$INSTANCE_A") || fail "instance B did not register with the hub" "$ROOT/instance_b.log"
echo "BOTH UP — INSTANCE_B=$INSTANCE_B"

# --- 3. R1 → A (warm G2) --------------------------------------------------
P1="$("$PYTHON_BIN" -c '
words = ["The","quick","brown","fox","jumps","over","the","lazy","dog","while","the","sun","sets","behind","the","ancient","mountains","casting","long","shadows","across","the","meadow","where","wildflowers","bloom","in","vibrant","colors","of","red","orange","yellow","and","purple","as","the","evening","breeze","carries","the","scent","of","pine","and","wet","earth","through","the","valley","below","where","a","stream","winds","its","way","between","moss","covered","stones"] * 4
print(" ".join(words))
')"

run_chat_stream() {
  local port="$1" label="$2" prompt="$3"
  "$PYTHON_BIN" - "$port" "$P2P_MODEL" "$ROOT/${label}-chat-stream.jsonl" "$prompt" <<'PY'
import json, sys, time, urllib.request
port, model, stream_path, prompt = sys.argv[1:5]
payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
           "max_tokens": 8, "temperature": 0, "stream": True}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"content-type": "application/json"}, method="POST")
chunks = 0
with urllib.request.urlopen(req, timeout=120) as resp, open(stream_path, "w") as out:
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        out.write(line + "\n")
        if line.startswith("data: ") and line[6:] != "[DONE]":
            chunks += 1
print(json.dumps({"chunks": chunks}))
PY
}

echo "=== R1: chat stream to instance A (8000) ==="
run_chat_stream 8000 r1 "$P1" | head -c 200; echo
sleep 3  # let offload finish + audit flush

# --- 4. discover A's offloaded G2 block hashes via the hub index ----------
# After R1, A published its G2 blocks to the indexer. Read them back as decimal
# `hash_u128` (exactly what `kvbmctl p2p pin --hashes` wants). Only A has run
# inference so far, so every indexed block is A's. Poll until blocks appear
# (ZMQ ingest is async) or time out.
INDEXER="$HUB1337/v1/features/indexer"
collect_hashes() {
  "$PYTHON_BIN" - "$INDEXER" "${P2P_MAX_MODEL_LEN:-1024}" <<'PY'
import json, sys, urllib.request
base, max_len = sys.argv[1], int(sys.argv[2])
hashes = []
for pos in range(0, max_len // 16 + 1):
    try:
        with urllib.request.urlopen(f"{base}/hashes/by_position/{pos}", timeout=5) as r:
            d = json.load(r)
    except Exception:
        break
    ents = d.get("entries", [])
    if not ents:
        if pos > 0:   # contiguous prefix — stop at the first gap past position 0
            break
        continue
    for e in ents:
        hashes.append(e["hash_u128"])
seen, out = set(), []
for h in hashes:
    if h not in seen:
        seen.add(h); out.append(h)
print(",".join(out))
PY
}
DECIMAL_CSV=""
hash_deadline=$(( $(date +%s) + 30 ))
while [ "$(date +%s)" -lt "$hash_deadline" ]; do
  DECIMAL_CSV=$(collect_hashes)
  [ -n "$DECIMAL_CSV" ] && break
  sleep 2
done
[ -n "$DECIMAL_CSV" ] || fail "no blocks appeared in the hub index after R1 (A didn't offload/publish?)" "$ROOT/instance_a.log"
HASH_COUNT=$(echo "$DECIMAL_CSV" | tr ',' '\n' | grep -c .)
echo "discovered $HASH_COUNT G2 block hash(es) from A via the hub index"

# --- 5. pin on A (kvbmctl p2p pin = open_session sync/prefix) -------------
echo "=== kvbmctl p2p pin on A ==="
PIN_RESP=$("$KVBMCTL" p2p pin --instance-id "$INSTANCE_A" --hashes "$DECIMAL_CSV") \
  || fail "kvbmctl p2p pin failed"
echo "pin response: $(echo "$PIN_RESP" | head -c 400)"
SESSION_ID=$(echo "$PIN_RESP" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["session_id"])')
COMMITTED=$(echo "$PIN_RESP" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["committed_count"])')
ENDPOINT=$(echo "$PIN_RESP" | "$PYTHON_BIN" -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["endpoint"]))')
echo "session_id=$SESSION_ID committed=$COMMITTED"
[ "$COMMITTED" -ge 1 ] || fail "pin committed 0 blocks (A's G2 lacks the published index hashes?)"

# --- 6. pull A→B (kvbmctl p2p pull auto-registers the peer, then pulls) ----
echo "=== kvbmctl p2p pull A→B ==="
PULL_RESP=$("$KVBMCTL" p2p pull --from "$INSTANCE_A" --to "$INSTANCE_B" \
  --session-id "$SESSION_ID" --endpoint "$ENDPOINT") \
  || { "$KVBMCTL" p2p unpin --instance-id "$INSTANCE_A" --session-id "$SESSION_ID" >/dev/null 2>&1 || true; fail "kvbmctl p2p pull failed"; }
echo "pull response: $(echo "$PULL_RESP" | head -c 400)"
PULLED=$(echo "$PULL_RESP" | "$PYTHON_BIN" -c 'import json,sys; print(len(json.load(sys.stdin).get("pulled", [])))')
echo "pulled $PULLED block(s) into B's G2"

# --- 7. unpin on A (close_session) ----------------------------------------
echo "=== kvbmctl p2p unpin on A ==="
"$KVBMCTL" p2p unpin --instance-id "$INSTANCE_A" --session-id "$SESSION_ID" || true

# Hard assertions on the transfer.
[ "$PULLED" -ge 1 ] || fail "pull returned 0 blocks (session/attach/transport failure)"
[ "$PULLED" -eq "$COMMITTED" ] || fail "partial pull — pulled=$PULLED committed=$COMMITTED"

# --- 8. verify the pulled blocks landed in B's G2 (via the hub index) ------
# B runs the indexer feature too, so when the pull registers blocks into B's G2
# the block-registry EventsManager publishes them and the index lists B as a
# co-owner. This proves the transfer populated B's G2 with real blocks WITHOUT
# relying on the Rust cache-hit log (which the EngineCore subprocess does not
# surface in this environment).
B_U128=$("$PYTHON_BIN" -c 'import uuid,sys; print(uuid.UUID(sys.argv[1]).int)' "$INSTANCE_B")
FIRST_HASH=$(echo "$DECIMAL_CSV" | cut -d, -f1)
echo "=== verify B ($B_U128) owns pulled block $FIRST_HASH in the index ==="
owners_of() {
  "$PYTHON_BIN" - "$INDEXER" "$1" <<'PY'
import json, sys, urllib.request
base, h = sys.argv[1], sys.argv[2]
body = json.dumps({"hashes": [list(int(h).to_bytes(16, "big"))]}).encode()
req = urllib.request.Request(f"{base}/query", data=body,
    headers={"content-type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=5) as r:
        d = json.load(r)
except Exception:
    print(""); raise SystemExit
hit = d.get("hit") or {}
print(",".join(hit.get("instances", [])))
PY
}
B_OWNS=0
v_deadline=$(( $(date +%s) + 30 ))
while [ "$(date +%s)" -lt "$v_deadline" ]; do
  OWNERS=$(owners_of "$FIRST_HASH")
  echo "  owners: ${OWNERS:-<none>}"
  if echo "$OWNERS" | tr ',' '\n' | grep -qxF "$B_U128"; then B_OWNS=1; break; fi
  sleep 2
done
[ "$B_OWNS" -eq 1 ] || fail "after pull, B ($B_U128) is not an index owner of the pulled block (pull didn't populate B's G2?)" "$ROOT/instance_b.log"
echo "  -> B owns the pulled block in the index (transfer populated B's G2)"

# --- 9. R2 → B (same prompt; B serves from its now-populated G2) -----------
echo "=== R2: chat stream to instance B (8002) ==="
run_chat_stream 8002 r2 "$P1" | head -c 200; echo

render_trace

echo
echo "p2p-smoke PASS: pin/pull/unpin via kvbmctl moved $PULLED block(s) (of $COMMITTED committed, $HASH_COUNT requested) A→B; index confirms B now holds the pulled blocks; trace gate at $TRACE_GATE"
