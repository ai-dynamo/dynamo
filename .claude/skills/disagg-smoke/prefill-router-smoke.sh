#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Two-request CD smoke (R1 cold + R2 warm) against the NEW PrefillRouter
# hub feature. Decode stays `vllm serve` (vanilla vLLM + kvbm v2 decode
# connector); prefill is selected by KVBM_PREFILL_VARIANT:
#
#   kvbm-wrap : python -m kvbm.vllm.prefill   (auto-attaches via kvbm.hub)
#   dynamo    : python -m dynamo.vllm --disaggregation-mode prefill
#               (auto-attaches via worker_factory.py hook)
#
# Both variants register with the hub's PrefillRouter as Velo backends, so
# the hub dispatches CD prefill requests via VeloExecutionBackend.
#
# Validations on top of the standard R1/R2 audit-log checks:
#   1. prefill log contains the "kvbm prefill router auto-wired" line
#   2. hub /v1/features/prefill-router/targets shows exactly one velo target
#   3. hub log shows the velo backend being used to dispatch prefill work
#
# Usage:
#   KVBM_PREFILL_VARIANT={kvbm-wrap|dynamo} bash prefill-router-smoke.sh [logs_dir]
set -eu

VARIANT=${KVBM_PREFILL_VARIANT:?must be kvbm-wrap or dynamo}
case "$VARIANT" in
  kvbm-wrap) PREFILL_LAUNCHER=launch-prefill-kvbm-wrap.sh ;;
  dynamo)    PREFILL_LAUNCHER=launch-prefill-dynamo.sh ;;
  *) echo "bad KVBM_PREFILL_VARIANT: $VARIANT (expected kvbm-wrap or dynamo)" >&2; exit 2 ;;
esac

SMOKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO=${KVBM_REPO:-$(cd "$SMOKE_DIR/../../.." && pwd)}
export KVBM_VENV=${KVBM_VENV:-$DYNAMO/.sandbox}
SKILL_BRINGUP=$DYNAMO/.claude/skills/disagg-bringup
SKILL_TRACE=$DYNAMO/.claude/skills/disagg-trace
HUB_READY_TIMEOUT=${KVBM_HUB_READY_TIMEOUT:-300}
VLLM_READY_TIMEOUT=${KVBM_VLLM_READY_TIMEOUT:-300}
LABEL=${KVBM_EXPERIMENT_LABEL:-prefill-router-$VARIANT}
KVBM_BLOCK_LAYOUT=${KVBM_BLOCK_LAYOUT:-operational}
KVBM_ONBOARD_MODE=${KVBM_ONBOARD_MODE:-inter}
export KVBM_BLOCK_LAYOUT KVBM_ONBOARD_MODE

ROOT=${1:-$(bash $SKILL_BRINGUP/new-experiment.sh "$LABEL")}
echo "EXP=$ROOT"
echo "$ROOT" > /tmp/cd-trace-current-exp

# Tear down stale.
pkill -f "vllm.entrypoints.openai" 2>/dev/null || true
pkill -f "kvbm.vllm.prefill" 2>/dev/null || true
pkill -f "dynamo.vllm" 2>/dev/null || true
sleep 3
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' || true)
[ -n "$PIDS" ] && echo "$PIDS" | xargs -r kill -9 2>/dev/null || true
sleep 2
pkill -9 -f kvbm_hub 2>/dev/null || true
rm -f /tmp/velo-kvbm-*.sock 2>/dev/null || true
sleep 1

fail_dump() {
  echo "FATAL: $1" >&2
  if [ -n "${2:-}" ] && [ -f "$2" ]; then
    echo "--- tail $2 ---" >&2
    tail -n 50 "$2" | sed 's/\x1b\[[0-9;]*m//g' >&2
  fi
  pkill -f "vllm.entrypoints.openai" 2>/dev/null || true
  pkill -f "kvbm.vllm.prefill"        2>/dev/null || true
  pkill -f "dynamo.vllm"               2>/dev/null || true
  pkill -9 -f kvbm_hub                  2>/dev/null || true
  exit 1
}

# Hub with the new --prefill-router flag (set by the updated start-hub.sh).
# Truncate every log this run will write to BEFORE launching the hub.
# Why: `kvbm-hub-bringup/start-hub.sh` redirects the hub binary with `>>`
# (append), and a caller may pass a previously-used $ROOT as $1. Without
# this truncation, hub.log can carry forward `PrefillRouter: completed
# backend="velo"` lines from a prior smoke and silently satisfy the velo
# proof assertion against stale state. prefill.log / decode.log are
# already truncated by the `>` redirects below, but we zero them here too
# so all three files share one consistent "freshness floor" written by
# THIS invocation of the smoke harness.
: > "$ROOT/hub.log"
: > "$ROOT/prefill.log"
: > "$ROOT/decode.log"

bash $SKILL_BRINGUP/start-hub.sh "$ROOT/hub.log" &
HUB_PID=$!

echo "waiting for hub /health (timeout ${HUB_READY_TIMEOUT}s)..."
hub_deadline=$(( $(date +%s) + HUB_READY_TIMEOUT ))
until curl -fsS -m 5 http://127.0.0.1:8337/health >/dev/null 2>&1; do
  kill -0 "$HUB_PID" 2>/dev/null || fail_dump "hub process exited before becoming ready" "$ROOT/hub.log"
  [ "$(date +%s)" -ge "$hub_deadline" ] && fail_dump "hub not ready after ${HUB_READY_TIMEOUT}s" "$ROOT/hub.log"
  sleep 2
done
echo "HUB UP $(date)"

# Sanity: hub must offer the prefill-router feature.
if ! curl -fsS -m 5 http://127.0.0.1:1337/v1/config | grep -q '"prefill_router"'; then
  fail_dump "hub did not register the prefill_router feature (check --prefill-router flag in start-hub.sh)" "$ROOT/hub.log"
fi

# Launch prefill first and wait for it to be fully ready (vLLM init done +
# kvbm prefill router auto-wired). vLLM's --gpu-memory-utilization is a
# fraction of TOTAL GPU memory, not free memory, so the second-to-start
# instance sees a tighter budget and may OOM the KV cache calculation. We
# stage prefill -> decode so each gets a clean accounting pass.
RUST_LOG=${RUST_LOG:-info,kvbm_connector=debug,kvbm_audit=info,kvbm_hub=debug} \
  bash $SKILL_BRINGUP/$PREFILL_LAUNCHER > "$ROOT/prefill.log" 2>&1 &
PREFILL_PID=$!

echo "waiting for prefill auto-wire log line (timeout ${VLLM_READY_TIMEOUT}s)..."
prefill_deadline=$(( $(date +%s) + VLLM_READY_TIMEOUT ))
until grep -aq "kvbm prefill router auto-wired" "$ROOT/prefill.log" 2>/dev/null; do
  kill -0 "$PREFILL_PID" 2>/dev/null || fail_dump "prefill exited before auto-wire" "$ROOT/prefill.log"
  [ "$(date +%s)" -ge "$prefill_deadline" ] && fail_dump "no auto-wire log line after ${VLLM_READY_TIMEOUT}s" "$ROOT/prefill.log"
  sleep 5
done
echo "PREFILL AUTO-WIRED $(date)"

RUST_LOG=${RUST_LOG:-info,kvbm_connector=debug,kvbm_audit=info} \
  bash $SKILL_BRINGUP/launch-decode.sh > "$ROOT/decode.log" 2>&1 &
DECODE_PID=$!

echo "waiting for decode (timeout ${VLLM_READY_TIMEOUT}s)..."
vllm_deadline=$(( $(date +%s) + VLLM_READY_TIMEOUT ))
until curl -fsS -m 5 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; do
  kill -0 "$DECODE_PID" 2>/dev/null || fail_dump "decode exited before ready" "$ROOT/decode.log"
  [ "$(date +%s)" -ge "$vllm_deadline" ] && fail_dump "decode not ready after ${VLLM_READY_TIMEOUT}s" "$ROOT/decode.log"
  sleep 5
done
echo "DECODE UP $(date)"

# Hub-side assertion: PrefillRouter has exactly one velo-backed target.
TARGETS_JSON=$(curl -fsS -m 5 http://127.0.0.1:8337/v1/features/prefill-router/targets) \
  || fail_dump "could not GET /v1/features/prefill-router/targets" "$ROOT/hub.log"
echo "Targets: $TARGETS_JSON"
N_VELO=$(echo "$TARGETS_JSON" | python3 -c '
import json,sys
d=json.load(sys.stdin)
print(sum(1 for t in d["targets"] if t["backend"]=="velo"))
')
if [ "$N_VELO" != "1" ]; then
  fail_dump "expected exactly one velo target, got $N_VELO (targets=$TARGETS_JSON)" "$ROOT/hub.log"
fi
echo "VELO TARGET REGISTERED (n=1)"

# Discover the P/D split via kvbmctl.
KVBMCTL=${KVBM_KVBMCTL_BIN:-$DYNAMO/target/debug/kvbmctl}
INSTS=$("$KVBMCTL" disagg instances --hub http://127.0.0.1:1337)
PREFILL_ID=$(echo "$INSTS" | python3 -c 'import json,sys; print(json.load(sys.stdin)["prefill"][0])')
DECODE_ID=$(echo "$INSTS" | python3 -c 'import json,sys; print(json.load(sys.stdin)["decode"][0])')
echo "PREFILL_ID=$PREFILL_ID DECODE_ID=$DECODE_ID"

# Verify block layout matches.
echo "--- block layout verification (expected: $KVBM_BLOCK_LAYOUT) ---"
bash "$SKILL_BRINGUP/verify-block-layout.sh" "$PREFILL_ID" "$KVBM_BLOCK_LAYOUT"
bash "$SKILL_BRINGUP/verify-block-layout.sh" "$DECODE_ID"  "$KVBM_BLOCK_LAYOUT"
echo "--- block layout OK ---"

# Reset both caches.
curl -sS -X POST http://127.0.0.1:8337/v1/instances/$PREFILL_ID/control/dev/reset \
  -H 'content-type: application/json' -d '{}' >/dev/null
curl -sS -X POST http://127.0.0.1:8337/v1/instances/$DECODE_ID/control/dev/reset \
  -H 'content-type: application/json' -d '{}' >/dev/null

# ---- R1 (cold) ----
P1='The quick brown fox jumps over the lazy dog and then keeps on running through the meadow until it reaches the river where it finally stops to drink some water and rest for a while before continuing its journey home. After resting, the fox decides to take a different path back'

echo === R1 SMOKE ===
R1=$(P="$P1" python3 -c 'import json,os; print(json.dumps({"model":"Qwen/Qwen3-0.6B","prompt":os.environ["P"],"max_tokens":16,"temperature":0}))' \
  | curl -m 120 -sS -X POST http://127.0.0.1:8001/v1/completions \
      -H "Content-Type: application/json" -d @-)
echo "$R1" | head -c 400; echo

sleep 2

# Reset prefill G2 ONLY.
echo === RESETTING prefill G2 ONLY ===
curl -sS -X POST http://127.0.0.1:8337/v1/instances/$PREFILL_ID/control/dev/reset \
  -H 'content-type: application/json' -d '{}' ; echo

# ---- R2 (warm decode, cleared prefill) ----
P2="$P1 The river flowed gently between the green hills and bright wildflowers swaying in the spring breeze, while birds sang above."

echo === R2 SMOKE ===
R2=$(P="$P2" python3 -c 'import json,os; print(json.dumps({"model":"Qwen/Qwen3-0.6B","prompt":os.environ["P"],"max_tokens":16,"temperature":0}))' \
  | curl -m 120 -sS -X POST http://127.0.0.1:8001/v1/completions \
      -H "Content-Type: application/json" -d @-)
echo "$R2" | head -c 400; echo

sleep 2

# ---- Validation report ----
echo
echo "================================================================"
echo "  Validation report  (variant=$VARIANT)"
echo "================================================================"

echo
echo "-- R1/R2 produced text? --"
R1_OK=$(echo "$R1" | python3 -c 'import json,sys
try: d=json.load(sys.stdin)
except Exception: print(0); sys.exit()
print(1 if d.get("choices") and d["choices"][0].get("text") else 0)')
R2_OK=$(echo "$R2" | python3 -c 'import json,sys
try: d=json.load(sys.stdin)
except Exception: print(0); sys.exit()
print(1 if d.get("choices") and d["choices"][0].get("text") else 0)')
echo "  R1 produced text: $R1_OK"
echo "  R2 produced text: $R2_OK"
[ "$R1_OK" = "1" ] || fail_dump "R1 returned no text" "$ROOT/decode.log"
[ "$R2_OK" = "1" ] || fail_dump "R2 returned no text" "$ROOT/decode.log"

echo
echo "-- R2 policy_decision (expect matched_tokens >= 48) --"
grep -aE "kvbm_audit.*event=\"policy_decision\"" "$ROOT/decode.log" | sed 's/\x1b\[[0-9;]*m//g' | tail -3

echo
echo "-- R2 prefill pull-from-decode events --"
grep -aE "kvbm_audit.*event=\"(session_pull_request|session_pull_rdma_done)\"" "$ROOT/prefill.log" \
  | sed 's/\x1b\[[0-9;]*m//g' | tail -5

echo
echo "-- Hub: prefill_router dispatch via Velo? --"
VELO_DISPATCH=$(grep -aE "PrefillRouter|velo.*dispatch|VeloExecutionBackend|backend=velo|backend = \"velo\"" "$ROOT/hub.log" \
  | sed 's/\x1b\[[0-9;]*m//g' | grep -v "no targets" | head -10)
echo "$VELO_DISPATCH"

# Hard assertions on the hub's PrefillRouter dispatch log: prove BOTH R1
# AND R2 actually traversed the new velo dispatch path end-to-end and
# completed cleanly. The smoke would otherwise be a false-positive: R1/R2
# can both return text even if velo was completely broken, because decode
# falls back to local prefill when the prefill engine dies or the request
# gets rejected. We require:
#
#   1. a `target registered ... backend="velo"` event — proves the
#      worker's auto-attach + hub registration handshake succeeded.
#   2. `PrefillRouter: completed backend="velo"` count == 2 — proves BOTH
#      R1 (cold) AND R2 (warm) were dispatched via velo and the worker
#      successfully ran the prefill. R2 is the warm-cache path that
#      historically triggers the kv_transfer_params bytes-vs-list-int
#      mismatch in the worker scheduler; counting both requests catches
#      regressions on either path.
#   3. `PrefillRouter: rejected` count == 0 — any rejection (e.g. an
#      EngineDeadError mid-step) means the smoke produced text via a
#      fallback path, not via velo. Hard-fail.
#   4. prefill audit shows R2's warm-cache pull lifecycle
#      (session_pull_request → session_pull_rdma_done) — proves the R2
#      velo dispatch actually drove the prefill engine through a real
#      pull-from-decode (not just an empty no-op).
# The hub log is tracing-formatted with ANSI color escapes between every
# field name and value (e.g. `backend\x1b[0m\x1b[2m=\x1b[0m"velo"`), so the
# literal substring `backend="velo"` does not appear contiguously in the raw
# bytes. Strip ANSI first, then grep.
strip_ansi() { sed -E 's/\x1b\[[0-9;]*m//g' "$1" 2>/dev/null; }
HUB_PLAIN=$(strip_ansi "$ROOT/hub.log")
count_in_hub() {
  echo "$HUB_PLAIN" | grep -aEc "$1" 2>/dev/null || true
}
VELO_REGISTERED=$(count_in_hub 'PrefillRouter: target registered.*backend="velo"')
VELO_COMPLETED=$(count_in_hub 'PrefillRouter: completed backend="velo"')
VELO_REJECTED=$(count_in_hub 'PrefillRouter: rejected backend="velo"')
# `grep -c` outputs 0 on no-match (with exit 1) so the values are always
# integers; the `|| true` only guards against pipefail in stricter shells.
VELO_REGISTERED=${VELO_REGISTERED:-0}
VELO_COMPLETED=${VELO_COMPLETED:-0}
VELO_REJECTED=${VELO_REJECTED:-0}
echo "  velo_registered_events=$VELO_REGISTERED velo_completed=$VELO_COMPLETED velo_rejected=$VELO_REJECTED"
[ "$VELO_REGISTERED" -ge 1 ] || fail_dump "no PrefillRouter velo target registration in hub.log — auto-attach hook never reached the hub" "$ROOT/hub.log"
[ "$VELO_COMPLETED" -eq 2 ] || fail_dump "expected exactly 2 PrefillRouter velo completions (R1 cold + R2 warm), got $VELO_COMPLETED. R1=text + R2=text can both come from decode-side local prefill fallback when velo dispatch fails; the velo completion count is the only way to prove BOTH requests actually round-tripped through the hub." "$ROOT/hub.log"
[ "$VELO_REJECTED" -eq 0 ] || fail_dump "$VELO_REJECTED PrefillRouter velo dispatch rejection(s) — any rejection means the request got handled by a fallback path, not by velo. The audit trace is not fully resolved." "$ROOT/hub.log"

# Audit-trace gate on R2 SPECIFICALLY. The CD flow emits two distinct
# pull-related event families on the prefill side, and only one of them
# is R2-specific:
#
#   * `session_recv_pull`  — prefill RECEIVES decode's pull request.
#     Fires for EVERY CD round-trip (R1 + R2 both have decode pulling
#     newly-computed blocks back from prefill). Counting this would let
#     an R1-only smoke false-pass.
#
#   * `session_pull_request` — prefill INITIATES a pull FROM decode for
#     blocks that decode already had warm-cached. ONLY R2 (the warm-
#     cache request) triggers this, because R1 starts cold.
#
#   * `session_pull_rdma_done` (prefill side) — the RDMA pull-from-decode
#     completion record. Also R2-specific (paired with the request).
#
# We gate on `session_pull_request` + `session_pull_rdma_done` so a smoke
# whose R2 never traversed the prefill engine — e.g. one that returned
# text via decode-side local prefill fallback — fails loud here.
PULL_REQ=$(sed -E 's/\x1b\[[0-9;]*m//g' "$ROOT/prefill.log" 2>/dev/null \
  | grep -aEc 'kvbm_audit.*event="session_pull_request"')
PULL_DONE=$(sed -E 's/\x1b\[[0-9;]*m//g' "$ROOT/prefill.log" 2>/dev/null \
  | grep -aEc 'kvbm_audit.*event="session_pull_rdma_done"')
PULL_REQ=${PULL_REQ:-0}
PULL_DONE=${PULL_DONE:-0}
echo "  prefill_audit: session_pull_request=$PULL_REQ session_pull_rdma_done=$PULL_DONE"
[ "$PULL_REQ" -ge 1 ] || fail_dump "no session_pull_request event in prefill.log — R2's warm-cache pull-from-decode never executed on the prefill worker. The smoke likely returned R2 text via decode-side local prefill fallback. Audit trace not resolved." "$ROOT/prefill.log"
[ "$PULL_DONE" -ge 1 ] || fail_dump "no session_pull_rdma_done event in prefill.log — R2's RDMA pull-from-decode did not complete on the prefill side. Audit trace not resolved." "$ROOT/prefill.log"

# Final counters snapshot.
echo
echo "-- Hub: final counters --"
curl -sS -m 5 http://127.0.0.1:8337/v1/features/prefill-router/counters | python3 -m json.tool 2>/dev/null || true

echo
echo "-- ERRORs across logs --"
TOTAL_ERRS=0
for s in hub prefill decode; do
  cnt=$(grep -aE "ERROR" "$ROOT/$s.log" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' \
        | grep -v "kvbm_audit\|UCX\|invalid configuration\|kernel_config" | wc -l)
  echo "  $s.log: $cnt error lines"
  TOTAL_ERRS=$((TOTAL_ERRS + cnt))
done

if [ -x "$SKILL_TRACE/cd-trace.py" ]; then
  python3 "$SKILL_TRACE/cd-trace.py" "$ROOT" 2>/dev/null && echo "Open: file://$ROOT/trace.html"
fi

echo
echo "smoke[$VARIANT]: DONE  (R1=$R1_OK R2=$R2_OK velo_targets=$N_VELO total_errs=$TOTAL_ERRS)"

# Tear down — leave logs in $ROOT for inspection.
pkill -f "vllm.entrypoints.openai" 2>/dev/null || true
pkill -f "kvbm.vllm.prefill"        2>/dev/null || true
pkill -f "dynamo.vllm"               2>/dev/null || true
sleep 2
pkill -9 -f kvbm_hub                  2>/dev/null || true
