#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Two-request CD smoke for the qwen3-tp2-experiments bundle. ASSUMES hub +
# Prefill + Decode are already up (the orchestrator does bringup + readiness).
#
# Mirrors .claude/skills/disagg-smoke/two-request-smoke.sh:140-300 but:
#   - Drops the bringup half (the orchestrator does that with TP=2 NUMA pinning).
#   - Prompts are ~340 / ~430 tokens so the matched_tokens >= 3 blocks rule
#     still holds at bs=64 (existing prompts were sized for bs=16).
#   - curl -m 90 -> -m 300 for 32B / 128k.
#   - Model name pulled from $KVBM_MODEL.
#
# R1: cold caches everywhere. Prefill computes; decode pulls.
# (between) Reset prefill G2 only -- decode keeps its cache.
# R2: same prefix + tail. Decode has a local G2 match -> forwards hashes;
#     prefill pulls those, forward-passes the new blocks, observer publishes.
#
# Usage: bash two-request-smoke.sh <experiment_root>
set -eu

SMOKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SMOKE_DIR/env.sh"

ROOT=${1:?"usage: $0 <experiment_root>"}
SKILL_BRINGUP=$KVBM_REPO/.claude/skills/disagg-bringup

strip_ansi() { sed 's/\x1b\[[0-9;]*m//g'; }
fail() {
  echo "FATAL: $1" >&2
  [ -n "${2:-}" ] && [ -f "$2" ] && { echo "--- tail $2 ---" >&2; tail -n 30 "$2" | strip_ansi >&2; }
  exit 1
}

# Discover the P/D split via kvbmctl (uses the feature's route const).
INSTS=$("$KVBM_KVBMCTL_BIN" disagg instances --hub http://127.0.0.1:$KVBM_HUB_DISCOVERY_PORT)
PREFILL_ID=$(echo "$INSTS" | python3 -c 'import json,sys; print(json.load(sys.stdin)["prefill"][0])')
DECODE_ID=$(echo "$INSTS" | python3 -c 'import json,sys; print(json.load(sys.stdin)["decode"][0])')
echo "PREFILL_ID=$PREFILL_ID DECODE_ID=$DECODE_ID"

# Verify block-layout mode matches requested mode (still useful at bs=64).
echo "--- block layout verification (expected: $KVBM_BLOCK_LAYOUT) ---"
bash "$SKILL_BRINGUP/verify-block-layout.sh" "$PREFILL_ID" "$KVBM_BLOCK_LAYOUT"
bash "$SKILL_BRINGUP/verify-block-layout.sh" "$DECODE_ID"  "$KVBM_BLOCK_LAYOUT"
echo "--- block layout OK ---"

# Clear both caches before R1.
HUB_CTRL="http://127.0.0.1:$KVBM_HUB_CONTROL_PORT"
curl -sS -X POST $HUB_CTRL/v1/instances/$PREFILL_ID/control/dev/reset \
  -H 'content-type: application/json' -d '{}' >/dev/null
curl -sS -X POST $HUB_CTRL/v1/instances/$DECODE_ID/control/dev/reset \
  -H 'content-type: application/json' -d '{}' >/dev/null

# ---- R1 (cold cache) ------------------------------------------------------
# ~340 tokens so >=3 full bs=64 blocks land in G2 on the prefill side. Pick a
# prompt that survives a 16-token continuation without falling off the end of
# the 128k context window.
P1='Conditional disaggregation splits the work of serving an autoregressive language model across two roles. The prefill role consumes the prompt and produces the keys and values for every layer in a single forward pass, while the decode role consumes those KVs and produces output tokens one at a time. The two roles are connected by a transfer plane: the prefill writes its layer-wise KV outputs into a host-side cache, the decode pulls the blocks it needs over RDMA, and the system continues from where the prefill stopped. The blocks themselves are addressed by content hashes, so when the same prompt arrives twice the second pass can skip recomputation entirely.'

echo === R1 SMOKE ===
R1=$(M="$KVBM_MODEL" P="$P1" python3 -c 'import json,os; print(json.dumps({"model":os.environ["M"],"prompt":os.environ["P"],"max_tokens":16,"temperature":0}))' \
  | curl -m 300 -sS -X POST http://127.0.0.1:8001/v1/completions \
      -H "Content-Type: application/json" -d @-)
echo "$R1" | head -c 400; echo

sleep 2  # let any tail-end audit events flush

# Reset prefill G2 ONLY -- decode keeps its 3+ block cache.
echo === RESETTING prefill G2 ONLY ===
curl -sS -X POST $HUB_CTRL/v1/instances/$PREFILL_ID/control/dev/reset \
  -H 'content-type: application/json' -d '{}' ; echo

# ---- R2 (warm decode, cleared prefill) -----------------------------------
P2="$P1 The hub itself plays no part in the actual transfer; it only mediates the discovery and validates that the layouts agree. Once the session is open the bytes flow directly between worker processes via NIXL, never through the hub. This keeps the hub a control-plane component and lets the transfer plane scale independently of the discovery plane."

echo === R2 SMOKE ===
R2=$(M="$KVBM_MODEL" P="$P2" python3 -c 'import json,os; print(json.dumps({"model":os.environ["M"],"prompt":os.environ["P"],"max_tokens":16,"temperature":0}))' \
  | curl -m 300 -sS -X POST http://127.0.0.1:8001/v1/completions \
      -H "Content-Type: application/json" -d @-)
echo "$R2" | head -c 400; echo

sleep 2

# ---- Validation report ---------------------------------------------------
echo
echo "================================================================"
echo "  Validation report  (logs in $ROOT)"
echo "================================================================"

echo
echo "-- R1 prefill: cache-hit rates (expect 0/N then forward-pass fills) --"
grep -a "Cache Hit Rates" "$ROOT/prefill.log" | head -3
echo
echo "-- R1 decode: full pull pipeline --"
grep -aE "kvbm_audit.*event=\"(worker_pull_chunk_start|worker_session_pull_call|session_pull_rdma_done|worker_session_pull_returned|worker_g2_to_g1_done|cd_payload_drop)\"" "$ROOT/decode.log" \
  | strip_ansi | head -20

echo
echo "-- R1 prefill: G1->G2 register events (proves G2 cache populated) --"
grep -aE "register_blocks|G1.*G2|register_g2" "$ROOT/prefill.log" | strip_ansi | head -10

echo
echo "-- R2 decode policy_decision (observed: matched_tokens=128 = 2 bs=64 blocks on the warm path) --"
grep -aE "kvbm_audit.*event=\"policy_decision\"" "$ROOT/decode.log" | strip_ansi

echo
echo "-- R2 prefill: gnmt path (expect ensure_started_async_onboard, NOT zero_passthrough) --"
grep -aE "kvbm_audit.*event=\"(cd_bound_ensure_started|ensure_started_async_onboard|ensure_started_zero_passthrough)\"" "$ROOT/prefill.log" \
  | strip_ansi | head -5

echo
echo "-- R2 prefill: pull-from-decode events (n>0 path) --"
grep -aE "kvbm_audit.*event=\"(session_pull_request|session_pull_send|session_pull_rdma_start|session_pull_rdma_done)\"" "$ROOT/prefill.log" \
  | strip_ansi | head -10

echo
echo "-- R2 decode: pull-from-prefill events (the net-new blocks) --"
grep -aE "kvbm_audit.*event=\"(worker_pull_chunk_start|worker_session_pull_call|session_pull_rdma_done|worker_g2_to_g1_done)\"" "$ROOT/decode.log" \
  | strip_ansi | tail -15

# ---- Asymmetric TP=2 decode tripwire (only fires when TP differs) --------
echo
echo "-- asymmetric pull events (only set on asymmetric TP runs) --"
grep -aE 'dispatch_asymmetric_pull|asymmetric.*pull|stamped.*pull' "$ROOT/decode.log" "$ROOT/prefill.log" 2>/dev/null \
  | strip_ansi | head -10 || true

# ---- Error summary -------------------------------------------------------
echo
echo "-- ANY ERRORs across all logs --"
TOTAL_ERRS=0
for s in hub prefill decode; do
  [ -f "$ROOT/$s.log" ] || continue
  cnt=$(grep -aE "ERROR" "$ROOT/$s.log" | strip_ansi | grep -v "kvbm_audit\|UCX\|invalid configuration\|kernel_config" | wc -l)
  echo "  $s.log: $cnt error lines"
  TOTAL_ERRS=$((TOTAL_ERRS + cnt))
done

echo
echo "two-request smoke: done"
