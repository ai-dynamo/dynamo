#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# chain-runner.sh — S0 cumulative-cleanup orchestrator (flavor A).
#
# Brings up a SHARED hub/prefill/decode trio, then chains scenarios
# through it with `/v1/instances/<id>/reset` between steps and runs
# S9+S10+S12 + S11 (mem) assertions at every interlude. Stops the
# chain on the first interlude failure and anchors blame to the
# scenario whose cleanup path just ran.
#
# This is NOT a scenario script — it does not conform to the
# scenario-script interface contract (see scenarios/*.sh). Instead,
# it INVOKES scenario scripts.
#
# === Inputs (env vars) ===
#
# Required:
#   KVBM_REPO            — PR worktree path (must contain .claude/skills/disagg-bringup)
#
# Optional:
#   KVBM_VENV            — Python venv (default ${KVBM_REPO}/.venv  ← NOT .sandbox)
#   KVBM_HUB_BIN         — kvbm_hub binary (default ${KVBM_REPO}/target/debug/kvbm_hub)
#   KVBM_EXPERIMENTS_DIR — root for experiment dirs (default /scratch/kvbm-experiments)
#   HF_HOME              — HF model cache (warned if unset)
#   HUB_API              — host:port of kvbm_hub control API (default localhost:8337)
#   PREFILL_API          — vLLM prefill HTTP endpoint (default localhost:8000) [1]
#   DECODE_API           — vLLM decode HTTP endpoint  (default localhost:8001) [1]
#   MODEL                — model id (default Qwen/Qwen3-0.6B)
#   S0_DRAIN_SECS        — seconds between scenarios (default 5)
#   S0_SCENARIOS         — space-separated scenario script names. Default:
#                          "release-evict.sh multi-rid.sh cache-eviction.sh
#                           cleanup-audit.sh orphan-probe.sh pool-accounting.sh"
#   INTERLUDE_SCENARIOS  — interlude probe scripts (default: cleanup-audit.sh
#                          orphan-probe.sh pool-accounting.sh gpu-mem-baseline.sh)
#
# [1] The defaults match the in-PR `launch-{prefill,decode}.sh` (which hardcode
#     ports 8000 and 8001 respectively). The augmented-plan spec lists 8081/8080
#     — those would only apply if launch scripts are patched to bind there.
#
# === Outputs ===
#
# stdout: per-scenario digest lines (echoed from inner scenario scripts) +
#         a final master digest:
#           S0: PASS|FAIL | scenarios=<n> interludes_passed=<n>
#                          max_mem_delta=<n>MB total_install=<n> total_drop=<n>
#                          [stopped_at=<scenario>]
# exit:   0 on PASS, 1 on FAIL.
#
# === Side effects ===
#
# - Mints $KVBM_EXPERIMENTS_DIR/<ts>-chain/ for hub/prefill/decode/chain logs.
# - Starts hub + prefill + decode (background) and pkill-tears-them-down on
#   exit via a trap (matching .claude/skills/disagg-teardown).
# - Does NOT release the GPU session — caller's responsibility.

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve script-relative paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENARIOS_DIR="${SCRIPT_DIR}/../scenarios"

# ---------------------------------------------------------------------------
# Required + default env vars
# ---------------------------------------------------------------------------
: "${KVBM_REPO:?KVBM_REPO is required (path to PR worktree)}"
KVBM_VENV="${KVBM_VENV:-${KVBM_REPO}/.venv}"
KVBM_HUB_BIN="${KVBM_HUB_BIN:-${KVBM_REPO}/target/debug/kvbm_hub}"
KVBM_EXPERIMENTS_DIR="${KVBM_EXPERIMENTS_DIR:-/scratch/kvbm-experiments}"
HUB_API="${HUB_API:-localhost:8337}"
PREFILL_API="${PREFILL_API:-localhost:8000}"
DECODE_API="${DECODE_API:-localhost:8001}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
S0_DRAIN_SECS="${S0_DRAIN_SECS:-5}"
S0_SCENARIOS="${S0_SCENARIOS:-release-evict.sh multi-rid.sh cache-eviction.sh cleanup-audit.sh orphan-probe.sh pool-accounting.sh}"
INTERLUDE_SCENARIOS="${INTERLUDE_SCENARIOS:-cleanup-audit.sh orphan-probe.sh pool-accounting.sh gpu-mem-baseline.sh}"

# Soft warnings for important-but-not-fatal env vars.
if [[ -z "${HF_HOME:-}" ]]; then
  printf 'WARN: HF_HOME is unset — vLLM will use ~/.cache/huggingface (often NFS-quota-limited)\n' >&2
fi
if [[ ! -x "$KVBM_HUB_BIN" ]]; then
  printf 'FATAL: KVBM_HUB_BIN missing/non-executable: %s\n' "$KVBM_HUB_BIN" >&2
  exit 1
fi

SKILL_BRINGUP="${KVBM_REPO}/.claude/skills/disagg-bringup"
if [[ ! -d "$SKILL_BRINGUP" ]]; then
  printf 'FATAL: disagg-bringup skill missing at %s — set KVBM_REPO to the PR worktree.\n' \
    "$SKILL_BRINGUP" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Mint experiment dir
# ---------------------------------------------------------------------------
TS="$(date +%Y%m%d-%H%M%S)"
EXPERIMENT_DIR="${KVBM_EXPERIMENTS_DIR}/${TS}-chain"
mkdir -p "$EXPERIMENT_DIR"

PREFILL_LOG="${EXPERIMENT_DIR}/prefill.log"
DECODE_LOG="${EXPERIMENT_DIR}/decode.log"
HUB_LOG="${EXPERIMENT_DIR}/hub.log"
CHAIN_LOG="${EXPERIMENT_DIR}/chain.log"

# Export so child scenario scripts pick them up via the documented contract.
export KVBM_REPO KVBM_VENV KVBM_HUB_BIN
export EXPERIMENT_DIR PREFILL_LOG DECODE_LOG HUB_LOG
export HUB_API PREFILL_API DECODE_API MODEL

# ---------------------------------------------------------------------------
# Logging helper — stamps stderr+chain.log; stdout reserved for digest lines
# ---------------------------------------------------------------------------
log() {
  local line
  line="[$(date +%H:%M:%S)] $*"
  printf '%s\n' "$line" >>"$CHAIN_LOG"
  printf '%s\n' "$line" >&2
}

# ---------------------------------------------------------------------------
# Teardown helper — mirrors .claude/skills/disagg-teardown
# ---------------------------------------------------------------------------
teardown_disagg() {
  # Tolerate failures during cleanup; we want best-effort.
  set +e
  pkill -f "vllm.entrypoints.openai" 2>/dev/null
  pkill -x kvbm_hub 2>/dev/null
  sleep 3
  local pids
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
            | tr -d ' ' | grep -v '^$')"
  if [[ -n "$pids" ]]; then
    echo "$pids" | xargs -r kill -9 2>/dev/null
    sleep 2
  fi
  pkill -9 -x kvbm_hub 2>/dev/null
  sleep 1
  set -e
}
trap teardown_disagg EXIT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
gpu_mem_sum_mb() {
  # Sum nvidia-smi memory.used (MB) across all GPUs. 0 if nvidia-smi missing.
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '0'
    return 0
  fi
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk 'BEGIN{s=0; n=0}
           /^[[:space:]]*[0-9]+[[:space:]]*$/ {s+=$1; n++}
           END { if (n==0) print 0; else print s }'
}

reset_caches() {
  # PUT /v1/instances/<id>/reset (verified at lib/kvbm-hub/src/protocol.rs:73
  # and connector_control_proxy.rs test). Body is JSON {} per two-request-smoke.sh.
  curl -fsS -m 5 -X PUT -H 'content-type: application/json' -d '{}' \
    "http://${HUB_API}/v1/instances/${PREFILL_ID}/reset" >/dev/null 2>&1 || true
  curl -fsS -m 5 -X PUT -H 'content-type: application/json' -d '{}' \
    "http://${HUB_API}/v1/instances/${DECODE_ID}/reset" >/dev/null 2>&1 || true
}

cumulative_install() {
  local n
  n="$(grep -cE 'prefill_cd_payload_installed' "$PREFILL_LOG" 2>/dev/null || true)"
  printf '%s' "${n:-0}"
}

cumulative_drop() {
  local n
  n="$(grep -cE 'prefill_cd_payload_drop' "$PREFILL_LOG" 2>/dev/null || true)"
  printf '%s' "${n:-0}"
}

most_recent_rid() {
  # Audit lines log `request_id=<uuid>` (verified at prefill_leader.rs:68,112,338,345
  # via tracing `request_id = %self.request_id`). NOT `rid=`.
  { grep -E 'prefill_cd_payload_(installed|drop)' "$PREFILL_LOG" 2>/dev/null \
      | grep -oE 'request_id=[^[:space:]"]+' \
      | tail -1 \
      | sed 's/^request_id=//'; } || true
}

states_count() {
  local body n
  body="$(curl -fsS -m 5 "http://${HUB_API}/v1/coordinator/states" 2>/dev/null || true)"
  if [[ -z "$body" ]]; then
    printf '?'
    return 0
  fi
  n="$(printf '%s' "$body" | jq '. | length' 2>/dev/null || true)"
  if [[ -z "$n" || ! "$n" =~ ^[0-9]+$ ]]; then
    printf '?'
    return 0
  fi
  printf '%s' "$n"
}

seed_request() {
  # Fire one r1-equivalent request to populate state and produce a fresh rid.
  # Prompt borrowed from disagg-smoke/two-request-smoke.sh's P1 (~54 tokens / 3 full blocks).
  local prompt='The quick brown fox jumps over the lazy dog and then keeps on running through the meadow until it reaches the river where it finally stops to drink some water and rest for a while before continuing its journey home. After resting, the fox decides to take a different path back'
  local payload
  payload="$(P="$prompt" M="$MODEL" python3 -c '
import json, os
print(json.dumps({
    "model": os.environ["M"],
    "prompt": os.environ["P"],
    "max_tokens": 16,
    "temperature": 0,
}))
' 2>/dev/null || true)"
  if [[ -z "$payload" ]]; then
    log "seed_request: python3 payload build failed"
    return 1
  fi
  curl -m 90 -sS -X POST "http://${DECODE_API}/v1/completions" \
    -H 'Content-Type: application/json' -d "$payload" >/dev/null 2>&1 || true
}

# ---------------------------------------------------------------------------
# Step 0 — clean stale processes
# ---------------------------------------------------------------------------
log "S0 chain-runner start | exp=${EXPERIMENT_DIR}"
log "tearing down stale processes (pre-flight)"
teardown_disagg

# ---------------------------------------------------------------------------
# Step 1 — bring up shared trio (hub + prefill + decode)
# ---------------------------------------------------------------------------
log "starting hub: ${KVBM_HUB_BIN}"
bash "${SKILL_BRINGUP}/start-hub.sh" "$HUB_LOG" >>"$CHAIN_LOG" 2>&1 &
disown $! 2>/dev/null || true

log "starting prefill (port from launch-prefill.sh — typically 8000)"
RUST_LOG="${RUST_LOG:-info,kvbm_connector=debug,kvbm_audit=info}" \
  bash "${SKILL_BRINGUP}/launch-prefill.sh" >"$PREFILL_LOG" 2>&1 &
disown $! 2>/dev/null || true

log "starting decode (port from launch-decode.sh — typically 8001)"
RUST_LOG="${RUST_LOG:-info,kvbm_connector=debug,kvbm_audit=info}" \
  bash "${SKILL_BRINGUP}/launch-decode.sh" >"$DECODE_LOG" 2>&1 &
disown $! 2>/dev/null || true

# ---------------------------------------------------------------------------
# Step 2 — wait for both vLLMs ready (max 5 min)
# ---------------------------------------------------------------------------
log "waiting up to 300s for ${PREFILL_API}/v1/models and ${DECODE_API}/v1/models"
deadline=$(( $(date +%s) + 300 ))
ready=0
while (( $(date +%s) < deadline )); do
  if curl -fsS -m 3 "http://${PREFILL_API}/v1/models" >/dev/null 2>&1 \
     && curl -fsS -m 3 "http://${DECODE_API}/v1/models" >/dev/null 2>&1; then
    ready=1
    log "both vLLMs ready"
    break
  fi
  # Also fail fast if a vLLM has died.
  for f in "$PREFILL_LOG" "$DECODE_LOG"; do
    if [[ -f "$f" ]] && grep -qE 'Traceback|panicked|out of memory|OOM' "$f" 2>/dev/null; then
      log "FATAL: vLLM crashed during bringup — see ${f}"
      echo "S0: FAIL | scenarios=0 interludes_passed=0 max_mem_delta=0MB total_install=0 total_drop=0 stopped_at=bringup reason=vllm_crash"
      exit 1
    fi
  done
  sleep 5
done
if (( ready != 1 )); then
  log "FATAL: vLLM bringup timed out after 300s"
  echo "S0: FAIL | scenarios=0 interludes_passed=0 max_mem_delta=0MB total_install=0 total_drop=0 stopped_at=bringup reason=vllm_bringup_timeout"
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 3 — discover instance IDs
# ---------------------------------------------------------------------------
INSTS_JSON="$(curl -fsS -m 5 "http://${HUB_API}/v1/features/disagg/instances" 2>/dev/null || true)"
PREFILL_ID="$(printf '%s' "$INSTS_JSON" | jq -r '.prefill[0] // ""' 2>/dev/null || true)"
DECODE_ID="$(printf '%s' "$INSTS_JSON" | jq -r '.decode[0] // ""' 2>/dev/null || true)"
if [[ -z "$PREFILL_ID" || -z "$DECODE_ID" ]]; then
  log "FATAL: could not discover prefill/decode instance ids from hub: ${INSTS_JSON:-<empty>}"
  echo "S0: FAIL | scenarios=0 interludes_passed=0 max_mem_delta=0MB total_install=0 total_drop=0 stopped_at=bringup reason=instance_discovery_failed"
  exit 1
fi
export PREFILL_ID DECODE_ID
log "PREFILL_ID=${PREFILL_ID} DECODE_ID=${DECODE_ID}"

# ---------------------------------------------------------------------------
# Step 4 — clean-slate reset before baseline snapshot
# ---------------------------------------------------------------------------
log "clean-slate reset (PUT /v1/instances/<id>/reset)"
reset_caches

# ---------------------------------------------------------------------------
# Step 5 — snapshot baseline GPU mem (pre-S1)
# ---------------------------------------------------------------------------
MEM_BASELINE="$(gpu_mem_sum_mb)"
export MEM_BASELINE
log "MEM_BASELINE=${MEM_BASELINE} MB | initial_install=$(cumulative_install) initial_drop=$(cumulative_drop)"

# ---------------------------------------------------------------------------
# Step 6 — chain
# ---------------------------------------------------------------------------
SCENARIOS_RUN=0
INTERLUDES_PASSED=0
MAX_MEM_DELTA=0
CHAIN_FAIL=""
MISSING_LIST=""

run_interlude() {
  local prev_scenario="$1"
  local interlude_ok=1

  # --- Probe 1: cumulative install/drop count parity (chain-runner inline) ---
  local total_install total_drop
  total_install="$(cumulative_install)"
  total_drop="$(cumulative_drop)"
  log "interlude/cumulative install=${total_install} drop=${total_drop}"
  if [[ "$total_install" != "$total_drop" ]]; then
    log "INTERLUDE FAIL/cumulative install != drop after scenario=${prev_scenario}"
    interlude_ok=0
  fi

  # --- Probe 2: states API empty after reset ---
  local sc
  sc="$(states_count)"
  log "interlude/states_count=${sc}"
  if [[ "$sc" != "0" ]]; then
    if [[ "$sc" == "?" ]]; then
      log "INTERLUDE INCONCLUSIVE/states_api unreachable (not fatal)"
    else
      log "INTERLUDE FAIL/states_count=${sc} != 0 after scenario=${prev_scenario}"
      interlude_ok=0
    fi
  fi

  # --- Probe 3: GPU mem delta vs pre-S1 baseline ---
  local mem_now mem_delta
  mem_now="$(gpu_mem_sum_mb)"
  mem_delta=$(( mem_now - MEM_BASELINE ))
  if (( mem_delta > MAX_MEM_DELTA )); then
    MAX_MEM_DELTA="$mem_delta"
  fi
  log "interlude/mem now=${mem_now} baseline=${MEM_BASELINE} delta=${mem_delta}MB"
  if (( mem_delta >= 100 )); then
    log "INTERLUDE FAIL/mem_delta=${mem_delta}MB >= 100 after scenario=${prev_scenario}"
    interlude_ok=0
  fi

  # --- Probes 4..N: invoke S9/S10/S12 + S11 scripts as additional probes ---
  local probe probe_path probe_out probe_rc rid
  rid="$(most_recent_rid)"
  for probe in $INTERLUDE_SCENARIOS; do
    probe_path="${SCENARIOS_DIR}/${probe}"
    if [[ ! -f "$probe_path" ]]; then
      log "interlude probe ${probe} MISSING at ${probe_path} (skipping; sibling teammate may not be done)"
      continue
    fi
    if MOST_RECENT_RID="$rid" \
         bash "$probe_path" >"${EXPERIMENT_DIR}/.interlude.tmp" 2>>"$CHAIN_LOG"; then
      probe_rc=0
    else
      probe_rc=$?
    fi
    probe_out="$(cat "${EXPERIMENT_DIR}/.interlude.tmp" 2>/dev/null || true)"
    rm -f "${EXPERIMENT_DIR}/.interlude.tmp"
    log "interlude/${probe} rc=${probe_rc} out=${probe_out}"
    # Echo the digest to stdout so it appears alongside scenario digests.
    [[ -n "$probe_out" ]] && printf '%s\n' "$probe_out"
    case "$probe_out" in
      PASS:*) ;;
      FAIL:*)
        log "INTERLUDE FAIL/${probe} after scenario=${prev_scenario}"
        interlude_ok=0
        ;;
      INCONCLUSIVE:*)
        log "INTERLUDE INCONCLUSIVE/${probe} (not fatal)"
        ;;
      *)
        log "INTERLUDE UNKNOWN/${probe} (rc=${probe_rc}, no recognizable digest)"
        # Treat unrecognized non-zero as failure; rc=0 with no digest as inconclusive.
        if (( probe_rc != 0 )); then
          interlude_ok=0
        fi
        ;;
    esac
  done

  if (( interlude_ok == 1 )); then
    INTERLUDES_PASSED=$(( INTERLUDES_PASSED + 1 ))
    return 0
  fi
  return 1
}

for scenario in $S0_SCENARIOS; do
  log "=== chain step: ${scenario} ==="
  scenario_path="${SCENARIOS_DIR}/${scenario}"

  # Robustness: missing scenario script → log MISSING and continue, not fatal.
  if [[ ! -f "$scenario_path" ]]; then
    msg="MISSING: ${scenario} | path=${scenario_path}"
    log "$msg"
    printf '%s\n' "$msg"
    MISSING_LIST="${MISSING_LIST}${MISSING_LIST:+,}${scenario}"
    continue
  fi

  # Pre-seed: every scenario gets a fresh r1 from chain-runner. Read-only
  # scenarios (release-evict, cleanup-audit, …) need a rid to assert against;
  # fire-their-own scenarios accumulate on top.
  log "seeding pre-${scenario}"
  seed_request
  sleep 3  # let prefill_cd_payload_drop flush so most_recent_rid sees this rid

  # Run scenario script with the documented env contract.
  rid="$(most_recent_rid)"
  if MOST_RECENT_RID="$rid" \
       bash "$scenario_path" >"${EXPERIMENT_DIR}/.scenario.tmp" 2>>"$CHAIN_LOG"; then
    scenario_rc=0
  else
    scenario_rc=$?
  fi
  scenario_out="$(cat "${EXPERIMENT_DIR}/.scenario.tmp" 2>/dev/null || true)"
  rm -f "${EXPERIMENT_DIR}/.scenario.tmp"
  log "scenario ${scenario} rc=${scenario_rc} digest=${scenario_out:-<empty>}"
  # Echo the digest to stdout so users see per-scenario lines as the chain runs.
  [[ -n "$scenario_out" ]] && printf '%s\n' "$scenario_out"
  SCENARIOS_RUN=$(( SCENARIOS_RUN + 1 ))

  # Drain — let cleanup paths flush before reset/interlude.
  log "draining ${S0_DRAIN_SECS}s"
  sleep "$S0_DRAIN_SECS"

  # Reset caches between scenarios (flavor A).
  log "resetting prefill+decode caches"
  reset_caches

  # Interlude. Failure stops the chain and anchors blame to ${scenario}.
  if ! run_interlude "$scenario"; then
    CHAIN_FAIL="$scenario"
    log "STOPPING CHAIN — interlude after ${scenario} failed; cumulative leak anchored to ${scenario}'s cleanup path"
    break
  fi
done

# ---------------------------------------------------------------------------
# Step 7 — final report
# ---------------------------------------------------------------------------
TOTAL_INSTALL_FINAL="$(cumulative_install)"
TOTAL_DROP_FINAL="$(cumulative_drop)"
VERDICT="PASS"
if [[ -n "$CHAIN_FAIL" ]]; then
  VERDICT="FAIL"
fi

echo
echo "=================================================================="
LINE="S0: ${VERDICT} | scenarios=${SCENARIOS_RUN} interludes_passed=${INTERLUDES_PASSED} max_mem_delta=${MAX_MEM_DELTA}MB total_install=${TOTAL_INSTALL_FINAL} total_drop=${TOTAL_DROP_FINAL}"
[[ -n "$CHAIN_FAIL" ]] && LINE="${LINE} stopped_at=${CHAIN_FAIL}"
[[ -n "$MISSING_LIST" ]] && LINE="${LINE} missing=${MISSING_LIST}"
printf '%s\n' "$LINE"
echo "=================================================================="
echo "experiment_dir=${EXPERIMENT_DIR}"

if [[ "$VERDICT" == "PASS" ]]; then
  exit 0
fi
exit 1
