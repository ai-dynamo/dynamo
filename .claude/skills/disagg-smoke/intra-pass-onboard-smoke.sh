#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Single-instance intra-pass-onboard smoke.
#
# Two requests against ONE KVBM-enabled vLLM (no P/D disagg, no hub). R1 is
# a cold-cache prompt that fills G1 and offloads to G2 in the background.
# R2 is the same prefix + extension; the connector finds the G2 match,
# and with `KVBM_ONBOARD_MODE=intra` scatters the cached blocks back into
# G1 one layer at a time during R2's forward pass via
# `execute_local_layerwise_onboard`.
#
# Why this smoke exists: the disagg `two-request-smoke.sh` runs through
# the disagg coordinator wrapper, which short-circuits
# `ConnectorLeader::update_state_after_alloc` and never calls
# `prepare_intra_pass_onboarding` — so `metadata.intra_pass_load` stays
# `None` even under `onboard_mode=intra`. Aggregated (single-instance)
# mode bypasses the CD wrapper and the intra-pass-onboard path actually
# fires. Set `KVBM_BLOCK_LAYOUT=universal` together with intra mode to
# exercise the Phase-4b kernel-catalog + `layer_range` path end-to-end;
# any stride regression in `dispatch_transform_kernel` surfaces as an
# R2 panic or a checksum-equivalent failure mode.
#
# Intra-pass *offload* (per-layer G1→G2 during forward pass) is not yet
# wired up to drive from vLLM in this build, so this smoke covers
# onboard only.
#
# Honors:
#   KVBM_REPO              (default: /home/ryan/repos/dynamo)
#   KVBM_BLOCK_LAYOUT      (operational | universal ; default operational)
#   KVBM_ONBOARD_MODE      (inter | intra            ; default intra)
#   KVBM_SINGLE_PORT       (default: 8002)
#   KVBM_EXPERIMENT_LABEL  (default: intra-pass-onboard)
#
# Usage: bash intra-pass-onboard-smoke.sh [logs_dir]
#   If logs_dir not given, mints $KVBM_EXPERIMENTS_DIR/<ts>-<label>/.
#
# Exit codes:
#   0 — intra-pass-onboard fired ≥1 time on R2 with no kernel transform errors
#   1 — validation failed (smoke reports specifics; check vllm.log + report)
set -eu

DYNAMO=${KVBM_REPO:-/home/ryan/repos/dynamo}
SKILL_BRINGUP=$DYNAMO/.claude/skills/disagg-bringup
LABEL=${KVBM_EXPERIMENT_LABEL:-intra-pass-onboard}
KVBM_BLOCK_LAYOUT=${KVBM_BLOCK_LAYOUT:-operational}
KVBM_ONBOARD_MODE=${KVBM_ONBOARD_MODE:-intra}
KVBM_SINGLE_PORT=${KVBM_SINGLE_PORT:-8002}
export KVBM_BLOCK_LAYOUT KVBM_ONBOARD_MODE KVBM_SINGLE_PORT

ROOT=${1:-$(bash $SKILL_BRINGUP/new-experiment.sh "$LABEL")}
echo "EXP=$ROOT"
echo "mode=$KVBM_ONBOARD_MODE block_layout=$KVBM_BLOCK_LAYOUT port=$KVBM_SINGLE_PORT"

# Tear down stale vLLMs (best-effort).
pkill -f "vllm.entrypoints.openai" 2>/dev/null || true
sleep 3
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' || true)
[ -n "$PIDS" ] && echo "$PIDS" | xargs -r kill -9 2>/dev/null || true
sleep 2

# Bring up the single instance.
RUST_LOG=${RUST_LOG:-info,kvbm_connector=debug,kvbm_audit=info} \
  bash "$SKILL_BRINGUP/launch-single.sh" > "$ROOT/vllm.log" 2>&1 &
disown

echo "waiting for vLLM..."
until curl -fsS "http://127.0.0.1:$KVBM_SINGLE_PORT/v1/models" >/dev/null 2>&1; do sleep 5; done
echo "UP $(date)"

# ---- R1 (cold) ----
P1='The quick brown fox jumps over the lazy dog and then keeps on running through the meadow until it reaches the river where it finally stops to drink some water and rest for a while before continuing its journey home. After resting, the fox decides to take a different path back through the forest where the trees grow tall and the moss covers every stone.'

echo === R1 SMOKE ===
R1=$(P="$P1" python3 -c 'import json,os; print(json.dumps({"model":"Qwen/Qwen3-0.6B","prompt":os.environ["P"],"max_tokens":16,"temperature":0}))' \
  | curl -m 90 -sS -X POST "http://127.0.0.1:$KVBM_SINGLE_PORT/v1/completions" \
      -H "Content-Type: application/json" -d @-)
echo "$R1" | head -c 400; echo

# Give the offload pipeline time to push G1 → G2 in the background.
# The intra-pass onboard on R2 only fires if the G2 cache actually
# contains the matching blocks by then.
sleep 5

# ---- R2 (warm prefix; intra-pass onboard exercised here) ----
P2="$P1 The river flowed gently between the green hills and bright wildflowers swaying in the spring breeze, while birds sang above."

echo === R2 SMOKE ===
R2=$(P="$P2" python3 -c 'import json,os; print(json.dumps({"model":"Qwen/Qwen3-0.6B","prompt":os.environ["P"],"max_tokens":16,"temperature":0}))' \
  | curl -m 90 -sS -X POST "http://127.0.0.1:$KVBM_SINGLE_PORT/v1/completions" \
      -H "Content-Type: application/json" -d @-)
echo "$R2" | head -c 400; echo

sleep 2

# ---- Validation report ----
echo
echo "================================================================"
echo "  Intra-pass onboard validation"
echo "================================================================"

# Single integer count helper — see two-request-smoke.sh for rationale
# (avoids the `[: 0\n0` arithmetic bug from grep -c's exit-1-on-no-match).
grep_count() {
  local pattern=$1
  shift
  grep -aEc "$pattern" "$@" 2>/dev/null | awk -F: '{ sum += $NF } END { print sum+0 }'
}

# Engine-side start/complete pair (INFO; carries num_layers + num_blocks).
IPO_START=$(grep_count "Starting layer-wise onboard from G2 to G1" "$ROOT/vllm.log")
IPO_DONE=$(grep_count  "Layer-wise onboard complete - events recorded" "$ROOT/vllm.log")
echo "  engine start (G2→G1 layer-wise): $IPO_START"
echo "  engine complete                 : $IPO_DONE"

# Connector-side breadcrumb (debug-level; visible because RUST_LOG includes
# kvbm_connector=debug). Cross-check against engine-side counts.
IPO_CONN=$(grep_count "Starting intra-pass layer-wise onboard" "$ROOT/vllm.log")
echo "  connector breadcrumb            : $IPO_CONN"

# Surface the start line so the reader can eyeball num_layers + num_blocks.
echo "  start lines:"
grep -a "Starting layer-wise onboard from G2 to G1" "$ROOT/vllm.log" 2>/dev/null \
  | sed 's/\x1b\[[0-9;]*m//g' | head -5 | sed 's/^/    /' || true

# Phase-4b regression guard: under Universal mode the per-layer onboard
# routes through `dispatch_transform_kernel`; any FFI/kernel-launch failure
# surfaces here. Under operational mode this same call goes through the
# legacy CUDA executor and the grep below is informational.
IPO_TRANSFORM_ERRS=$(grep -aE "dispatch_transform_kernel.*(fail|launch failed|out of bounds)" \
  "$ROOT/vllm.log" 2>/dev/null | wc -l | tr -d ' ')
echo "  permute-kernel error lines      : $IPO_TRANSFORM_ERRS"

# Any catch-all kvbm ERROR lines in vllm.log (ignoring noise we know about).
echo
echo "-- ANY ERRORs across vllm.log --"
err_cnt=$(grep -aE "ERROR" "$ROOT/vllm.log" | sed 's/\x1b\[[0-9;]*m//g' \
  | grep -v "kvbm_audit\|UCX\|invalid configuration\|kernel_config" | wc -l)
echo "  vllm.log: $err_cnt error lines"

# ---- Decision ----
INTRA_PASS_ONBOARD_OK=1
if [ "$KVBM_ONBOARD_MODE" = "intra" ]; then
  if [ "$IPO_START" -lt 1 ]; then
    echo "  FAIL: KVBM_ONBOARD_MODE=intra but no 'Starting layer-wise onboard' in vllm.log (R2 warm) — connector did not produce intra_pass_load metadata"
    INTRA_PASS_ONBOARD_OK=0
  fi
  if [ "$IPO_DONE" -lt "$IPO_START" ]; then
    echo "  FAIL: $IPO_START starts but only $IPO_DONE completes — onboard panicked mid-loop?"
    INTRA_PASS_ONBOARD_OK=0
  fi
  if [ "$IPO_TRANSFORM_ERRS" -gt 0 ]; then
    echo "  FAIL: kernel transform errors during intra-pass onboard ($IPO_TRANSFORM_ERRS)"
    INTRA_PASS_ONBOARD_OK=0
  fi
else
  echo "  skipped (mode=$KVBM_ONBOARD_MODE; set KVBM_ONBOARD_MODE=intra to exercise this path)"
fi

if [ "$INTRA_PASS_ONBOARD_OK" -eq 1 ]; then
  echo "  intra-pass onboard: PASS"
else
  echo "  intra-pass onboard: FAIL"
fi

# Teardown.
pkill -f "vllm.entrypoints.openai" 2>/dev/null || true

if [ "$KVBM_ONBOARD_MODE" = "intra" ] && [ "$INTRA_PASS_ONBOARD_OK" -ne 1 ]; then
  echo
  echo "smoke: intra-pass onboard validation FAILED — see report above (logs: $ROOT/vllm.log)"
  exit 1
fi
echo
echo "smoke: done (logs: $ROOT/vllm.log)"
