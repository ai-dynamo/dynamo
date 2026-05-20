#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# kvbm-v2-agg-matrix — run the 4-cell (G1 LW/FC × G2 Op/Universal) agg
# determinism matrix on Qwen3-0.6B against the local sandbox venv.
#
# Usage:
#   run.sh                       # all 4 cells, default iterations (10)
#   run.sh -k g2uni-g1fc         # single cell via pytest -k substring match
#   run.sh -- --tb=long          # extra args after `--` go to pytest
#
# Env vars:
#   KVBM_REPO                    (default: derived from this script's
#                                   location — the worktree/main checkout
#                                   that owns this skill). Set explicitly
#                                   only when you intentionally want to
#                                   point at a different checkout.
#   KVBM_VENV                    (default: $KVBM_REPO/.sandbox)
#   KVBM_MATRIX_MAX_ITERATIONS   (default: 40) — per-phase iteration count.
#                                   Conservative calibration value; ratchet
#                                   up after the first run reports actual
#                                   G2 utilization per cell. Aim for ~75%
#                                   utilization on the heaviest cell.
#   KVBM_MATRIX_CPU_BLOCKS       (default: 4000) — G2 host blocks
#   KVBM_MATRIX_GPU_BLOCKS       (default: 2048)
#   KVBM_MATRIX_ONBOARD_MODE     (default: inter — production default)
#   KVBM_SERVER_START_TIMEOUT    (default: 300)
#
# Output:
#   - pytest verbose output streamed live.
#   - After completion, prints a 4-cell verdict matrix sourced from the
#     per-cell `[matrix]` log lines: pass/fail × G2 utilization per cell.
#   - Exit 0 iff every parametrized cell passed.
set -euo pipefail

# Derive the repo root from this script's location so a skill invoked
# from a worktree executes against THAT worktree's test file and
# bindings — not against a stale main checkout. Layout:
#   $REPO_ROOT/.claude/skills/kvbm-v2-agg-matrix/run.sh
# So the repo root is three parent directories up.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_DYNAMO=$(cd "$SCRIPT_DIR/../../.." && pwd)
DYNAMO=${KVBM_REPO:-$DEFAULT_DYNAMO}
VENV=${KVBM_VENV:-"$DYNAMO/.sandbox"}
TEST_FILE="tests/kvbm_integration/test_determinism_agg_v2_matrix.py"

# Sanity-check: the test file MUST exist at the resolved repo. Otherwise
# we'd run pytest against a checkout that doesn't have the matrix test
# (e.g., user explicitly pointed KVBM_REPO at a branch that lacks it)
# and silently report "no tests collected".
if [[ ! -f "$DYNAMO/$TEST_FILE" ]]; then
  echo "[matrix] FAIL: $TEST_FILE not found in resolved repo $DYNAMO" >&2
  echo "[matrix]   (script lives at: $SCRIPT_DIR)" >&2
  echo "[matrix]   (default repo derived as: $DEFAULT_DYNAMO)" >&2
  echo "[matrix]   (KVBM_REPO override: ${KVBM_REPO:-<unset>})" >&2
  echo "[matrix] If you intentionally targeted a different checkout, make sure" >&2
  echo "[matrix] the matrix test exists there. Otherwise unset KVBM_REPO." >&2
  exit 2
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[matrix] FAIL: sandbox venv not present at $VENV/bin/python" >&2
  echo "[matrix] run /dynamo:kvbm:sandbox-venv first" >&2
  exit 2
fi
if ! "$VENV/bin/python" -c "import kvbm" 2>/dev/null; then
  echo "[matrix] FAIL: kvbm not importable from $VENV" >&2
  echo "[matrix] run /dynamo:kvbm:maturin-dev to rebuild bindings" >&2
  exit 2
fi

# Put the venv's bin on PATH so the test's child subprocess can find `vllm`.
# The matrix spawns `vllm serve …` as a BARE command; invoking
# "$VENV/bin/pytest" alone does NOT add $VENV/bin to PATH, so without this
# every cell errors at setup with `FileNotFoundError: 'vllm'` (2026-05-19).
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

# GB10 (Spark) is a unified-memory box: ~20 GiB of the 120 GiB is permanently
# held by the OS/desktop, so vLLM's default `--gpu-memory-utilization 0.9`
# (≈108 GiB) exceeds free memory and every cell dies at init with
# "Free memory … less than desired GPU memory utilization" (2026-05-19).
# KV-cache size is pinned by KVBM_MATRIX_GPU_BLOCKS (--num-gpu-blocks-override),
# not by GMU, so a lower ceiling is harmless for a correctness run. Default to
# a GB10-safe 0.7; override KVBM_GPU_MEMORY_UTILIZATION for other hardware.
export KVBM_GPU_MEMORY_UTILIZATION=${KVBM_GPU_MEMORY_UTILIZATION:-0.7}

cd "$DYNAMO"

# Parse args: support -k, --, and pass-through.
PYTEST_K=""
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -k)
      PYTEST_K="$2"; shift 2;;
    --)
      shift; EXTRA+=("$@"); break;;
    *)
      EXTRA+=("$1"); shift;;
  esac
done

LOG=$(mktemp -t kvbm-matrix-XXXXXX.log)
trap 'rm -f "$LOG"' EXIT

# -rA emits an end-of-session summary (`PASSED <nodeid>` lines) AFTER all
# server stdout, so the per-cell verdict parser below has un-interleaved
# status lines to match. Without it, `-s` glues pytest's inline `PASSED`
# token onto the next server's stdout and the parser sees zero passes.
CMD=("$VENV/bin/pytest" "$TEST_FILE" -v -s -rA --tb=short)
if [[ -n "$PYTEST_K" ]]; then
  CMD+=(-k "$PYTEST_K")
fi
CMD+=("${EXTRA[@]+"${EXTRA[@]}"}")

echo "[matrix] resolved repo: $DYNAMO"
echo "[matrix] running: ${CMD[*]}"
echo "[matrix] env: KVBM_MATRIX_MAX_ITERATIONS=${KVBM_MATRIX_MAX_ITERATIONS:-40} \
KVBM_MATRIX_CPU_BLOCKS=${KVBM_MATRIX_CPU_BLOCKS:-4000} \
KVBM_MATRIX_ONBOARD_MODE=${KVBM_MATRIX_ONBOARD_MODE:-inter}"

# Run pytest; capture for verdict-extraction. PIPESTATUS used because tee
# would otherwise mask pytest's exit status.
#
# `set +e` is load-bearing: under `set -euo pipefail`, any pytest failure
# (which is the common case — that's what we're here to diagnose) makes
# the pipeline exit non-zero, `set -e` triggers at the pipeline, and the
# script dies BEFORE reaching the verdict-matrix loop. We disable -e for
# the pipeline so we always reach the per-cell verdict reporting; the
# pipeline's real exit status survives via PIPESTATUS.
set +e
"${CMD[@]}" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e

echo
echo "================================================================"
echo "[matrix] per-cell verdict matrix"
echo "================================================================"

# Pull per-cell pass/fail from the pytest result lines + G2 util from our prints.
EXPECTED_CELLS=(
  "v2-Qwen3-0.6B-${KVBM_MATRIX_ONBOARD_MODE:-inter}-g2ope-g1lw"
  "v2-Qwen3-0.6B-${KVBM_MATRIX_ONBOARD_MODE:-inter}-g2ope-g1fc"
  "v2-Qwen3-0.6B-${KVBM_MATRIX_ONBOARD_MODE:-inter}-g2uni-g1lw"
  "v2-Qwen3-0.6B-${KVBM_MATRIX_ONBOARD_MODE:-inter}-g2uni-g1fc"
)

FAIL_COUNT=0
for cell in "${EXPECTED_CELLS[@]}"; do
  # pytest result line looks like: tests/...::TestX::test_y[<cell-id>] PASSED
  #
  # `|| true` is load-bearing: under `set -euo pipefail`, a grep that
  # finds no match exits 1, pipefail propagates the nonzero exit, the
  # $() assignment fails, and `set -e` kills the whole script. That
  # happens on EVERY filtered run (-k narrows pytest, so 3 of 4 cells
  # have no match in the log), causing the script to die on the first
  # UNRUN cell and never print the verdict matrix. Trailing `|| true`
  # turns the empty-match case into a clean empty `verdict` string.
  # Prefer the -rA end-of-session summary line ("PASSED <nodeid>[cell]"),
  # which is printed after all server stdout and so is not interleaved.
  # Fall back to the inline verbose form ("...[cell] PASSED") for safety.
  verdict=$(grep -E "^(PASSED|FAILED|SKIPPED|ERROR) .*\[${cell}\]" "$LOG" \
    | tail -1 | grep -oE "^(PASSED|FAILED|SKIPPED|ERROR)" | head -1 || true)
  if [[ -z "$verdict" ]]; then
    verdict=$(grep -E "\[${cell}\][[:space:]]+(PASSED|FAILED|SKIPPED|ERROR)" "$LOG" \
      | tail -1 | grep -oE "(PASSED|FAILED|SKIPPED|ERROR)" | head -1 || true)
  fi
  if [[ -z "$verdict" ]]; then
    verdict="UNRUN"
  fi
  # G2 utilization printed by the test body.
  util=$(grep -E "\[matrix\] ${cell}: G2 offload total" "$LOG" | tail -1 \
    | sed -E 's/.*= ([0-9.]+%).*/\1/' || true)
  [[ -z "$util" ]] && util="--"
  printf "  %-44s %-9s G2 util %s\n" "$cell" "$verdict" "$util"
  if [[ "$verdict" != "PASSED" && "$verdict" != "SKIPPED" && "$verdict" != "UNRUN" ]]; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
  # UNRUN counts as fail when no -k filter was used (cells were filtered out
  # intentionally if -k was given).
  if [[ -z "$PYTEST_K" && "$verdict" == "UNRUN" ]]; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
done

echo "================================================================"
if [[ "$RC" -eq 0 && "$FAIL_COUNT" -eq 0 ]]; then
  echo "[matrix] PASS — all $(if [[ -n "$PYTEST_K" ]]; then echo "filtered"; else echo "4"; fi) cells passed"
  exit 0
else
  echo "[matrix] FAIL — pytest rc=$RC, $FAIL_COUNT cell(s) failed/unrun" >&2
  echo "[matrix] full log: $LOG (will be removed on shell exit)" >&2
  # Don't auto-remove on fail — keep the trap from firing the cleanup.
  trap - EXIT
  echo "[matrix] log preserved: $LOG" >&2
  exit 1
fi
