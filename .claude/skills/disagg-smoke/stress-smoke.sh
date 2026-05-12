#!/bin/bash
# Stress wrapper around two-request-smoke.sh — runs N consecutive iterations
# against a warm session and reports pass/fail tally. Surfaces intermittent
# races (e.g. NIXL double-load TOCTOU, observer-before-attach timing) that
# a single smoke run can't catch.
#
# Each iter mints its own experiment dir under $KVBM_EXPERIMENTS_DIR so the
# failing one can be grepped after the fact.
#
# Env knobs:
#   KVBM_STRESS_ITERS         (default: 5)   number of iterations to run
#   KVBM_STRESS_INTER_SLEEP   (default: 5)   seconds between iters (lets
#                                            prior NIXL sessions drain)
#   KVBM_STRESS_LABEL_PREFIX  (default: tp1-stress) experiment label prefix
#
# Usage:
#   bash stress-smoke.sh             # 5x against the legacy single-leader path
#   KVBM_STRESS_ITERS=10 bash stress-smoke.sh
#
# Honors all envs that two-request-smoke.sh honors (KVBM_DISAGG_LEADER, etc).
# Stops early on first failure if KVBM_STRESS_STOP_ON_FAIL=1.

set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ITERS=${KVBM_STRESS_ITERS:-5}
INTER_SLEEP=${KVBM_STRESS_INTER_SLEEP:-5}
LABEL_PREFIX=${KVBM_STRESS_LABEL_PREFIX:-tp1-stress}
STOP_ON_FAIL=${KVBM_STRESS_STOP_ON_FAIL:-0}

# Track results across iters
declare -a RESULTS=()
declare -a FAIL_EXPS=()
PASS_COUNT=0
FAIL_COUNT=0

echo "=== stress-smoke: $ITERS iterations, ${INTER_SLEEP}s inter-iter sleep ==="
echo "=== started @ $(date -u +%FT%TZ) ==="

for i in $(seq 1 "$ITERS"); do
    echo ""
    echo "=== iter $i/$ITERS @ $(date -u +%FT%TZ) ==="

    # Each iter gets a unique experiment label. The smoke creates a
    # timestamped dir under $KVBM_EXPERIMENTS_DIR; we just label it so
    # the failing iter can be located by name later.
    export KVBM_EXPERIMENT_LABEL="${LABEL_PREFIX}-iter${i}"

    bash "$HERE/two-request-smoke.sh"
    rc=$?

    RESULTS+=("iter${i}=$rc")
    if [ "$rc" -eq 0 ]; then
        PASS_COUNT=$((PASS_COUNT + 1))
        echo "=== iter $i PASS ==="
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        # Find the most recent experiment dir matching our label
        if [ -n "${KVBM_EXPERIMENTS_DIR:-}" ]; then
            FAIL_EXP=$(ls -dt "${KVBM_EXPERIMENTS_DIR}"/*"${LABEL_PREFIX}-iter${i}" 2>/dev/null | head -1 || true)
            [ -n "$FAIL_EXP" ] && FAIL_EXPS+=("$FAIL_EXP")
            echo "=== iter $i FAIL rc=$rc (experiment: ${FAIL_EXP:-unknown}) ==="
        else
            echo "=== iter $i FAIL rc=$rc ==="
        fi
        if [ "$STOP_ON_FAIL" = "1" ]; then
            echo "=== KVBM_STRESS_STOP_ON_FAIL=1, halting after first failure ==="
            break
        fi
    fi

    # Brief pause so NIXL/peer state from prior iter has time to drain.
    # Without this, fast-back-to-back iters can hit "Remote worker already
    # loaded" purely from the slow shutdown of the previous run.
    if [ "$i" -lt "$ITERS" ]; then
        sleep "$INTER_SLEEP"
    fi
done

echo ""
echo "=== STRESS-DONE @ $(date -u +%FT%TZ) ==="
echo "results: ${RESULTS[*]}"
echo "pass=$PASS_COUNT  fail=$FAIL_COUNT  total=$ITERS"
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo ""
    echo "FAILING EXPERIMENTS (grep decode.log for 'CD request failed' + chain):"
    for e in "${FAIL_EXPS[@]}"; do
        echo "  $e"
    done
fi

# Exit non-zero if ANY iter failed — caller scripts can chain on this.
[ "$FAIL_COUNT" -eq 0 ] && exit 0 || exit 1
