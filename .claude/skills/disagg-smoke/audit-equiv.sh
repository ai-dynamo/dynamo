#!/bin/bash
# Audit-trace equivalence: run the two-request smoke twice (legacy
# leader, then unified leader), and run audit_diff on the per-side
# kvbm_audit streams to assert equivalence.
#
# Used to prove the UnifiedDisaggLeader is behaviorally equivalent
# to the role-dispatching ConditionalDisaggLeader under live traffic.
#
# Usage: bash audit-equiv.sh
set -eu

DYNAMO=/home/ryan/repos/dynamo
SKILL=$DYNAMO/.claude/skills/disagg-smoke
SKILL_BRINGUP=$DYNAMO/.claude/skills/disagg-bringup
DIFF=$DYNAMO/target/debug/audit_diff

if [ ! -x "$DIFF" ]; then
    echo "audit_diff binary missing at $DIFF"
    echo "build via: cargo build -p kvbm-connector --bin audit_diff"
    exit 2
fi

LEGACY_ROOT=$(bash $SKILL_BRINGUP/new-experiment.sh two-request-legacy)
UNIFIED_ROOT=$(bash $SKILL_BRINGUP/new-experiment.sh two-request-unified)

echo "=== legacy phase ==="
KVBM_DISAGG_LEADER=legacy KVBM_EXPERIMENT_LABEL=two-request-legacy \
    bash $SKILL/two-request-smoke.sh "$LEGACY_ROOT"

echo
echo "=== unified phase ==="
KVBM_DISAGG_LEADER=unified KVBM_EXPERIMENT_LABEL=two-request-unified \
    bash $SKILL/two-request-smoke.sh "$UNIFIED_ROOT"

echo
echo "================================================================"
echo "  Audit equivalence diff"
echo "================================================================"

# Per-side comparison (cross-side mixing would be non-deterministic).
# Filter the background heartbeat gauge — leader-independent.
FILTER="--filter-prefixes session_factory_active_gauge --normalize-request-ids"

PREFILL_RC=0
DECODE_RC=0

echo
echo "=== PREFILL diff ==="
"$DIFF" --legacy "$LEGACY_ROOT/prefill.log" --unified "$UNIFIED_ROOT/prefill.log" $FILTER || PREFILL_RC=$?

echo
echo "=== DECODE diff ==="
"$DIFF" --legacy "$LEGACY_ROOT/decode.log" --unified "$UNIFIED_ROOT/decode.log" $FILTER || DECODE_RC=$?

echo
echo "================================================================"
if [ "$PREFILL_RC" -eq 0 ] && [ "$DECODE_RC" -eq 0 ]; then
    echo "OK: leader-emitted audit streams equivalent on both sides."
    echo "  legacy : $LEGACY_ROOT"
    echo "  unified: $UNIFIED_ROOT"
    exit 0
else
    echo "DIVERGENCE: prefill_rc=$PREFILL_RC decode_rc=$DECODE_RC"
    echo "  legacy : $LEGACY_ROOT"
    echo "  unified: $UNIFIED_ROOT"
    exit 1
fi
