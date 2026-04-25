#!/usr/bin/env bash
# DIS-1850 — run every parity harness and report a single PASS/FAIL summary.
# Usage:  ./_run_all.sh
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

HARNESSES=(
    "_parity_check.py"            # 4 V4 golden fixtures
    "_parity_check_v32.py"        # 3 V3.2 golden fixtures
    "_gap_fixtures.py"            # 9 synthetic V4 gap fixtures
    "_inline_form_check.py"       # 16 fixtures (golden + gap) on inline form
)

total_run=0
total_fail=0

for h in "${HARNESSES[@]}"; do
    if [[ ! -f "$h" ]]; then
        echo "SKIP $h (not found)"
        continue
    fi
    out=$(python3 "$h" 2>&1)
    if [[ -n "$out" ]]; then
        # Count PASS / FAIL occurrences in the output.
        passes=$(grep -c -E '\bPASS\b' <<<"$out" || true)
        fails=$(grep -c -E '\bFAIL\b' <<<"$out" || true)
        run=$((passes + fails))
        total_run=$((total_run + run))
        total_fail=$((total_fail + fails))
        if [[ "$fails" -eq 0 && "$run" -gt 0 ]]; then
            printf '  %-30s  %d/%d\n' "$h" "$passes" "$run"
        else
            printf '  %-30s  %d/%d  FAIL\n' "$h" "$passes" "$run"
            echo "$out" | grep -E '(FAIL|RENDER_ERROR)' | sed 's/^/    /'
        fi
    fi
done

echo "----"
if [[ "$total_fail" -eq 0 ]]; then
    echo "ALL ${total_run}/${total_run} PASS"
    exit 0
else
    echo "$((total_run - total_fail))/${total_run} pass, ${total_fail} fail"
    exit 1
fi
