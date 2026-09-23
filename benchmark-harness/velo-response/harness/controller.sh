#!/usr/bin/env bash
set -euo pipefail
ROOT=/lustre/fsw/coreai_comparch_trtllm/jothomson/dynamo-velo-response-20260923
JOB=$(cat "$ROOT/control/job-id")
mapfile -t labels < <(python3 -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))))' "${1:-$ROOT/configs/order.json}")
for label in "${labels[@]}"; do
    printf '%s\n' "$label" > "$ROOT/control/current-condition"
    bash "$ROOT/harness/run_condition.sh" "$label" > "$ROOT/logs/$JOB-$label-controller.log" 2>&1
    printf '%s\n' "$JOB" > "$ROOT/control/$label-job-id"
    date -Is > "$ROOT/control/$label-COMPLETE"
done
date -Is > "$ROOT/control/RUNS_COMPLETE"
# Each run already checks complete exports and the harness acceptance result.
# Parse full exports after timing runs so offline analysis does not use their
# allocation window or overlap their load generator.
mapfile -t measured < <(python3 -c 'import json,sys; print("\n".join(x for x in json.load(open(sys.argv[1])) if x.startswith("r")))' "$ROOT/configs/order.json")
for label in "${measured[@]}"; do
    python3 "$ROOT/harness/analyze_campaign.py" "$label" > "$ROOT/logs/$JOB-$label-analysis.log"
    python3 "$ROOT/harness/analyze_packets.py" "$label"
    PYTHONPATH="$ROOT/harness" python3 -c 'import json,sys; from analyze_campaign import run_location; assert json.load(open(run_location(sys.argv[1])[1]/"matching-window-results.json"))["quality"]["accepted"], "run failed quality checks"' "$label"
done
date -Is > "$ROOT/control/ANALYSIS_COMPLETE"
