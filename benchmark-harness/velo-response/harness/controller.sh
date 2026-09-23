#!/usr/bin/env bash
set -euo pipefail
ROOT=/lustre/fsw/coreai_comparch_trtllm/jothomson/dynamo-velo-response-20260923
mapfile -t labels < <(python3 -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))))' "$ROOT/configs/order.json")
for label in "${labels[@]}"; do
    printf '%s\n' "$label" > "$ROOT/control/current-condition"
    bash "$ROOT/harness/run_condition.sh" "$label" > "$ROOT/logs/$label-controller.log" 2>&1
    python3 "$ROOT/harness/analyze_campaign.py" "$label" > "$ROOT/logs/$label-analysis.log"
    python3 "$ROOT/harness/analyze_packets.py" "$label"
    python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["quality"]["accepted"], "run failed quality checks"' "$ROOT/results/$(cat "$ROOT/control/job-id")-main-$label/ablations/main-$label/matching-window-results.json"
    date -Is > "$ROOT/control/$label-COMPLETE"
done
date -Is > "$ROOT/control/RUNS_COMPLETE"
