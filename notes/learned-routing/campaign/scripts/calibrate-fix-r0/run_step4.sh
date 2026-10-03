#!/bin/bash
# fixer r0: step 4 reference (default@defaults + round_robin x 8 CRN replicates) on lr-cells-v5
# train/val/noise (unchanged cells are result-cache hits), then the doubled-warm-up check.
set -u
cd <campaign-root>/runs/calibrate-fix-r0/scripts
export PYTHONDONTWRITEBYTECODE=1 LR_ROOT=<campaign-root> DYN_LOG=warn
PY=<worktree>/.venv/bin/python
$PY step4.py ../step4_r5 ../final_r5/train.jsonl ../final_r5/val.jsonl ../final_r5/noise.jsonl --repeats 8 --slots 20 > ../step4_r5/run.log 2>&1
$PY warm_check.py eval > ../warm_check/run.log 2>&1
echo done > ../step4_r5/chain.done
