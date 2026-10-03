#!/bin/bash
# fixer r0 final: step 4 reference on lr-cells-v5 candidates (final_r6) + doubled-warm-up check
set -u
cd <campaign-root>/runs/calibrate-fix-r0/scripts
export PYTHONDONTWRITEBYTECODE=1 LR_ROOT=<campaign-root> DYN_LOG=warn WARM_CHECK_DIR=warm_check_r2
PY=<worktree>/.venv/bin/python
$PY step4.py ../step4_r6 ../final_r6/train.jsonl ../final_r6/val.jsonl ../final_r6/noise.jsonl --repeats 8 --slots 20 > ../step4_r6/run.log 2>&1
$PY warm_check.py build ../final_r6 > ../warm_check_r2/build.log 2>&1
$PY warm_check.py eval > ../warm_check_r2/run.log 2>&1
echo done > ../step4_r6/chain.done
