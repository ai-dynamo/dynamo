#!/bin/bash
# fixer r0: ss-open re-sweep then sessions-open transform re-sweep (default + round_robin, K = 2)
set -u
cd <campaign-root>/runs/calibrate-fix-r0/scripts
export PYTHONDONTWRITEBYTECODE=1 LR_ROOT=<campaign-root> DYN_LOG=warn
PY=<worktree>/.venv/bin/python
$PY sweep.py eval ss-open --slots 20 > ../sweep3/eval_ss_open.log 2>&1
$PY tsweep.py eval --slots 20 > ../sweep3/eval_tx.log 2>&1
echo done > ../sweep3/sweeps.done
