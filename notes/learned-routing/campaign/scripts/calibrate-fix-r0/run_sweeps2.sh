#!/bin/bash
# fixer r0, round 2: finer sessions grids (existing points are result-cache hits)
set -u
cd <campaign-root>/runs/calibrate-fix-r0/scripts
export PYTHONDONTWRITEBYTECODE=1 LR_ROOT=<campaign-root> DYN_LOG=warn
PY=<worktree>/.venv/bin/python
$PY sweep.py eval ss-open --slots 20 > ../sweep3/eval_ss_open_fine.log 2>&1
$PY sweep.py eval ss-closed --slots 20 > ../sweep3/eval_ss_closed_fine.log 2>&1
$PY tsweep.py eval --slots 20 > ../sweep3/eval_tx_fine.log 2>&1
echo done > ../sweep3/sweeps2.done
