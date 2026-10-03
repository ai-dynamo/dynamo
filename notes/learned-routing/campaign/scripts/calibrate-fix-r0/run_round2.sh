#!/bin/bash
# fixer r0 round 2: two-lifetime warm-up. Re-sweep ss-open + session-open transforms, then the
# doubled-history (W=2 lifetimes vs 2W) stationarity sweep.
set -u
cd <campaign-root>/runs/calibrate-fix-r0/scripts
export PYTHONDONTWRITEBYTECODE=1 LR_ROOT=<campaign-root> DYN_LOG=warn
PY=<worktree>/.venv/bin/python
mkdir -p ../sweep4 ../hist_sweep2
$PY sweep.py build ss-open > ../sweep4/build_ss_open.log 2>&1
$PY tsweep.py build > ../sweep4/build_tx.log 2>&1
$PY hist_sweep.py build > ../hist_sweep2/build.log 2>&1
$PY sweep.py eval ss-open --slots 20 > ../sweep4/eval_ss_open.log 2>&1
$PY tsweep.py eval --slots 20 > ../sweep4/eval_tx.log 2>&1
$PY hist_sweep.py eval > ../hist_sweep2/eval.log 2>&1
echo done > ../sweep4/round2.done
