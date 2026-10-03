#!/bin/bash
cd <campaign-root>/runs/calibrate-fix-r0/scripts
export PYTHONDONTWRITEBYTECODE=1 LR_ROOT=<campaign-root> DYN_LOG=warn
PY=<worktree>/.venv/bin/python
mkdir -p ../hist_sweep
$PY hist_sweep.py build > ../hist_sweep/build.log 2>&1
$PY hist_sweep.py eval > ../hist_sweep/eval.log 2>&1
echo done > ../hist_sweep/chain.done
