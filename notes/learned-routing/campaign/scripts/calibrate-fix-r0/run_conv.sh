#!/bin/bash
cd <campaign-root>/runs/calibrate-fix-r0/scripts
export PYTHONDONTWRITEBYTECODE=1 LR_ROOT=<campaign-root> DYN_LOG=warn
PY=<worktree>/.venv/bin/python
mkdir -p ../conv_check
$PY conv_check.py build > ../conv_check/build.log 2>&1
$PY conv_check.py eval > ../conv_check/eval.log 2>&1
echo done > ../conv_check/chain.done
