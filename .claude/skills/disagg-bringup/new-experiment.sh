#!/bin/bash
# Mint a fresh experiment directory under
#   $KVBM_EXPERIMENTS_DIR/<timestamp>-<label>/
# (default $KVBM_EXPERIMENTS_DIR = /tmp/kvbm-experiments) and print
# its absolute path.
#
# Used by smoke harnesses to put hub.log / prefill.log / decode.log
# / trace.html in one directory keyed by run.
#
# Usage:
#   ROOT=$(bash new-experiment.sh smoke-cd)
#   echo "logs go in $ROOT"
set -eu

LABEL="${1:-exp}"
TS="$(date +%Y%m%d-%H%M%S)"
BASE=${KVBM_EXPERIMENTS_DIR:-/tmp/kvbm-experiments}
ROOT="$BASE/${TS}-${LABEL}"
mkdir -p "$ROOT"
echo "$ROOT"
