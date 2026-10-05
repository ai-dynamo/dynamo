#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# AIPerf payload for deploy/job_node.sh (A13 GPU smoke and finalist runs). It runs inside the
# vLLM container on the head node, pinned to LR_CLIENT_CPUS, after the cold reset and a fresh
# frontend. For each generator output under LR_SMOKE_INPUTS, in LR_SMOKE_RUNS order, it runs AIPerf
# with exactly the manifest's argv (run_aiperf.py) and brackets the run with metric scrapes; a
# background scraper records the frontend and every worker every LR_SCRAPE_INTERVAL_S seconds.
#
# From job_node.sh: LR_ENDPOINT, LR_PAYLOAD_DIR, LR_NUM_WORKERS, LR_POLICY_SLUG; in pairs mode
# also LR_SMOKE_RUNS (the pair's one input), LR_RUN_TIMEOUT_S and LR_PAIR_LABEL.
# From the job env file (submit.sh --env):
#   LR_SMOKE_INPUTS     directory of generator outputs, one subdirectory per run (manifest.json)
#   LR_AIPERF_ENV       AIPerf 0.13.0 environment; its python also runs the stdlib tools here
#   LR_SMOKE_RUNS       space-separated subdirectories to run, in order (default "cell idle")
#   LR_RUN_TIMEOUT_S    per-run AIPerf timeout (default 2400)
#   LR_SCRAPE_INTERVAL_S  (default 10)
# Outputs in LR_PAYLOAD_DIR: <run>/ (AIPerf artifacts), <run>.run.json, <run>.log,
# metrics-series.jsonl, metrics-marks.jsonl, payload-env.txt, payload-summary.json.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${LR_ENDPOINT:?}" "${LR_PAYLOAD_DIR:?}" "${LR_SMOKE_INPUTS:?}" "${LR_AIPERF_ENV:?}"
py="$LR_AIPERF_ENV/bin/python"
out="$LR_PAYLOAD_DIR"
mkdir -p "$out"
# AIPerf and Python caches go to node-local disk, not the shared home.
node_root="${LR_NODE_ROOT:-${LR_NODE_BASE:-/raid/scratch}/${LR_NODE_TAG:+$LR_NODE_TAG-}lr-${SLURM_JOB_ID:-nojob}}"
export HOME="$node_root/payload-home"
export XDG_CACHE_HOME="$HOME/.cache" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$HOME"
ulimit -n "$(ulimit -Hn)" 2> /dev/null || true

host="${LR_ENDPOINT#*://}"
host="${host%%:*}"
targets=(--target "frontend=$LR_ENDPOINT/metrics")
for ((i = 0; i < ${LR_NUM_WORKERS:-4}; i++)); do
  targets+=(--target "w$i=http://$host:$((19100 + i))/metrics")
done

{
  date -u +%FT%TZ
  echo "host=$(hostname) endpoint=$LR_ENDPOINT policy=${LR_POLICY_SLUG:-} workers=${LR_NUM_WORKERS:-}"
  echo "pair=${LR_PAIR_LABEL:-} runs=${LR_SMOKE_RUNS:-cell idle} run_timeout_s=${LR_RUN_TIMEOUT_S:-2400}"
  echo "nproc=$(nproc) affinity=$(taskset -pc $$ 2> /dev/null | awk -F': ' '{print $2}')"
  echo "ulimit_n=$(ulimit -n)"
  "$py" --version
  "$LR_AIPERF_ENV/bin/aiperf" --version 2>&1 | tail -1
} > "$out/payload-env.txt" 2>&1

"$py" "$here/scrape_metrics.py" loop "${targets[@]}" --interval "${LR_SCRAPE_INTERVAL_S:-10}" \
  --out "$out/metrics-series.jsonl" > "$out/scraper.log" 2>&1 &
scraper=$!

declare -A status=()
failed=0
for run in ${LR_SMOKE_RUNS:-cell idle}; do
  inputs="$LR_SMOKE_INPUTS/$run"
  if [[ ! -s "$inputs/manifest.json" ]]; then
    echo "missing $inputs/manifest.json" >&2
    status[$run]=missing
    failed=1
    continue
  fi
  "$py" "$here/scrape_metrics.py" once "${targets[@]}" --tag "$run-start" --out "$out/metrics-marks.jsonl"
  "$py" "$here/run_aiperf.py" --inputs "$inputs" --url "$LR_ENDPOINT" --artifact-dir "$out/$run" \
    --aiperf "$LR_AIPERF_ENV/bin/aiperf" --verify-input --timeout-s "${LR_RUN_TIMEOUT_S:-2400}" \
    > "$out/$run.log" 2>&1
  rc=$?
  "$py" "$here/scrape_metrics.py" once "${targets[@]}" --tag "$run-end" --out "$out/metrics-marks.jsonl"
  status[$run]=$rc
  [[ $rc -eq 0 ]] || failed=1
done

kill -TERM "$scraper" 2> /dev/null || true
wait "$scraper" 2> /dev/null || true
{
  printf '{"ended_utc": "%s", "runs": {' "$(date -u +%FT%TZ)"
  sep=""
  for run in "${!status[@]}"; do
    printf '%s"%s": "%s"' "$sep" "$run" "${status[$run]}"
    sep=", "
  done
  printf '}}\n'
} > "$out/payload-summary.json"
exit "$failed"
