#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# CPU-only preflight of the deploy scripts on a workstation: the real frontend.sh and
# health_check.py against mocker workers (no vLLM, no GPUs). It exercises plan -> frontend flags,
# discovery and registration, token-ID completions with forced OSL, KV events reaching the router,
# prefix reuse routing and policy evidence for each planned policy. The cold reset needs vLLM's
# /engine/flush_cache and is covered only on the GPU node.
#
# Usage: tests/local_mocker_smoke.sh PLAN_DIR OUT_DIR [POLICY_SLUG...]
#   LR_WT (worktree; default: derived), LR_TOKENIZER_DIR (a Qwen3 tokenizer snapshot),
#   LR_MOCK_WORKERS (default 2)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
deploy="$(dirname "$here")"
[[ $# -ge 2 ]] || { echo "usage: $0 PLAN_DIR OUT_DIR [POLICY_SLUG...]" >&2; exit 2; }
plan_dir=$1 out=$2
shift 2
wt="${LR_WT:-$(cd "$deploy/../../../.." && pwd)}"
py="$wt/.venv/bin/python"
tokenizer="${LR_TOKENIZER_DIR:-$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/ | head -1)}"
workers="${LR_MOCK_WORKERS:-2}"
[[ -d "$tokenizer" ]] || { echo "error: set LR_TOKENIZER_DIR to a Qwen3 tokenizer snapshot" >&2; exit 2; }
[[ ! -e "$out" ]] || { echo "error: $out exists" >&2; exit 2; }
mkdir -p "$out"
if [[ $# -eq 0 ]]; then
  mapfile -t slugs < <(ls "$plan_dir/policies")
else
  slugs=("$@")
fi

export LR_VENV="$wt/.venv" LR_NODE_ROOT="$out/node" LR_CACHE_ROOT="$out/cache"
export LR_NAMESPACE="lrlocal-$$" LR_HTTP_PORT="${LR_HTTP_PORT:-18431}"
export LR_FRONTEND_CPUS="0-$(($(nproc) - 1))" SLURM_JOB_ID="local$$"
source "$deploy/common.sh"
lr_runtime_env
block_size="$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["block_size"])' "$plan_dir/engine_plan.json")"

pids=()
stop_all() {
  local pid
  for pid in "${pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap stop_all EXIT

worker_args=()
for ((i = 0; i < workers; i++)); do
  port=$((19400 + i))
  DYN_SYSTEM_PORT=$port DYN_LOG=info "$py" -m dynamo.mocker \
    --discovery-backend file --request-plane tcp --response-plane tcp --event-plane zmq \
    --endpoint "dyn://$LR_NAMESPACE.backend.generate" \
    --model-path "$tokenizer" --model-name Qwen/Qwen3-32B \
    --block-size "$block_size" --num-gpu-blocks-override 4096 --max-num-seqs 256 \
    --max-num-batched-tokens 8192 --enable-prefix-caching --speedup-ratio 20 \
    > "$out/mocker-$i.log" 2>&1 &
  pids+=("$!")
  worker_args+=(--worker "127.0.0.1:$port")
done

status=0
for slug in "${slugs[@]}"; do
  run="$out/$slug"
  mkdir -p "$run"
  LR_FRONTEND_PYTHON="$py" bash "$deploy/frontend.sh" "$plan_dir" "$slug" "$run" "$workers" \
    > "$run/frontend.log" 2>&1 &
  fe=$!
  pids+=("$fe")
  common=(--frontend "http://127.0.0.1:$LR_HTTP_PORT" "${worker_args[@]}"
    --namespace "$LR_NAMESPACE" --block-size "$block_size")
  if ! "$py" "$deploy/health_check.py" wait "${common[@]}" --expect "$workers" --timeout 120 \
    --out "$run/wait.json" \
    || ! "$py" "$deploy/health_check.py" smoke "${common[@]}" \
      --policy-plan "$plan_dir/policies/$slug/policy_plan.json" \
      --config-dump "$run/frontend-config.json" --out "$run/smoke.json"; then
    status=1
  fi
  kill -TERM "$fe"
  wait "$fe" || true
done
exit "$status"
