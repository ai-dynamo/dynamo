#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One vLLM 0.24 TP2 worker (python -m dynamo.vllm) inside the job's container.
# Usage: worker.sh LOCAL_INDEX PLAN_DIR LOG_DIR
# The vLLM flags come from PLAN_DIR/engine_plan.json (plan.py mirrors CR/config/engine.json).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ $# -eq 3 ]] || die "usage: worker.sh LOCAL_INDEX PLAN_DIR LOG_DIR"
index=$1 plan_dir=$2 log_dir=$3
lr_container_env
lr_runtime_env

gpus="$(lr_worker_gpus "$index")"
cpus="$(lr_worker_cpus "$index")"
system_port="$(lr_worker_system_port "$index")"
kv_port="$(lr_worker_kv_port "$index")"
export CUDA_VISIBLE_DEVICES="$gpus"
export DYN_SYSTEM_PORT="$system_port"
export OMP_NUM_THREADS=8
export DYN_LOG="${LR_WORKER_LOG:-info}"

# NUL-separated vLLM flags from the frozen plan; the model path is node-local.
mapfile -d '' vllm_args < <("$LR_VENV/bin/python" - "$plan_dir/engine_plan.json" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
sys.stdout.write("\0".join(plan["vllm_args"]) + "\0")
PY
)
kv_events="{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:$kv_port\",\"enable_kv_cache_events\":true}"
cmd=(
  taskset -c "$cpus" "$LR_VENV/bin/python" -m dynamo.vllm
  --discovery-backend "$DYN_DISCOVERY_BACKEND" --request-plane tcp --response-plane tcp
  --event-plane zmq --namespace "$LR_NAMESPACE"
  --model "$LR_MODEL_DIR"
  "${vllm_args[@]}"
  --kv-events-config "$kv_events"
)
mkdir -p "$log_dir"
printf '%s\0' "${cmd[@]}" > "$log_dir/worker-$index.cmdline"
printf 'worker_index=%s host=%s pid=%s gpus=%s cpus=%s system_port=%s kv_port=%s\n' \
  "$index" "$(hostname)" "$$" "$gpus" "$cpus" "$system_port" "$kv_port"
exec "${cmd[@]}"
