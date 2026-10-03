#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The Dynamo frontend with its embedded aggregated KV router, configured for one policy.
# Usage: frontend.sh PLAN_DIR POLICY_SLUG LOG_DIR NUM_WORKERS
# Router flags come only from PLAN_DIR/policies/POLICY_SLUG/policy_plan.json; plan.py proved that
# they give the frontend the same KvRouterConfig kwargs that replay passes for that policy.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ $# -eq 4 ]] || die "usage: frontend.sh PLAN_DIR POLICY_SLUG LOG_DIR NUM_WORKERS"
plan_dir=$1 slug=$2 log_dir=$3 num_workers=$4
policy_dir="$plan_dir/policies/$slug"
[[ -f "$policy_dir/policy_plan.json" ]] || die "no policy plan at $policy_dir"
lr_container_env
lr_runtime_env
# The frontend never needs a GPU and must not bind a worker's system port.
export CUDA_VISIBLE_DEVICES=
unset DYN_SYSTEM_PORT
export DYN_LOG="${LR_FRONTEND_LOG:-info}"
python_bin="${LR_FRONTEND_PYTHON:-$LR_VENV/bin/python}"

mapfile -d '' router_args < <("$python_bin" - "$policy_dir" <<'PY'
import hashlib, json, sys
from pathlib import Path
policy_dir = Path(sys.argv[1])
plan = json.loads((policy_dir / "policy_plan.json").read_text())
yaml_path = policy_dir / "policy.yaml"
if plan["policy_yaml"] is not None:
    digest = hashlib.sha256(yaml_path.read_bytes()).hexdigest()
    if digest != plan["policy_yaml_sha256"]:
        sys.exit(f"error: {yaml_path} sha256 {digest} != plan {plan['policy_yaml_sha256']}")
flags = [str(yaml_path) if f == "{policy_yaml}" else f for f in plan["frontend_flags"]]
sys.stdout.write("\0".join(flags) + "\0")
PY
)
block_size="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["block_size"])' "$plan_dir/engine_plan.json")"
mkdir -p "$log_dir"
cmd=(
  taskset -c "$LR_FRONTEND_CPUS" "$python_bin" -m dynamo.frontend
  --discovery-backend "$DYN_DISCOVERY_BACKEND" --request-plane tcp --response-plane tcp
  --event-plane zmq --namespace "$LR_NAMESPACE"
  --http-host 0.0.0.0 --http-port "$LR_HTTP_PORT"
  "${router_args[@]}"
  --kv-cache-block-size "$block_size"
  --router-min-initial-workers "$num_workers"
  --dump-config-to "$log_dir/frontend-config.json"
)
printf '%s\0' "${cmd[@]}" > "$log_dir/frontend.cmdline"
printf 'frontend host=%s pid=%s policy=%s cpus=%s port=%s\n' \
  "$(hostname)" "$$" "$slug" "$LR_FRONTEND_CPUS" "$LR_HTTP_PORT"
exec "${cmd[@]}"
