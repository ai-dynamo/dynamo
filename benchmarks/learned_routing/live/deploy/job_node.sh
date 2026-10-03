#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Slurm batch script for one live deployment on 1 (N = 4) or 2 (N = 8) whole 8 x H100 SXM nodes.
# Submitted by submit.sh, which passes LR_JOB_ENV (a file of LR_* settings) via --export.
#
# Stages, all inside this one job so nothing outlives it:
#   1. node prep on every node (hardware check, node-local root, image copy) and container create
#   2. in parallel: weights to node-local disk on every node | wheel build on the head node host
#   3. venv + verify_env.py inside the container (head node; the venv is on the shared filesystem)
#   4. [two-node] etcd on the head node; vLLM workers on every node (one step per node)
#   5. per policy in LR_POLICIES: frontend -> wait -> smoke -> cold reset -> stop; then, when
#      LR_PAYLOAD is set, a fresh frontend -> wait -> metrics snapshot -> payload -> snapshot -> stop
#   6. teardown (workers, etcd) and runtime-manifest.json
# The job exits when the last policy is done, so the allocation is never held idle.
set -euo pipefail
# Export every job setting: srun steps (build, workers, frontend, payload) re-source common.sh and
# need LR_COMMIT and the --env extras in their environment, not just in this shell.
if [[ -n "${LR_JOB_ENV:-}" ]]; then
  set -a
  source "$LR_JOB_ENV"
  set +a
fi
: "${LR_DEPLOY_DIR:?}" "${LR_COMMIT:?}" "${LR_PLAN_DIR:?}"
source "$LR_DEPLOY_DIR/common.sh"
lr_need LR_SHARED_ROOT LR_LIVE_ROOT LR_IMAGE_SQSH LR_MODEL_MASTER LR_TOOLCHAIN_ENV

mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
nnodes=${#nodes[@]}
head=${nodes[0]}
all_nodes="$(IFS=,; echo "${nodes[*]}")"
num_workers=$((nnodes * LR_WORKERS_PER_NODE))
cpn="${SLURM_CPUS_ON_NODE:-224}"
run_dir="$LR_RUNS_ROOT/$SLURM_JOB_ID-${LR_RUN_TAG:-live}"
mkdir -p "$run_dir/logs" "$run_dir/policies"
exec > >(tee -a "$run_dir/job.log") 2>&1
log "job=$SLURM_JOB_ID nodes=$all_nodes workers=$num_workers commit=${LR_COMMIT:0:12} plan=$LR_PLAN_DIR"
scontrol show job "$SLURM_JOB_ID" > "$run_dir/scontrol-job.txt"
cp "$LR_PLAN_DIR/PLAN.json" "$run_dir/PLAN.json"
[[ -n "${LR_JOB_ENV:-}" ]] && cp "$LR_JOB_ENV" "$run_dir/job.env"

export ENROOT_DATA_PATH="$LR_NODE_ROOT/enroot-data" ENROOT_RUNTIME_PATH="$LR_NODE_ROOT/enroot-runtime"
export ENROOT_CACHE_PATH="$LR_NODE_ROOT/enroot-cache" NVIDIA_DISABLE_REQUIRE=1
if [[ $nnodes -gt 1 ]]; then
  export LR_DISCOVERY=etcd
  export ETCD_ENDPOINTS="http://$(lr_node_ip "$head"):2379"
fi

# Every step overlaps the batch step, spans all CPUs and memory of its node (taskset in the
# launchers narrows them), and never reads the batch script's stdin.
step() {
  local nodelist=$1 count=$2
  shift 2
  srun --overlap --nodes="$count" --ntasks="$count" --ntasks-per-node=1 --nodelist="$nodelist" \
    --cpus-per-task="$cpn" --mem=0 --cpu-bind=none "$@" < /dev/null
}
cstep() {
  local nodelist=$1 count=$2
  shift 2
  step "$nodelist" "$count" --container-name="$LR_CONTAINER" --container-mounts="$LR_MOUNTS" \
    --no-container-entrypoint "$@"
}
# Long-running background steps: srun is started as a simple background command, so bg_pid is
# the srun process itself. A SIGTERM to it reaches srun, which forwards it to the step's tasks;
# backgrounding the step() function instead gives the PID of a wrapper subshell, and killing that
# leaves the step (and its port) running.
bg_step() {
  local log=$1 nodelist=$2 count=$3
  shift 3
  srun --overlap --nodes="$count" --ntasks="$count" --ntasks-per-node=1 --nodelist="$nodelist" \
    --cpus-per-task="$cpn" --mem=0 --cpu-bind=none "$@" < /dev/null > "$log" 2>&1 &
  bg_pid=$!
}
bg_cstep() {
  local log=$1 nodelist=$2 count=$3
  shift 3
  bg_step "$log" "$nodelist" "$count" --container-name="$LR_CONTAINER" \
    --container-mounts="$LR_MOUNTS" --no-container-entrypoint "$@"
}
port_open() {
  (exec 3<> "/dev/tcp/$1/$2") 2> /dev/null
}

failures=()
fe_pid=""
worker_pids=()
etcd_pid=""
teardown() {
  trap - EXIT TERM INT
  log "teardown"
  local pid i
  for pid in $fe_pid "${worker_pids[@]}" $etcd_pid; do
    [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null || true
  done
  # Bounded: a step that ignores SIGTERM must not hold the allocation; job exit ends it anyway.
  for i in $(seq 120); do
    local alive=0
    for pid in $fe_pid "${worker_pids[@]}" $etcd_pid; do
      [[ -n "$pid" ]] && kill -0 "$pid" 2> /dev/null && alive=1
    done
    [[ $alive -eq 1 ]] || break
    sleep 1
  done
  for pid in $fe_pid "${worker_pids[@]}" $etcd_pid; do
    [[ -n "$pid" ]] && kill -KILL "$pid" 2> /dev/null || true
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
  python3 - "$run_dir" "$LR_SRC" "$LR_WHEEL_DIR" "${failures[*]:-}" <<'PY' || true
import glob, json, os, sys
run_dir, src, wheel_dir, failures = sys.argv[1:5]
def load(path):
    try:
        return json.load(open(path))
    except Exception as exc:  # recorded, never fatal at teardown
        return {"unreadable": path, "error": str(exc)}
manifest = {
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "nodes": os.environ.get("SLURM_JOB_NODELIST"),
    "plan": load(f"{run_dir}/PLAN.json"),
    "source": load(f"{src}.source-manifest.json"),
    "wheel": load(f"{wheel_dir}/build-manifest.json"),
    "env_verify": load(f"{run_dir}/build/env-verify.json"),
    "weights": {os.path.basename(p): load(p) for p in sorted(glob.glob(f"{run_dir}/weights/*.json"))},
    "policies": {os.path.basename(os.path.dirname(p)): load(p)
                 for p in sorted(glob.glob(f"{run_dir}/policies/*/status.json"))},
    "failures": failures.split() if failures else [],
}
json.dump(manifest, open(f"{run_dir}/runtime-manifest.json", "w"), indent=1, sort_keys=True)
PY
  log "done failures=${failures[*]:-none} run_dir=$run_dir"
}
trap teardown EXIT TERM INT

# 1. Node prep and container creation.
step "$all_nodes" "$nnodes" bash "$LR_DEPLOY_DIR/node_prep.sh" "$run_dir"
step "$all_nodes" "$nnodes" --container-image="$LR_NODE_SQSH" --container-name="$LR_CONTAINER" \
  --container-mounts="$LR_MOUNTS" --no-container-entrypoint \
  bash -c 'python3 -c "import platform, torch, vllm; print(platform.node(), vllm.__version__, torch.__version__)"'

# 2. Weights on every node in parallel with the wheel build on the head node host.
step "$all_nodes" "$nnodes" bash "$LR_DEPLOY_DIR/stage_weights.sh" "$LR_PLAN_DIR" "$run_dir/weights" \
  > "$run_dir/logs/stage-weights.log" 2>&1 &
weights_pid=$!
step "$head" 1 bash "$LR_DEPLOY_DIR/build_env.sh" wheel "$run_dir/build" \
  > "$run_dir/logs/build-wheel.log" 2>&1 &
wheel_pid=$!
wait "$weights_pid" || die "weight staging failed; see $run_dir/logs/stage-weights.log"
wait "$wheel_pid" || die "wheel build failed; see $run_dir/logs/build-wheel.log"

# 3. Venv and environment verification inside the image.
cstep "$head" 1 bash "$LR_DEPLOY_DIR/build_env.sh" venv "$run_dir/build" \
  > "$run_dir/logs/build-venv.log" 2>&1 || die "venv/verify failed; see $run_dir/logs/build-venv.log"

# 4. Discovery (two-node) and workers.
if [[ $nnodes -gt 1 ]]; then
  etcd_bin="${LR_ETCD_BIN:-$LR_LIVE_ROOT/tools/etcd-v3.5.21-linux-amd64/etcd}"
  [[ -x "$etcd_bin" ]] || die "missing $etcd_bin; run fetch_etcd.sh on the login node"
  bg_step "$run_dir/logs/etcd.log" "$head" 1 taskset -c "$LR_AUX_CPUS" "$etcd_bin" \
    --data-dir "$LR_NODE_ROOT/etcd" --listen-client-urls http://0.0.0.0:2379 \
    --advertise-client-urls "$ETCD_ENDPOINTS" --listen-peer-urls http://127.0.0.1:2380
  etcd_pid=$bg_pid
  for _ in $(seq 60); do curl -sf "$ETCD_ENDPOINTS/health" > /dev/null && break; sleep 1; done
  curl -sf "$ETCD_ENDPOINTS/health" > /dev/null || die "etcd did not become healthy"
fi
for node in "${nodes[@]}"; do
  bg_cstep "$run_dir/logs/workers-$node.log" "$node" 1 bash "$LR_DEPLOY_DIR/node_workers.sh" \
    "$LR_PLAN_DIR" "$run_dir/logs/$node"
  worker_pids+=("$bg_pid")
done

model="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "$LR_PLAN_DIR/engine_plan.json")"
block="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["block_size"])' "$LR_PLAN_DIR/engine_plan.json")"
worker_args=()
for node in "${nodes[@]}"; do
  for ((i = 0; i < LR_WORKERS_PER_NODE; i++)); do
    worker_args+=(--worker "$node:$(lr_worker_system_port "$i")")
  done
done
hc() {
  local command=$1
  shift
  python3 "$LR_DEPLOY_DIR/health_check.py" "$command" --frontend "http://$head:$LR_HTTP_PORT" \
    "${worker_args[@]}" --namespace "$LR_NAMESPACE" --model "$model" --block-size "$block" "$@"
}
start_frontend() {
  local slug=$1 dir=$2
  mkdir -p "$dir"
  # A frontend still bound to the port would answer the readiness checks in place of this one.
  ! port_open "$head" "$LR_HTTP_PORT" || die "port $LR_HTTP_PORT is busy before starting $slug"
  bg_cstep "$dir/frontend.log" "$head" 1 bash "$LR_DEPLOY_DIR/frontend.sh" "$LR_PLAN_DIR" "$slug" \
    "$dir" "$num_workers"
  fe_pid=$bg_pid
}
# hc wait, but fail fast when a worker step or the frontend step exits during the wait.
wait_ready() {
  local timeout=$1 out=$2 pid hc_pid
  hc wait --expect "$num_workers" --timeout "$timeout" --out "$out" &
  hc_pid=$!
  while kill -0 "$hc_pid" 2> /dev/null; do
    for pid in "${worker_pids[@]}" $fe_pid; do
      if ! kill -0 "$pid" 2> /dev/null; then
        kill -TERM "$hc_pid" 2> /dev/null || true
        wait "$hc_pid" 2> /dev/null || true
        log "step $pid exited while waiting for readiness"
        return 1
      fi
    done
    sleep 10
  done
  wait "$hc_pid" || return 1
  # The readiness answer must come from the frontend this job just started.
  if [[ -n "$fe_pid" ]] && ! kill -0 "$fe_pid" 2> /dev/null; then
    log "frontend step $fe_pid exited; readiness was answered by another process"
    return 1
  fi
}
stop_frontend() {
  [[ -n "$fe_pid" ]] || return 0
  kill -TERM "$fe_pid" 2>/dev/null || true
  local i
  for i in $(seq 90); do
    kill -0 "$fe_pid" 2> /dev/null || break
    sleep 1
  done
  kill -0 "$fe_pid" 2> /dev/null && kill -KILL "$fe_pid" 2> /dev/null
  wait "$fe_pid" 2>/dev/null || true
  fe_pid=""
  for i in $(seq 60); do
    port_open "$head" "$LR_HTTP_PORT" || return 0
    sleep 1
  done
  die "port $LR_HTTP_PORT is still open after stopping the frontend"
}

# 5. Policies. The first wait also covers model load and CUDA graph capture.
if [[ -n "${LR_POLICIES:-}" ]]; then
  read -r -a policies <<< "$LR_POLICIES"
else
  mapfile -t policies < <(ls "$LR_PLAN_DIR/policies")
fi
wait_timeout="${LR_READY_TIMEOUT_S:-2400}"
for slug in "${policies[@]}"; do
  pdir="$run_dir/policies/$slug"
  status="ok"
  start_frontend "$slug" "$pdir/check"
  if wait_ready "$wait_timeout" "$pdir/check/wait.json" \
    && hc smoke --policy-plan "$LR_PLAN_DIR/policies/$slug/policy_plan.json" \
      --config-dump "$pdir/check/frontend-config.json" --out "$pdir/check/smoke.json" \
    && hc reset --out "$pdir/check/reset.json"; then
    wait_timeout=300
  else
    status="check_failed"
  fi
  stop_frontend
  if [[ "$status" == ok && -n "${LR_PAYLOAD:-}" ]]; then
    start_frontend "$slug" "$pdir/serve"
    if wait_ready 300 "$pdir/serve/wait.json"; then
      hc snapshot --snapshot-dir "$pdir/metrics" --tag before --out "$pdir/metrics/before.json" || true
      started=$(date -u +%FT%TZ)
      set +e
      cstep "$head" 1 env LR_ENDPOINT="http://$head:$LR_HTTP_PORT" LR_MODEL="$model" \
        LR_POLICY_SLUG="$slug" LR_POLICY_DIR="$LR_PLAN_DIR/policies/$slug" \
        LR_PAYLOAD_DIR="$pdir/payload" LR_NUM_WORKERS="$num_workers" \
        timeout "${LR_PAYLOAD_TIMEOUT_S:-5400}" taskset -c "$LR_CLIENT_CPUS" bash "$LR_PAYLOAD" \
        > "$pdir/payload.log" 2>&1
      payload_status=$?
      set -e
      hc snapshot --snapshot-dir "$pdir/metrics" --tag after --out "$pdir/metrics/after.json" || true
      [[ $payload_status -eq 0 ]] || status="payload_exit_$payload_status"
      printf '{"payload_started_utc": "%s", "payload_ended_utc": "%s", "payload_exit": %d}\n' \
        "$started" "$(date -u +%FT%TZ)" "$payload_status" > "$pdir/payload-status.json"
    else
      status="serve_wait_failed"
    fi
    stop_frontend
  fi
  printf '{"policy": "%s", "status": "%s"}\n' "$slug" "$status" > "$pdir/status.json"
  log "policy $slug: $status"
  [[ "$status" == ok ]] || failures+=("$slug:$status")
  if [[ "$status" == check_failed && "${LR_STOP_ON_FAILURE:-1}" == 1 ]]; then
    log "stopping after a failed check (LR_STOP_ON_FAILURE=1)"
    break
  fi
done
[[ ${#failures[@]} -eq 0 ]]
