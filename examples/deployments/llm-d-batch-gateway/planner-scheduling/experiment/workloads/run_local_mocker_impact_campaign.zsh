#!/bin/zsh
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

fail() {
  print -u2 -- "fatal: $*"
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command is missing: $1"
}

for command_name in comm curl envsubst git grep helm jq kubectl sort; do
  require_command $command_name
done

script_dir=${0:A:h}
experiment=${EXPERIMENT_ROOT:-${script_dir:h}}
repo=${DYNAMO_REPO_ROOT:-$(git -C $script_dir rev-parse --show-toplevel)}
planner_root=${experiment:h}
example_root=${planner_root:h}
raw=$experiment/results/raw

context=${KUBE_CONTEXT:-$(kubectl config current-context)}
namespace=${NAMESPACE:?set NAMESPACE to the campaign namespace}
dynamo_image=${DYNAMO_IMAGE:?set DYNAMO_IMAGE to an immutable Dynamo image}
planner_image=${PLANNER_IMAGE:-$dynamo_image}
async_image=${ASYNC_IMAGE:-ghcr.io/llm-d/llm-d-async:v0.10.0}
async_chart_version=${ASYNC_CHART_VERSION:-v0.10.0}
dynamo_runtime_version=${DYNAMO_RUNTIME_VERSION:-1.5.0}
tenant=${BATCH_TENANT:-planner-poc-batch-impact}
model=${MODEL:-Qwen/Qwen3-0.6B}
# The paired online/Batch campaign intentionally keeps one warm worker in all
# arms so cold-start and topology do not confound drain-control measurements.
batch_warm_replicas=1

trace=${TRACE:?set TRACE to the generated Mooncake JSONL trace}
dataset=${DATASET:?set DATASET to the converted OpenAI Batch JSONL dataset}
trace_sha256=${TRACE_SHA256:-b8dcdd7c630831dbd37f4d49a4d7f347c256941626f347d737ee50162d9ad314}
dataset_sha256=${DATASET_SHA256:-488c0fb6b64ce349e8d8da1ea001ccec4f8339194949aec7a03b6fcddaaeab8f}

planner_gym_root=${PLANNER_GYM_ROOT:?set PLANNER_GYM_ROOT to the Planner Gym gym directory}
planner_gym_revision=${PLANNER_GYM_REVISION:-cd8e42b04aa8389d5f66d8d765513bd5308f3fdc}
dynamo_python=${DYNAMO_PYTHON:-$repo/.venv/bin/python}
planner_gym_python=${PLANNER_GYM_PYTHON:-$planner_gym_root/.venv/bin/python}
aiperf_executable=${AIPERF_EXECUTABLE:-$planner_gym_root/.venv/bin/aiperf}

frontend_port=${FRONTEND_PORT:-18000}
batch_port=${BATCH_PORT:-18001}
async_metrics_port=${ASYNC_METRICS_PORT:-18002}
prometheus_port=${PROMETHEUS_PORT:-19091}
cooldown_seconds=${COOLDOWN_SECONDS:-30}

campaign_id=${CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-planner-gym-batch-impact}
if [[ -n ${CAMPAIGN_STATE_DIR:-} ]]; then
  campaign=$CAMPAIGN_STATE_DIR
  [[ ! -e $campaign ]] || fail "campaign state directory already exists: $campaign"
  mkdir -p $campaign
else
  campaign=$(mktemp -d "${TMPDIR:-/tmp}/dynamo-planner-gym.XXXXXX")
fi
index=$campaign/runs.tsv
compiled=${COMPILED_OUTPUT:-$experiment/results/compiled/$campaign_id-r5}
rendered=$campaign/rendered
mkdir -p $rendered
: > $index

runner=$script_dir/run_planner_gym_batch_impact.py
compiler=$script_dir/compile_planner_gym_batch_impact.py
mocker_renderer=$script_dir/render_mocker_campaign.py
attestation_resolver=$script_dir/resolve_campaign_image_attestation.py
base_values=$example_root/llm-d-async-values.yaml
stock_values_template=$planner_root/llm-d-async-stock-campaign-values.yaml
native_values_template=$example_root/llm-d-async-planner-values.yaml
campaign_metrics_values=$planner_root/llm-d-async-mocker-campaign-values.yaml
planner_manifest_template=$planner_root/planner-poc.yaml
mocker_manifest_template=$planner_root/mocker-campaign.yaml
prometheus_manifest_template=$planner_root/experiment-prometheus.yaml

stock_values=$rendered/async-stock.yaml
native_values=$rendered/async-native.yaml
planner_manifest=$rendered/planner.yaml
mocker_manifest=$rendered/mocker.yaml
prometheus_manifest=$rendered/prometheus.yaml

typeset -a forward_pids
forward_pids=()
async_pf_pid=
pf_generation=0

stop_pid() {
  local pid=$1
  if [[ -n $pid ]] && kill -0 $pid 2>/dev/null; then
    kill $pid
    wait $pid 2>/dev/null || true
  fi
}

cleanup() {
  local pid
  for pid in $forward_pids; do
    stop_pid $pid
  done
}
trap cleanup EXIT INT TERM

for path in \
  $dynamo_python \
  $planner_gym_python \
  $aiperf_executable \
  $trace \
  $dataset \
  $runner \
  $compiler \
  $mocker_renderer \
  $attestation_resolver \
  $base_values \
  $stock_values_template \
  $native_values_template \
  $campaign_metrics_values \
  $planner_manifest_template \
  $mocker_manifest_template \
  $prometheus_manifest_template; do
  [[ -e $path ]] || fail "required path does not exist: $path"
done
[[ ! -e $compiled ]] || fail "compiled output already exists: $compiled"

planner_gym_repo=$(git -C $planner_gym_root rev-parse --show-toplevel)
actual_planner_gym_revision=$(git -C $planner_gym_repo rev-parse HEAD)
[[ $actual_planner_gym_revision == $planner_gym_revision ]] || fail \
  "Planner Gym revision is $actual_planner_gym_revision; expected $planner_gym_revision"
[[ -z $(git -C $planner_gym_repo status --short) ]] || fail \
  "Planner Gym checkout must be clean: $planner_gym_repo"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    fail 'sha256sum or shasum is required'
  fi
}

[[ $(sha256_file $trace) == $trace_sha256 ]] || fail \
  "trace SHA-256 does not match TRACE_SHA256: $trace"
[[ $(sha256_file $dataset) == $dataset_sha256 ]] || fail \
  "dataset SHA-256 does not match DATASET_SHA256: $dataset"

case $async_image in
  *@*) fail 'ASYNC_IMAGE must use a tag because this chart cannot render digests' ;;
esac
async_repository=${async_image%:*}
async_tag=${async_image##*:}
[[ $async_repository != $async_image && -n $async_tag && $async_tag != */* ]] || \
  fail 'ASYNC_IMAGE must be a complete repository:tag reference'

export NAMESPACE=$namespace
export PLANNER_IMAGE=$planner_image
export BATCH_TENANT=$tenant
export BATCH_WARM_REPLICAS=$batch_warm_replicas

envsubst '${NAMESPACE}' < $stock_values_template > $stock_values
envsubst '${NAMESPACE}' < $native_values_template > $native_values
envsubst '${NAMESPACE} ${PLANNER_IMAGE} ${BATCH_TENANT} ${BATCH_WARM_REPLICAS}' \
  < $planner_manifest_template > $planner_manifest
envsubst '${NAMESPACE}' < $prometheus_manifest_template > $prometheus_manifest

typeset -a mocker_render_args
mocker_render_args=(
  --template $mocker_manifest_template
  --output $mocker_manifest
  --namespace $namespace
  --dynamo-image $dynamo_image
  --dynamo-runtime-version $dynamo_runtime_version
  --mocker-max-num-seqs ${MOCKER_MAX_NUM_SEQS:-16}
  --mocker-timing-flag ${MOCKER_TIMING_FLAG:---extra-engine-args}
  --mocker-timing-value ${MOCKER_TIMING_VALUE:-/etc/dynamo/mocker/engine-args.json}
)
if [[ -n ${MODEL_CACHE_CLAIM:-} ]]; then
  mocker_render_args+=(--model-cache-claim $MODEL_CACHE_CLAIM)
fi
if [[ -n ${IMAGE_PULL_SECRET:-} ]]; then
  mocker_render_args+=(--image-pull-secret $IMAGE_PULL_SECRET)
fi
$dynamo_python $mocker_renderer $mocker_render_args

for rendered_file in \
  $stock_values \
  $native_values \
  $planner_manifest \
  $mocker_manifest \
  $prometheus_manifest; do
  grep -Fq '${' $rendered_file && fail "unresolved template variable: $rendered_file"
done
kubectl --context $context --namespace $namespace apply --dry-run=client \
  --filename $mocker_manifest >/dev/null
kubectl --context $context --namespace $namespace apply --dry-run=client \
  --filename $planner_manifest >/dev/null
kubectl --context $context --namespace $namespace apply --dry-run=client \
  --filename $prometheus_manifest >/dev/null
kubectl --context $context --namespace $namespace get secret \
  hf-token-secret >/dev/null || fail \
  "hf-token-secret is required by the parent Batch backend recipe in $namespace"

kubectl --context $context --namespace $namespace apply \
  --filename $mocker_manifest
kubectl --context $context --namespace $namespace apply \
  --filename $prometheus_manifest
kubectl --context $context --namespace $namespace wait --for=condition=Ready \
  dynamographdeployment/qwen3-0-6b-batch --timeout=10m
kubectl --context $context --namespace $namespace rollout status \
  deployment/planner-impact-prometheus --timeout=5m

start_forward() {
  local name=$1
  local resource=$2
  local mapping=$3
  local log=$campaign/$name-port-forward.log
  kubectl --context $context --namespace $namespace port-forward \
    $resource $mapping >$log 2>&1 &
  REPLY=$!
  forward_pids+=($REPLY)
}

wait_for_url() {
  local name=$1
  local pid=$2
  local url=$3
  shift 3
  for _ in {1..60}; do
    if curl -fsS "$@" $url >/dev/null; then
      return
    fi
    kill -0 $pid 2>/dev/null || fail "$name port-forward exited"
    sleep 1
  done
  fail "$name port-forward did not become healthy: $url"
}

start_forward frontend service/qwen3-0-6b-batch-frontend \
  $frontend_port:8000
frontend_pf_pid=$REPLY
start_forward batch service/batch-gateway-apiserver $batch_port:8000
batch_pf_pid=$REPLY
start_forward prometheus service/planner-impact-prometheus $prometheus_port:9090
prometheus_pf_pid=$REPLY

wait_for_url frontend $frontend_pf_pid \
  http://127.0.0.1:$frontend_port/v1/models
wait_for_url batch $batch_pf_pid \
  "http://127.0.0.1:$batch_port/v1/batches?limit=1" \
  -H "X-MaaS-Username: $tenant"
wait_for_url prometheus $prometheus_pf_pid \
  http://127.0.0.1:$prometheus_port/-/ready

clear_planner() {
  kubectl --context $context --namespace $namespace delete \
    dynamographdeployment/qwen3-0-6b-batch-planner \
    --ignore-not-found --wait=true --timeout=5m
  for _ in {1..120}; do
    if ! kubectl --context $context --namespace $namespace get pods -o name | \
      grep -Eq '^pod/qwen3-0-6b-batch-planner-'; then
      return
    fi
    sleep 2
  done
  fail 'Planner pod did not disappear'
}

helm_async() {
  local overlay=$1
  helm upgrade --install async-dispatch \
    oci://ghcr.io/llm-d/charts/llm-d-async \
    --kube-context $context \
    --namespace $namespace \
    --version $async_chart_version \
    --reset-values \
    --values $base_values \
    --values $overlay \
    --values $campaign_metrics_values \
    --set-string ap.image.repository=$async_repository \
    --set-string ap.image.tag=$async_tag \
    --atomic \
    --wait \
    --timeout 5m
  kubectl --context $context --namespace $namespace rollout status \
    deployment/async-dispatch-llm-d-async --timeout=5m
}

stop_async_pf() {
  stop_pid $async_pf_pid
  async_pf_pid=
}

start_async_pf() {
  stop_async_pf
  (( pf_generation += 1 ))
  local log=$campaign/async-port-forward-$pf_generation.log
  kubectl --context $context --namespace $namespace port-forward \
    service/async-dispatch-llm-d-async-metrics \
    $async_metrics_port:9090 >$log 2>&1 &
  async_pf_pid=$!
  forward_pids+=($async_pf_pid)
  wait_for_url async $async_pf_pid \
    http://127.0.0.1:$async_metrics_port/metrics
}

assert_base_ready() {
  kubectl --context $context --namespace $namespace wait --for=condition=Ready \
    dynamographdeployment/qwen3-0-6b-batch --timeout=10m
  kubectl --context $context --namespace $namespace rollout status \
    deployment/async-dispatch-llm-d-async --timeout=5m
  kubectl --context $context --namespace $namespace get pods -o json | jq -e '
    [.items[]
     | select(.metadata.name | test("^(qwen3-0-6b-batch-(frontend|worker)|batch-gateway-|planner-impact-prometheus|async-dispatch-)"))
     | .status.containerStatuses[]?
     | select(.ready != true or .restartCount != 0)]
    | length == 0
  ' >/dev/null
  curl -fsS http://127.0.0.1:$frontend_port/v1/models >/dev/null
  curl -fsS -H "X-MaaS-Username: $tenant" \
    "http://127.0.0.1:$batch_port/v1/batches?limit=1" >/dev/null
  curl -fsS http://127.0.0.1:$async_metrics_port/metrics >/dev/null
  curl -fsS http://127.0.0.1:$prometheus_port/-/ready >/dev/null
}

assert_no_active_batches() {
  curl -fsS -H "X-MaaS-Username: $tenant" \
    "http://127.0.0.1:$batch_port/v1/batches?limit=100" | jq -e '
      [.data[]?
       | select(.status as $status
         | ["validating", "queued", "in_progress", "finalizing", "cancelling"]
         | index($status))]
      | length == 0
    ' >/dev/null
}

switch_stock() {
  print -- 'switching to stock Async'
  clear_planner
  stop_async_pf
  helm_async $stock_values
  start_async_pf
  if kubectl --context $context --namespace $namespace get deployment \
    async-dispatch-llm-d-async -o json | \
    jq -e '[.. | strings | select(contains("redis-leased-rate"))] | length > 0' \
      >/dev/null; then
    fail 'stock Async unexpectedly contains redis-leased-rate'
  fi
  assert_base_ready
}

switch_native() {
  print -- 'switching to native Planner control'
  clear_planner
  stop_async_pf
  helm_async $native_values
  start_async_pf
  kubectl --context $context --namespace $namespace get configmap \
    async-dispatch-llm-d-async-config \
    -o jsonpath='{.data.worker-pools\.json}' | \
    grep -Fq 'redis-leased-rate' || fail 'native Async is missing redis-leased-rate'
  kubectl --context $context --namespace $namespace apply \
    --filename $planner_manifest
  kubectl --context $context --namespace $namespace wait --for=condition=Ready \
    dynamographdeployment/qwen3-0-6b-batch-planner --timeout=10m
  assert_base_ready
}

run_campaign_cell() {
  local arm=$1
  local -a args
  args=(
    --arm $arm
    --experiment-root $experiment
    --dynamo-repo-root $repo
    --planner-gym-root $planner_gym_root
    --planner-gym-script $planner_gym_root/scripts/run_match_config.py
    --planner-gym-python $planner_gym_python
    --aiperf-executable $aiperf_executable
    --batch-harness $script_dir/baseline_harness.py
    --batch-python $dynamo_python
    --endpoint-url http://127.0.0.1:$frontend_port
    --batch-base-url http://127.0.0.1:$batch_port
    --model $model
    --endpoint-type chat
    --tokenizer builtin
    --trace $trace
    --trace-name batch-impact-online-5-8-5
    --trace-block-size 512
    --trace-presorted
    --seed 0
    --arrival-speedup 1
    --slo-name interactive
    --slo-ttft-ms 300
    --slo-itl-ms 50
    --namespace $namespace
    --context $context
    --tenant $tenant
    --metrics-url frontend=http://127.0.0.1:$frontend_port/metrics
    --metrics-url async=http://127.0.0.1:$async_metrics_port/metrics
    --metrics-interval-seconds 5
    --pod-name-regex '^(batch-gateway-|async-dispatch-llm-d-async-|qwen3-0-6b-batch-(frontend|worker|planner)-)'
    --expected-worker-pool-id dynamo-batch
    --validation-timeout-seconds 120
    --campaign-timeout-seconds 2400
  )

  if [[ $arm == online-only ]]; then
    args+=(--expected-gate-type prometheus-query)
  else
    args+=(
      --dataset $dataset
      --batch-size 1500
      --batch-start-index 0
      --batch-max-tokens 128
      --batch-temperature 0
      --completion-window 24h
      --batch-request-timeout-seconds 120
      --batch-poll-interval-seconds 1
      --batch-timeout-seconds 1800
      --batch-abort-cleanup-timeout-seconds 180
      --batch-start-timeout-seconds 300
    )
    if [[ $arm == planner-native ]]; then
      args+=(
        --expected-gate-type redis-leased-rate
        --native-planner-configmap qwen3-0-6b-batch-planner-config
        --native-planner-pod-name-regex '^qwen3-0-6b-batch-planner-'
        --native-planner-decision-log-regex 'Batch scheduling decision:'
        --native-planner-min-decision-logs 2
      )
    else
      args+=(--expected-gate-type prometheus-query)
    fi
  fi

  PYTHONUNBUFFERED=1 $dynamo_python $runner $args
}

run_cell() {
  local label=$1
  local arm=$2
  local before=$campaign/$label.before
  local after=$campaign/$label.after
  local delta=$campaign/$label.delta
  local log=$campaign/$label.log
  local rc count run_dir validation

  assert_base_ready
  assert_no_active_batches
  print -- "$label: $cooldown_seconds-second pre-cell cooldown"
  sleep $cooldown_seconds
  find $raw -mindepth 1 -maxdepth 1 -type d \
    -name '*planner-gym-batch-impact-*' -print | sort > $before

  set +e
  run_campaign_cell $arm 2>&1 | tee $log
  rc=$pipestatus[1]
  set -e

  find $raw -mindepth 1 -maxdepth 1 -type d \
    -name '*planner-gym-batch-impact-*' -print | sort > $after
  comm -13 $before $after > $delta
  count=$(wc -l < $delta | tr -d ' ')
  [[ $count == 1 ]] || \
    fail "$label produced $count new result directories (runner exit $rc)"
  run_dir=$(sed -n '1p' $delta)
  [[ $rc == 0 ]] || fail "$label failed with exit $rc; artifacts: $run_dir"
  jq -e '.status == "completed" and .exit_code == 0' \
    $run_dir/metadata.json >/dev/null

  if [[ $arm != online-only ]]; then
    validation=$(find $run_dir/batch-harness/results/raw -type f \
      -name result-validation.json -print -quit)
    [[ -n $validation ]] || fail "$label is missing Batch result validation"
    jq -e '.valid == true and .failed == 0' $validation >/dev/null
  fi

  printf '%s\t%s\t%s\n' $label $arm $run_dir | tee -a $index
  print -- "$label: $cooldown_seconds-second post-cell cooldown"
  sleep $cooldown_seconds
  assert_base_ready
  assert_no_active_batches
}

# Repeat the three-order Latin square, then its first two rows. This is the
# frozen chronological order used by the reportable five-repetition campaign.
switch_stock
run_cell r1-online online-only
run_cell r1-stock stock
switch_native
run_cell r1-native planner-native

switch_stock
run_cell r2-stock stock
switch_native
run_cell r2-native planner-native
switch_stock
run_cell r2-online online-only

switch_native
run_cell r3-native planner-native
switch_stock
run_cell r3-online online-only
run_cell r3-stock stock

run_cell r4-online online-only
run_cell r4-stock stock
switch_native
run_cell r4-native planner-native

switch_stock
run_cell r5-stock stock
switch_native
run_cell r5-native planner-native
switch_stock
run_cell r5-online online-only

typeset -a compile_args
compile_args=()
while IFS=$'\t' read -r label arm run_dir; do
  compile_args+=(--run-directory $run_dir)
done < $index

typeset -a attestation_args
attestation_args=(
  --runs-index $index
  --dynamo-image $dynamo_image
)
if [[ -n ${DYNAMO_IMAGE_ID_ATTESTATION:-} ]]; then
  attestation_args+=(--resolved-image-id $DYNAMO_IMAGE_ID_ATTESTATION)
fi
image_id_attestation=$(
  $dynamo_python $attestation_resolver $attestation_args
)
if [[ -n $image_id_attestation ]]; then
  compile_args+=(--image-id-attestation $image_id_attestation)
fi

$dynamo_python $compiler $compile_args \
  --output-directory $compiled \
  --minimum-repetitions 5 \
  --cv-threshold-percent 5 \
  --minimum-batch-overlap-fraction 0.95

print -- "campaign state: $campaign"
print -- "compiled evidence: $compiled"
