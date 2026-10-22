#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Gateway A/B benchmark: arm A (Gateway -> Frontend) against arm B
# (Gateway -> EPP), both on mocker workers. aiperf generates and measures the
# load; this script deploys each arm, runs its aiperf Job and copies the
# results back. README.md describes every setting below.
#
# Usage:
#   run.sh all                  deploy, benchmark and tear down each arm in $ORDER
#   run.sh deploy <a|b>         apply an arm and wait until all its workers are ready
#   run.sh bench <a|b>          run the aiperf Job against a deployed arm, then collect
#   run.sh collect <a|b>        copy the results of a finished aiperf Job again
#   run.sh teardown <a|b>       delete an arm and its aiperf Job
#   run.sh plot <results-dir>   aiperf plot of both arms, with A as the baseline

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$HERE/../../.." && pwd)

# The model is fixed by the manifests: the mocker arguments and the route header.
MODEL=Qwen/Qwen3-0.6B

KUBE_CONTEXT=${KUBE_CONTEXT:-$(kubectl config current-context)}
NAMESPACE=${NAMESPACE:-default}
WORKERS=${WORKERS:-2048}
RAMP_STEP=${RAMP_STEP:-512}
DYNAMO_VERSION=${DYNAMO_VERSION:-1.5.1}
FRONTEND_IMAGE=${FRONTEND_IMAGE:-nvcr.io/nvidia/ai-dynamo/dynamo-frontend:$DYNAMO_VERSION}
WORKER_IMAGE=${WORKER_IMAGE:-nvcr.io/nvidia/ai-dynamo/dynamo-planner:$DYNAMO_VERSION}
# The frontend image ships aiperf. Pin this to keep one aiperf across the builds you compare.
AIPERF_IMAGE=${AIPERF_IMAGE:-$FRONTEND_IMAGE}
MODEL_CACHE_PVC=${MODEL_CACHE_PVC:-model-cache}
IMAGE_PULL_SECRET=${IMAGE_PULL_SECRET:-}
GATEWAY_NAME=${GATEWAY_NAME:-inference-gateway}
GATEWAY_NAMESPACE=${GATEWAY_NAMESPACE:-agentgateway-system}
READY_TIMEOUT=${READY_TIMEOUT:-3600}
SETTLE_SECONDS=${SETTLE_SECONDS:-60}
BENCH_TIMEOUT=${BENCH_TIMEOUT:-21600}
ORDER=${ORDER:-a,b}
KEEP=${KEEP:-0}
RESULTS=${RESULTS:-$REPO_ROOT/benchmarks/results/gateway_epp_ab/$(date -u +%Y%m%dT%H%M%SZ)}

# The workload, identical for both arms.
REQUEST_RATES=${REQUEST_RATES:-100,200,400}
CONCURRENCIES=${CONCURRENCIES:-}
ROUNDS=${ROUNDS:-3}
ISL=${ISL:-1024}
OSL=${OSL:-128}
PREFIX_LENGTH=${PREFIX_LENGTH:-512}
NUM_PREFIXES=${NUM_PREFIXES:-256}
DURATION_S=${DURATION_S:-60}
WARMUP_S=${WARMUP_S:-10}
COOLDOWN_S=${COOLDOWN_S:-10}
SEED=${SEED:-42}
SMOKE_TIMEOUT_S=${SMOKE_TIMEOUT_S:-600}
SAMPLE_WORKERS=${SAMPLE_WORKERS:-4}
SERVER_METRICS_FORMATS=${SERVER_METRICS_FORMATS:-json csv}

FAILED_RUNS=0
BACKGROUND=()
WORK=$(mktemp -d)
cleanup() {
  if ((${#BACKGROUND[@]} > 0)); then kill "${BACKGROUND[@]}" 2>/dev/null || true; fi
  rm -rf "$WORK"
}
trap cleanup EXIT

log() { echo "[$(date -u +%H:%M:%S)] $*" >&2; }
die() {
  log "error: $*"
  exit 1
}
k() { kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE" "$@"; }
kg() { kubectl --context "$KUBE_CONTEXT" -n "$GATEWAY_NAMESPACE" "$@"; }

# Arm a is the Frontend, arm b the EPP.
check_arm() { [[ ${1:-} == a || ${1:-} == b ]] || die "the arm is 'a' or 'b', not '${1:-}'"; }
dgd_of() { echo "qwen-$1"; }
label_of() { if [[ $1 == a ]]; then echo a-frontend; else echo b-epp; fi; }
router_of() { if [[ $1 == a ]]; then echo frontend; else echo epp; fi; }
arm_header() { echo "$NAMESPACE/$(dgd_of "$1")"; }
dgd_pods() { echo "nvidia.com/dynamo-graph-deployment-name=$(dgd_of "$1")"; }
job_pod() { k get pods -l "job-name=$(dgd_of "$1")-aiperf" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true; }

preflight() {
  k get dynamographdeployments,httproutes,inferencepools.inference.networking.k8s.io >/dev/null 2>"$WORK/err" ||
    die "cannot list DGDs, HTTPRoutes and InferencePools in $NAMESPACE ($(cat "$WORK/err")); see Prerequisites in README.md"
  [[ $(kg get gateway "$GATEWAY_NAME" -o jsonpath='{.status.conditions[?(@.type=="Programmed")].status}') == True ]] ||
    die "Gateway $GATEWAY_NAMESPACE/$GATEWAY_NAME is missing or not Programmed"
  k get pvc "$MODEL_CACHE_PVC" >/dev/null ||
    die "PVC $MODEL_CACHE_PVC is missing in $NAMESPACE; see Prerequisites in README.md"
}

record_settings() {
  mkdir -p "$RESULTS"
  local name
  for name in KUBE_CONTEXT NAMESPACE WORKERS RAMP_STEP FRONTEND_IMAGE WORKER_IMAGE AIPERF_IMAGE MODEL_CACHE_PVC \
    GATEWAY_NAME GATEWAY_NAMESPACE ORDER REQUEST_RATES CONCURRENCIES ROUNDS ISL OSL PREFIX_LENGTH \
    NUM_PREFIXES DURATION_S WARMUP_S COOLDOWN_S SEED SAMPLE_WORKERS SERVER_METRICS_FORMATS; do
    printf '%s=%q\n' "$name" "${!name}"
  done > "$RESULTS/settings.env"
  git -C "$REPO_ROOT" rev-parse HEAD > "$RESULTS/commit.txt" 2>/dev/null || true
}

# The arm's DGD and HTTPRoute with this run's settings, as JSON.
render_arm() {
  k create --dry-run=client -o json -f "$HERE/$(label_of "$1").yaml" |
    python3 "$HERE/render.py" arm --replicas "$2" \
      --frontend-image "$FRONTEND_IMAGE" --worker-image "$WORKER_IMAGE" \
      --model-cache-pvc "$MODEL_CACHE_PVC" --image-pull-secret "$IMAGE_PULL_SECRET" \
      --gateway-name "$GATEWAY_NAME" --gateway-namespace "$GATEWAY_NAMESPACE" \
      --arm-header "$(arm_header "$1")"
}

worker_readiness() {
  k get pods -l "$(dgd_pods "$1"),nvidia.com/dynamo-component-type=decode" \
    -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}'
}
ready_workers() { worker_readiness "$1" | awk '$2 == "True"' | wc -l; }
unready_workers() { worker_readiness "$1" | awk '$2 != "True" {print $1}' | head -20 | xargs; }

deploy() {
  local arm=$1 label replicas ready deadline accepted last since
  label=$(label_of "$arm")
  mkdir -p "$RESULTS/$label"
  render_arm "$arm" "$WORKERS" > "$RESULTS/$label/manifest.json"
  # Workers come up RAMP_STEP at a time: every Dynamo process watches every
  # worker's discovery record, so a smaller burst spares the API server.
  replicas=$WORKERS
  if ((RAMP_STEP > 0 && RAMP_STEP < WORKERS)); then replicas=$RAMP_STEP; fi
  while :; do
    log "$label: applying with $replicas workers"
    render_arm "$arm" "$replicas" | k apply -f - >&2
    deadline=$((SECONDS + READY_TIMEOUT))
    last=-1
    since=$SECONDS
    until ready=$(ready_workers "$arm") && ((ready >= replicas)); do
      ((SECONDS < deadline)) ||
        die "$label: $ready of $replicas workers ready after ${READY_TIMEOUT}s; not ready: $(unready_workers "$arm")"
      if ((ready != last)); then
        last=$ready
        since=$SECONDS
      elif ((SECONDS - since >= 300)); then
        log "$label: no worker became ready in 5 minutes; not ready: $(unready_workers "$arm")"
        since=$SECONDS
      fi
      log "$label: $ready/$replicas workers ready"
      sleep 15
    done
    if ((replicas == WORKERS)); then break; fi
    replicas=$((replicas + RAMP_STEP > WORKERS ? WORKERS : replicas + RAMP_STEP))
  done
  k wait "dynamographdeployment/$(dgd_of "$arm")" --for=condition=Ready --timeout="${READY_TIMEOUT}s" >&2
  deadline=$((SECONDS + 300))
  until accepted=$(k get httproute "$(dgd_of "$arm")" \
    -o jsonpath='{.status.parents[*].conditions[?(@.type=="Accepted")].status}') && [[ $accepted == *True* ]]; do
    ((SECONDS < deadline)) || die "Gateway $GATEWAY_NAMESPACE/$GATEWAY_NAME did not accept HTTPRoute $(dgd_of "$arm")"
    sleep 5
  done
  log "$label: $WORKERS workers ready; giving the router ${SETTLE_SECONDS}s to discover them"
  sleep "$SETTLE_SECONDS"
}

gateway_url() {
  local service port
  service=$(kg get service -l "gateway.networking.k8s.io/gateway-name=$GATEWAY_NAME" -o jsonpath='{.items[0].metadata.name}')
  port=$(kg get gateway "$GATEWAY_NAME" -o jsonpath='{.spec.listeners[?(@.name=="http")].port}')
  [[ -n $service && -n $port ]] || die "cannot find the Service or the http listener of Gateway $GATEWAY_NAMESPACE/$GATEWAY_NAME"
  echo "http://$service.$GATEWAY_NAMESPACE.svc.cluster.local:$port"
}

write_params() {
  local url
  url=$(gateway_url)
  printf '%s=%q\n' \
    ARM "$(label_of "$1")" \
    ARM_HEADER "$(arm_header "$1")" \
    MODEL "$MODEL" \
    GATEWAY_URL "$url" \
    REQUEST_RATES "$REQUEST_RATES" \
    CONCURRENCIES "$CONCURRENCIES" \
    ROUNDS "$ROUNDS" \
    ISL "$ISL" \
    OSL "$OSL" \
    PREFIX_LENGTH "$PREFIX_LENGTH" \
    NUM_PREFIXES "$NUM_PREFIXES" \
    DURATION_S "$DURATION_S" \
    WARMUP_S "$WARMUP_S" \
    COOLDOWN_S "$COOLDOWN_S" \
    SEED "$SEED" \
    SMOKE_TIMEOUT_S "$SMOKE_TIMEOUT_S" \
    SERVER_METRICS_FORMATS "$SERVER_METRICS_FORMATS"
}

bench() {
  local arm=$1 dgd label inputs
  dgd=$(dgd_of "$arm")
  label=$(label_of "$arm")
  inputs=$RESULTS/$label/inputs
  mkdir -p "$inputs"
  k get dynamographdeployment "$dgd" >/dev/null || die "$dgd is not deployed; run: $0 deploy $arm"
  k get pods -l "$(dgd_pods "$arm")" -o json > "$WORK/pods.json"
  kg get pods -l "gateway.networking.k8s.io/gateway-name=$GATEWAY_NAME" -o json > "$WORK/gateway-pods.json"
  python3 "$HERE/render.py" targets --pods "$WORK/pods.json" --gateway-pods "$WORK/gateway-pods.json" \
    --router "$(router_of "$arm")" --sample-workers "$SAMPLE_WORKERS" \
    --targets-out "$inputs/targets.tsv" --server-metrics-out "$inputs/server-metrics.txt"
  write_params "$arm" > "$inputs/params.env"

  # Foreground, so the pod of an earlier Job is gone before the new one starts.
  k delete job "$dgd-aiperf" --ignore-not-found --cascade=foreground --wait >&2
  k delete configmap "$dgd-bench" --ignore-not-found >&2
  # create, not apply: the targets of 2048 workers overflow apply's annotation.
  k create configmap "$dgd-bench" --from-file="$HERE/bench.sh" --from-file="$inputs/params.env" \
    --from-file="$inputs/targets.tsv" --from-file="$inputs/server-metrics.txt" >&2
  k create --dry-run=client -o json -f "$HERE/aiperf-job.yaml" |
    python3 "$HERE/render.py" job --name "$dgd-aiperf" --configmap "$dgd-bench" \
      --image "$AIPERF_IMAGE" --model-cache-pvc "$MODEL_CACHE_PVC" \
      --image-pull-secret "$IMAGE_PULL_SECRET" |
    tee "$inputs/aiperf-job.json" | k create -f - >&2
  wait_done "$arm"
  collect "$arm"
}

# Wait until bench.sh has written /results/DONE in the aiperf pod.
wait_done() {
  local arm=$1 label pod phase deadline
  label=$(label_of "$arm")
  deadline=$((SECONDS + READY_TIMEOUT))
  until pod=$(job_pod "$arm") && [[ -n $pod && $(k get pod "$pod" -o jsonpath='{.status.phase}') == Running ]]; do
    ((SECONDS < deadline)) || die "the aiperf pod of $label did not start; see: kubectl describe job $(dgd_of "$arm")-aiperf"
    sleep 5
  done
  log "$label: aiperf running in pod $pod; its log goes to $RESULTS/$label/aiperf-job.log"
  k logs -f "$pod" > "$RESULTS/$label/aiperf-job.log" 2>&1 &
  BACKGROUND+=("$!")
  deadline=$((SECONDS + BENCH_TIMEOUT))
  until k exec "$pod" -- test -e /results/DONE 2>/dev/null; do
    phase=$(k get pod "$pod" -o jsonpath='{.status.phase}')
    [[ $phase == Running ]] || die "aiperf pod $pod is $phase before it finished; see $RESULTS/$label/aiperf-job.log"
    ((SECONDS < deadline)) || die "$label: aiperf did not finish within ${BENCH_TIMEOUT}s"
    sleep 30
  done
}

# Copy an arm's results out of its aiperf pod, record what was deployed, and
# let the pod exit.
collect() {
  local arm=$1 label pod status run
  label=$(label_of "$arm")
  pod=$(job_pod "$arm")
  [[ -n $pod ]] || die "no aiperf pod for $label"
  k exec "$pod" -- test -e /results/DONE || die "aiperf in pod $pod has not finished"
  mkdir -p "$RESULTS"
  k exec "$pod" -- tar cf - -C /results "$label" | tar xf - -C "$RESULTS"
  record_deployment "$arm"
  k exec "$pod" -- touch /results/COLLECTED
  log "$label: results in $RESULTS/$label"
  while IFS=$'\t' read -r run status; do
    log "  $run: aiperf exit status $status"
    if [[ $status != 0 ]]; then FAILED_RUNS=$((FAILED_RUNS + 1)); fi
  done < <(cat "$RESULTS/$label/runs.tsv" 2>/dev/null || true)
  status=$(cat "$RESULTS/$label/exit-code" 2>/dev/null || echo unknown)
  if [[ $status != 0 ]]; then
    log "$label: bench.sh exited with status $status; see $RESULTS/$label/aiperf-job.log"
    FAILED_RUNS=$((FAILED_RUNS + 1))
  fi
}

# What actually ran: the DGD with its status, the pods and their image digests.
record_deployment() {
  local arm=$1 dgd dir
  dgd=$(dgd_of "$arm")
  dir=$RESULTS/$(label_of "$arm")/deployed
  mkdir -p "$dir"
  k get dynamographdeployment "$dgd" -o yaml > "$dir/dgd.yaml"
  k get httproute "$dgd" -o yaml > "$dir/httproute.yaml"
  if [[ $arm == b ]]; then
    k get inferencepools.inference.networking.k8s.io "$dgd-pool" -o yaml > "$dir/inferencepool.yaml"
  fi
  k get pods -l "$(dgd_pods "$arm")" -o wide > "$dir/pods.txt"
  k get pods -l "$(dgd_pods "$arm")" \
    -o jsonpath='{range .items[*]}{range .status.containerStatuses[*]}{.name}{"\t"}{.imageID}{"\n"}{end}{end}' |
    sort | uniq -c > "$dir/images.txt"
  kg get pods -l "gateway.networking.k8s.io/gateway-name=$GATEWAY_NAME" -o wide > "$dir/gateway-pods.txt"
  kubectl --context "$KUBE_CONTEXT" get nodes -o wide > "$dir/nodes.txt" 2>/dev/null || true
  kubectl --context "$KUBE_CONTEXT" get deployments -A -l app.kubernetes.io/name=dynamo-operator \
    -o jsonpath='{range .items[*]}{.spec.template.spec.containers[*].image}{"\n"}{end}' \
    > "$dir/operator-image.txt" 2>/dev/null || true
}

teardown() {
  local arm=$1 dgd deadline
  dgd=$(dgd_of "$arm")
  log "$(label_of "$arm"): tearing down"
  k delete job "$dgd-aiperf" --ignore-not-found --wait=false >&2
  k delete configmap "$dgd-bench" --ignore-not-found >&2
  k delete httproute "$dgd" --ignore-not-found >&2
  k delete dynamographdeployment "$dgd" --ignore-not-found >&2
  deadline=$((SECONDS + READY_TIMEOUT))
  while [[ -n $(k get pods -l "$(dgd_pods "$arm")" -o name) ]]; do
    ((SECONDS < deadline)) || die "pods of $dgd still exist after ${READY_TIMEOUT}s"
    sleep 10
  done
}

all() {
  local arm arms
  IFS=, read -ra arms <<< "$ORDER"
  for arm in "${arms[@]}"; do check_arm "$arm"; done
  preflight
  record_settings
  for arm in "${arms[@]}"; do
    deploy "$arm"
    bench "$arm"
    if [[ $KEEP != 1 ]]; then teardown "$arm"; fi
  done
  log "compare the arms with: $0 plot $RESULTS"
}

# aiperf's own plots, with arm A as the baseline and arm B as the treatment.
plot() {
  local dir=${1:-} aiperf python
  [[ -d $dir ]] || die "usage: $0 plot <results-dir>"
  aiperf=$(command -v aiperf) || die "aiperf is not installed; install the version in <arm>/aiperf-version.txt"
  # The interpreter aiperf runs under, which can import its default plot config.
  read -r -a python <<< "$(sed -n '1s/^#!//p' "$aiperf")"
  "${python[@]}" - "$WORK/plot-config.yaml" <<'EOF'
import pathlib
import sys

import aiperf.plot

config = pathlib.Path(aiperf.plot.__file__).with_name("default_plot_config.yaml").read_text()
if "\nexperiment_classification:" in config:
    sys.exit("this aiperf already sets experiment_classification; edit ~/.aiperf/plot_config.yaml instead")
# aiperf groups runs by the name of their parent directory, the arm, and tells
# baseline from treatment by the run's path; each arm needs a pattern for both.
config += """
experiment_classification:
  baselines: ["a-frontend", "*/a-frontend/*"]
  treatments: ["b-epp", "*/b-epp/*"]
  default: treatment
"""
pathlib.Path(sys.argv[1]).write_text(config)
EOF
  aiperf plot "$dir" --config "$WORK/plot-config.yaml" --output "$dir/plots"
}

main() {
  local command=${1:-}
  case $command in
    all)
      all
      ;;
    deploy | bench)
      check_arm "${2:-}"
      preflight
      record_settings
      "$command" "$2"
      ;;
    collect | teardown)
      check_arm "${2:-}"
      "$command" "$2"
      ;;
    plot)
      plot "${2:-}"
      ;;
    *)
      cat >&2 <<EOF
usage: $0 all
       $0 deploy|bench|collect|teardown a|b
       $0 plot <results-dir>
EOF
      exit 2
      ;;
  esac
  if ((FAILED_RUNS > 0)); then die "$FAILED_RUNS run(s) failed; see the logs under $RESULTS"; fi
}

main "$@"
