# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# shellcheck shell=bash
# shellcheck disable=SC2034  # variables are shared with the other sourced files
#
# Shared settings and helpers of the end-to-end test: output, polling and
# assertion bookkeeping, and kubectl, helm and registry REST wrappers. Sourced
# by run.sh; compatible with bash 3.2 (the macOS default).

E2E_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${E2E_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${MODULE_DIR}/../.." && pwd)"
CHART_DIR="${REPO_ROOT}/deploy/helm/charts/global-discovery-mcs"
MANIFESTS="${E2E_DIR}/manifests"

# Phases to run, separated by spaces or commas.
E2E_PHASES="${E2E_PHASES:-1 2 3}"
# 1 keeps the clusters when the run fails; always keeps them after a pass too.
E2E_KEEP="${E2E_KEEP:-0}"
# 1 reuses the images that an earlier run built and loaded into Docker.
E2E_SKIP_BUILD="${E2E_SKIP_BUILD:-0}"
# Treat a documented behaviour that the Global Discovery Service does not
# have yet (a known gap) as a failure.
E2E_STRICT="${E2E_STRICT:-0}"
# Seconds an assertion polls before it fails.
E2E_TIMEOUT="${E2E_TIMEOUT:-90}"
E2E_HELM_TIMEOUT="${E2E_HELM_TIMEOUT:-5m}"

# The kind clusters are <prefix>-hub, <prefix>-a and <prefix>-b. Nothing else
# is ever touched.
E2E_PREFIX="${E2E_PREFIX:-mcs-e2e}"
E2E_CLUSTERSET="${E2E_CLUSTERSET:-e2e}"
# Phases 1 and 2 use all three clusters; phase 3 needs only the hub and a, so
# E2E_PHASES=3 creates two. E2E_CLUSTER_KEYS overrides this (steps that need a
# missing cluster are skipped).
if [ -z "${E2E_CLUSTER_KEYS:-}" ]; then
  case " ${E2E_PHASES//,/ } " in
    *" 1 "* | *" 2 "*) E2E_CLUSTER_KEYS="hub a b" ;;
    *) E2E_CLUSTER_KEYS="hub a" ;;
  esac
fi
CLUSTER_KEYS="${E2E_CLUSTER_KEYS}"

# KIND is the binary make kind installs into bin/. The node image is pinned
# by digest, as the kind release notes ask.
KIND="${KIND:-${MODULE_DIR}/bin/kind-v0.33.0}"
KIND_NODE_IMAGE="${KIND_NODE_IMAGE:-kindest/node:v1.35.8@sha256:07b2536e30b803ed61d1677a79df6115f798ce64c80f9e22f6ed45afd09323c0}"

# Distinct pod and service subnets per cluster. The pod subnets are routed
# between the kind nodes; the service subnets are not (see README.md).
POD_SUBNET_HUB="${POD_SUBNET_HUB:-10.241.0.0/16}"
POD_SUBNET_A="${POD_SUBNET_A:-10.242.0.0/16}"
POD_SUBNET_B="${POD_SUBNET_B:-10.243.0.0/16}"
SERVICE_SUBNET_HUB="${SERVICE_SUBNET_HUB:-10.251.0.0/16}"
SERVICE_SUBNET_A="${SERVICE_SUBNET_A:-10.252.0.0/16}"
SERVICE_SUBNET_B="${SERVICE_SUBNET_B:-10.253.0.0/16}"
# The Docker network of kind clusters; all nodes get an address on it.
KIND_NETWORK="${KIND_NETWORK:-kind}"

# Images. The registry and controller image is built from the module
# Dockerfile; phase 3 also builds the Global Discovery Service image from
# container/Dockerfile.global-discovery-service.
MCS_IMAGE_REPO="${MCS_IMAGE_REPO:-global-discovery-mcs}"
GDS_IMAGE_REPO="${GDS_IMAGE_REPO:-global-discovery-service}"
BUSYBOX_IMAGE="${BUSYBOX_IMAGE:-busybox:1.37}"
case "$(uname -m)" in
  arm64 | aarch64) DEFAULT_PLATFORM="linux/arm64" ;;
  *) DEFAULT_PLATFORM="linux/amd64" ;;
esac
E2E_PLATFORM="${E2E_PLATFORM:-${DEFAULT_PLATFORM}}"
MCS_IMAGE_TAG=""
GDS_IMAGE_TAG=""

MCS_NAMESPACE="mcs-system"
MCS_RELEASE="mcs"

E2E_WORK_DIR="${E2E_WORK_DIR:-${MODULE_DIR}/bin/e2e}"
KUBECONFIG_DIR="${E2E_WORK_DIR}/kubeconfig"

# Set by setup.sh: the registry address the member clusters dial.
REGISTRY_URL=""

RESULTS=()
PASS_COUNT=0
FAIL_COUNT=0
GAP_COUNT=0
SKIP_COUNT=0
LAST_OBSERVED=""
PHASE_PASS=0
PHASE_FAIL=0
PHASE_GAP=0
PHASE_SKIP=0
PHASE_NAMES=""

# ---------------------------------------------------------------------------
# Output

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

section() {
  printf '\n===== %s =====\n' "$*"
}

die() {
  printf 'e2e: %s\n' "$*" >&2
  exit 2
}

pass() {
  PASS_COUNT=$((PASS_COUNT + 1))
  PHASE_PASS=$((PHASE_PASS + 1))
  RESULTS+=("PASS  $1${2:+ ($2)}")
  log "PASS  $1${2:+ ($2)}"
}

fail() {
  FAIL_COUNT=$((FAIL_COUNT + 1))
  PHASE_FAIL=$((PHASE_FAIL + 1))
  RESULTS+=("FAIL  $1${2:+ ($2)}")
  log "FAIL  $1${2:+ ($2)}"
}

skip() {
  SKIP_COUNT=$((SKIP_COUNT + 1))
  PHASE_SKIP=$((PHASE_SKIP + 1))
  RESULTS+=("SKIP  $1")
  log "SKIP  $1"
}

# gap NAME DETAIL: a documented behaviour the service lacks today. It does
# not fail the run, unless E2E_STRICT=1, and the summary counts it apart.
gap() {
  if [ "${E2E_STRICT}" = "1" ]; then
    fail "$1" "known gap, E2E_STRICT=1: $2"
    return
  fi
  GAP_COUNT=$((GAP_COUNT + 1))
  PHASE_GAP=$((PHASE_GAP + 1))
  RESULTS+=("GAP   $1 ($2)")
  log "GAP   $1 ($2)"
}

# end_phase NAME: records the counters of the phase that just ran.
end_phase() {
  PHASE_NAMES="${PHASE_NAMES}${PHASE_NAMES:+|}phase $1: ${PHASE_PASS} passed, ${PHASE_FAIL} failed, ${PHASE_GAP} known gaps, ${PHASE_SKIP} skipped"
  PHASE_PASS=0
  PHASE_FAIL=0
  PHASE_GAP=0
  PHASE_SKIP=0
}

# print_summary [EXIT_STATUS]: a non-zero status (a setup error) is a failure
# even when no assertion failed.
print_summary() {
  local r status="${1:-0}"
  section "SUMMARY"
  for r in ${RESULTS[@]+"${RESULTS[@]}"}; do
    printf '%s\n' "${r}"
  done
  printf -- '---------------------------------------------\n'
  local IFS='|' line
  for line in ${PHASE_NAMES}; do
    printf '%s\n' "${line}"
  done
  printf -- '---------------------------------------------\n'
  if [ "${FAIL_COUNT}" -eq 0 ] && [ "${status}" -eq 0 ]; then
    printf 'RESULT: PASS (%d passed, %d failed, %d known gaps, %d skipped)\n' \
      "${PASS_COUNT}" "${FAIL_COUNT}" "${GAP_COUNT}" "${SKIP_COUNT}"
  else
    printf 'RESULT: FAIL (%d passed, %d failed, %d known gaps, %d skipped)\n' \
      "${PASS_COUNT}" "${FAIL_COUNT}" "${GAP_COUNT}" "${SKIP_COUNT}"
  fi
}

# ---------------------------------------------------------------------------
# Polling. A check function returns 0 when satisfied and may set
# LAST_OBSERVED to describe what it saw.

poll() {
  local timeout="$1"
  shift
  local deadline=$((SECONDS + timeout))
  while :; do
    if "$@"; then
      return 0
    fi
    if [ "${SECONDS}" -ge "${deadline}" ]; then
      return 1
    fi
    sleep 2
  done
}

# expect NAME CHECK [ARGS...]: poll CHECK for E2E_TIMEOUT seconds and record
# the outcome.
expect() {
  local name="$1"
  shift
  local start="${SECONDS}"
  LAST_OBSERVED=""
  if poll "${E2E_TIMEOUT}" "$@"; then
    pass "${name}" "$((SECONDS - start))s"
    return 0
  fi
  fail "${name}" "timed out after ${E2E_TIMEOUT}s; last observed: ${LAST_OBSERVED:-nothing}"
  return 1
}

# once NAME CHECK [ARGS...]: run CHECK once, for steps that write.
once() {
  local name="$1"
  shift
  LAST_OBSERVED=""
  if "$@"; then
    pass "${name}"
    return 0
  fi
  fail "${name}" "observed: ${LAST_OBSERVED:-nothing}"
  return 1
}

# ---------------------------------------------------------------------------
# Clusters. Every kubectl and helm call names the kubeconfig file and context
# of one cluster, so the default kubeconfig is never read or changed.

cluster_name() {  # cluster_name KEY: the kind cluster of hub, a or b
  printf '%s-%s' "${E2E_PREFIX}" "$1"
}

kubeconfig_of() {
  printf '%s/%s' "${KUBECONFIG_DIR}" "$(cluster_name "$1")"
}

# kc KEY ARGS...: kubectl against one cluster.
kc() {
  local key="$1"
  shift
  kubectl --kubeconfig "$(kubeconfig_of "${key}")" --context "kind-$(cluster_name "${key}")" "$@"
}

# hm KEY ARGS...: helm against one cluster.
hm() {
  local key="$1"
  shift
  helm --kubeconfig "$(kubeconfig_of "${key}")" --kube-context "kind-$(cluster_name "${key}")" "$@"
}

# kjson KEY ARGS...: kubectl get ... -o json, or empty when the call fails.
kjson() {
  local key="$1"
  shift
  kc "${key}" --request-timeout=10s get "$@" -o json 2>/dev/null || true
}

node_container() {  # node_container KEY: the Docker container of the node
  printf '%s-control-plane' "$(cluster_name "$1")"
}

node_ip() {  # node_ip KEY: the node's address on KIND_NETWORK
  docker inspect -f "{{with index .NetworkSettings.Networks \"${KIND_NETWORK}\"}}{{.IPAddress}}{{end}}" \
    "$(node_container "$1")" 2>/dev/null
}

has_cluster() {  # has_cluster KEY: the run has this cluster
  case " ${CLUSTER_KEYS} " in
    *" $1 "*) return 0 ;;
  esac
  return 1
}

cluster_exists() {  # cluster_exists KEY
  "${KIND}" get clusters 2>/dev/null | grep -qx -- "$(cluster_name "$1")"
}

# ---------------------------------------------------------------------------
# Objects

# object_present KEY NAMESPACE KIND/NAME
object_present() {
  kc "$1" --request-timeout=10s -n "$2" get "$3" >/dev/null 2>&1 && return 0
  LAST_OBSERVED="$3 absent in $1/$2"
  return 1
}

# object_absent KEY NAMESPACE KIND/NAME
object_absent() {
  if kc "$1" --request-timeout=10s -n "$2" get "$3" >/dev/null 2>&1; then
    LAST_OBSERVED="$3 still exists in $1/$2"
    return 1
  fi
  LAST_OBSERVED="$3 absent in $1/$2"
}

# reset_namespace NAMESPACE: deletes and re-creates the namespace in every
# cluster, so a phase starts clean. The controllers are running, so the
# ServiceExport finalizers deregister the exports.
reset_namespace() {
  local ns="$1" key
  for key in ${CLUSTER_KEYS}; do
    kc "${key}" delete namespace "${ns}" --ignore-not-found --wait=false >/dev/null 2>&1 || true
  done
  for key in ${CLUSTER_KEYS}; do
    if ! poll 180 object_absent "${key}" default "namespace/${ns}"; then
      die "namespace ${ns} in ${key} did not terminate: ${LAST_OBSERVED}"
    fi
  done
  for key in ${CLUSTER_KEYS}; do
    kc "${key}" create namespace "${ns}" >/dev/null || die "cannot create namespace ${ns} in ${key}"
  done
}

# apply KEY NAMESPACE FILE: applies a manifest with its placeholders replaced.
apply() {
  sed -e "s|__BUSYBOX_IMAGE__|${BUSYBOX_IMAGE}|g" \
    -e "s|__GDS_IMAGE__|${GDS_IMAGE_REPO}:${GDS_IMAGE_TAG}|g" "$3" |
    kc "$1" -n "$2" apply -f - >/dev/null
}

label_service() {  # label_service KEY NAMESPACE NAME LABEL...
  local key="$1" ns="$2" name="$3"
  shift 3
  kc "${key}" -n "${ns}" label service "${name}" "$@" --overwrite >/dev/null 2>"${E2E_WORK_DIR}/kubectl.err" ||
    { LAST_OBSERVED="$(cat "${E2E_WORK_DIR}/kubectl.err")"; return 1; }
}

scale() {  # scale KEY NAMESPACE KIND/NAME REPLICAS
  kc "$1" -n "$2" scale "$3" --replicas="$4" >/dev/null 2>"${E2E_WORK_DIR}/kubectl.err" ||
    { LAST_OBSERVED="$(cat "${E2E_WORK_DIR}/kubectl.err")"; return 1; }
}

# rollout_ready KEY NAMESPACE KIND/NAME: the workload is rolled out.
rollout_ready() {
  kc "$1" -n "$2" rollout status "$3" --timeout=180s >/dev/null 2>&1 ||
    { LAST_OBSERVED="$3 in $1/$2 is not rolled out"; return 1; }
}

# ---------------------------------------------------------------------------
# Registry REST, through the API server's Service proxy of the hub, so no
# port-forward is needed. Sets REST_STATUS (0 or the kubectl failure) and
# REST_BODY.

REST_STATUS=""
REST_BODY=""

registry_get() {  # registry_get PATH
  local err="${E2E_WORK_DIR}/rest.err"
  if REST_BODY="$(kc hub --request-timeout=15s get --raw "/api/v1/namespaces/${MCS_NAMESPACE}/services/mcs-registry:http/proxy$1" 2>"${err}")"; then
    REST_STATUS=0
  else
    REST_STATUS=1
    REST_BODY="$(cat "${err}")"
  fi
}

# registry_admin PATH: the admin port (/readyz).
registry_admin() {
  kc hub --request-timeout=15s get --raw "/api/v1/namespaces/${MCS_NAMESPACE}/services/mcs-registry:admin/proxy$1" 2>&1
}

# rest_json FILTER: jq on REST_BODY, raw output.
rest_json() {
  printf '%s' "${REST_BODY}" | jq -r "$1" 2>/dev/null
}

CS_PATH="/v1/clustersets/${E2E_CLUSTERSET}"

# cluster_list: the cluster keys of this run, sorted and comma-separated, as
# the registry lists them.
cluster_list() {
  tr ' ' '\n' <<<"${CLUSTER_KEYS}" | sort | paste -sd, -
}
