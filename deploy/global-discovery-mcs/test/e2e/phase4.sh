# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# shellcheck shell=bash
# shellcheck disable=SC2034  # variables are shared with the other sourced files
#
# Phase 4: continuous repair and leader takeover. The controllers run with
# short resync periods (E2E_RESYNC_PERIOD). With every controller left alone
# (no restart, reconnect or new registration), what an operator or a failure
# removes or edits is restored within one resync period: a managed
# EndpointSlice, a derived Service, a registry export record (registration
# kept) and a registry endpoint set. Cluster a runs two controller replicas:
# the standby makes no registry writes, and when the leader pod is deleted the
# Lease moves and the exports and imports converge. Namespace e2e-p4.

P4_NS="e2e-p4"
P4_METRICS_PORT=8080
M_="dynamo_mcs_controller_"
# Set by p4_replicas.
P4_LEADER=""
P4_STANDBY=""
P4_NEW_LEADER=""
P4_STATE_BEFORE=""
# What a repair step damaged: the resourceVersion of the edited slice and the
# UID of the deleted Service, so that the check sees a different object.
P4_EDIT_SLICE=""
P4_EDIT_RV=""
P4_DELETED_UID=""

# expect_repair NAME CHECK [ARGS...]: like expect, with the time of a repair:
# one resync period plus the propagation (E2E_REPAIR_TIMEOUT).
expect_repair() {
  local name="$1" saved="${E2E_TIMEOUT}" rc=0
  shift
  E2E_TIMEOUT="${E2E_REPAIR_TIMEOUT}"
  expect "${name}" "$@" || rc=$?
  E2E_TIMEOUT="${saved}"
  return "${rc}"
}

# controller_pods KEY: the names of the Running controller pods, sorted.
controller_pods() {
  kc_names "$1" -n "${MCS_NAMESPACE}" pods -l app.kubernetes.io/component=controller \
    --field-selector=status.phase=Running || return 1
  printf '%s\n' "${KC_NAMES//pod\//}" | sort
}

# lease_holder_pod KEY: the pod that holds the leader election Lease
# (controller-runtime names the holder <pod>_<uuid>).
lease_holder_pod() {
  local holder
  holder="$(kc "$1" --request-timeout="${E2E_REQUEST_TIMEOUT}" -n "${MCS_NAMESPACE}" get lease "${LEADER_LEASE}" \
    -o jsonpath='{.spec.holderIdentity}' 2>/dev/null)" || return 1
  [ -n "${holder}" ] || return 1
  printf '%s' "${holder%%_*}"
}

# pod_metric KEY POD NAME: the sum of the series of the metric NAME of a pod's
# /metrics (read through the API server's pod proxy); fails when the pod has
# no such series.
pod_metric() {
  local body
  body="$(kc "$1" --request-timeout=15s get --raw "/api/v1/namespaces/${MCS_NAMESPACE}/pods/$2:${P4_METRICS_PORT}/proxy/metrics" 2>/dev/null)" || return 1
  awk -v m="$3" '
    $1 == m || index($1, m "{") == 1 { sum += $NF; n++ }
    END { if (n == 0) exit 1; printf "%d\n", sum }' <<<"${body}"
}

# pod_series KEY POD NAME: the non-zero series of a metric of a pod, as
# "name{labels} value" lines (for the log).
pod_series() {
  kc "$1" --request-timeout=15s get --raw "/api/v1/namespaces/${MCS_NAMESPACE}/pods/$2:${P4_METRICS_PORT}/proxy/metrics" 2>/dev/null |
    awk -v m="$3" '(index($1, m "{") == 1 || $1 == m) && $NF != 0 { print $1 " " $NF }'
}

# p4_replicas: cluster a has 2 Running controller pods, the Lease names one of
# them. Sets P4_LEADER and P4_STANDBY.
p4_replicas() {
  local pods leader
  pods="$(controller_pods a)" || return 1
  leader="$(lease_holder_pod a)" || { LAST_OBSERVED="no Lease holder; controller pods $(tr '\n' ' ' <<<"${pods}")"; return 1; }
  LAST_OBSERVED="Lease holder ${leader}; running controller pods $(tr '\n' ' ' <<<"${pods}")"
  [ "$(wc -l <<<"${pods}" | tr -d ' ')" = 2 ] && grep -qx -- "${leader}" <<<"${pods}" || return 1
  P4_LEADER="${leader}"
  P4_STANDBY="$(grep -vx -- "${leader}" <<<"${pods}")"
}

# p4_standby_idle: the standby made no registry write and holds no
# registration; the leader registered the cluster and wrote the exports.
p4_standby_idle() {
  local m writes="" leader_writes
  for m in "${M_}export_writes_total" "${M_}endpoint_replacements_total" "${M_}registrations_total" "${M_}cluster_registered" leader_election_master_status; do
    writes="${writes} ${m#"${M_}"}=$(pod_metric a "${P4_STANDBY}" "$m" 2>/dev/null || printf 'none')"
  done
  leader_writes="$(pod_metric a "${P4_LEADER}" "${M_}export_writes_total" 2>/dev/null || printf 'none')"
  LAST_OBSERVED="standby ${P4_STANDBY}:${writes}; leader ${P4_LEADER}: export_writes_total=${leader_writes}, cluster_registered=$(pod_metric a "${P4_LEADER}" "${M_}cluster_registered" 2>/dev/null || printf 'none')"
  [ "${writes}" = " export_writes_total=0 endpoint_replacements_total=0 registrations_total=0 cluster_registered=0 leader_election_master_status=0" ] || return 1
  [ "${leader_writes}" != none ] && [ "${leader_writes}" -gt 0 ] &&
    [ "$(pod_metric a "${P4_LEADER}" "${M_}cluster_registered" 2>/dev/null)" = 1 ]
}

# p4_lease_moved: the Lease is held by a Running controller pod other than the
# deleted leader. Sets P4_NEW_LEADER.
p4_lease_moved() {
  local holder pods
  holder="$(lease_holder_pod a)" || { LAST_OBSERVED="no Lease holder"; return 1; }
  pods="$(controller_pods a)" || return 1
  LAST_OBSERVED="Lease holder ${holder} (deleted leader ${P4_LEADER}); running controller pods $(tr '\n' ' ' <<<"${pods}")"
  [ "${holder}" != "${P4_LEADER}" ] && grep -qx -- "${holder}" <<<"${pods}" || return 1
  P4_NEW_LEADER="${holder}"
}

# p4_new_leader_registered: the leader holds a registered cluster lease.
p4_new_leader_registered() {
  local got
  got="$(pod_metric a "${P4_NEW_LEADER}" "${M_}cluster_registered" 2>/dev/null)" || got="none"
  LAST_OBSERVED="${M_}cluster_registered of ${P4_NEW_LEADER}: ${got}"
  [ "${got}" = 1 ]
}

# p4_takeover_resyncs: a takeover is one resync of the new leader, the startup
# one; the registration and the registry reconnects it sees do not add more.
p4_takeover_resyncs() {
  local series
  series="$(pod_series a "${P4_NEW_LEADER}" "${M_}resyncs_total")" || return 1
  LAST_OBSERVED="${M_}resyncs_total of ${P4_NEW_LEADER}: $(tr '\n' ' ' <<<"${series}")"
  grep -qx "${M_}resyncs_total{reason=\"startup\"} 1" <<<"${series}" &&
    ! grep -q 'reason="\(reconnect\|reregister\)"' <<<"${series}"
}

# controller_state KEY: what a controller keeps while nothing disturbs it, one
# line per pod: name, UID, restart count, registry client reconnects and
# registrations (resyncs do not count: the periodic ones are expected).
controller_state() {
  local key="$1" rows pod line
  rows="$(kc "${key}" --request-timeout=15s -n "${MCS_NAMESPACE}" get pods -l app.kubernetes.io/component=controller \
    -o jsonpath='{range .items[*]}{.metadata.name} {.metadata.uid} {.status.containerStatuses[0].restartCount}{"\n"}{end}' 2>/dev/null)" || return 1
  [ -n "${rows}" ] || return 1
  while read -r pod line; do
    printf '%s %s reconnects=%s registrations=%s\n' "${pod}" "${line}" \
      "$(pod_metric "${key}" "${pod}" "${M_}stream_reconnects_total" || printf 'none')" \
      "$(pod_metric "${key}" "${pod}" "${M_}registrations_total" || printf 'none')"
  done <<<"${rows}"
}

p4_state_all() {
  local key
  for key in a b; do
    printf '%s:\n' "${key}"
    controller_state "${key}" || return 1
  done
}

p4_undisturbed() {
  local now
  now="$(p4_state_all)" || { LAST_OBSERVED="cannot read the controller state"; return 1; }
  LAST_OBSERVED="before: $(tr '\n' ';' <<<"${P4_STATE_BEFORE}") now: $(tr '\n' ';' <<<"${now}")"
  [ "${now}" = "${P4_STATE_BEFORE}" ]
}

# ---------------------------------------------------------------------------
# Registry writes through the hub's Service proxy, as an operator would.

registry_path() {
  printf '/api/v1/namespaces/%s/services/mcs-registry:http/proxy%s' "${MCS_NAMESPACE}" "$1"
}

# registry_delete_export CLUSTER NS NAME: deletes the export record of a
# cluster; its registration and lease stay.
registry_delete_export() {
  local err="${E2E_WORK_DIR}/rest.err"
  kc hub --request-timeout=15s delete --raw "$(registry_path "${CS_PATH}/clusters/$1/exports/$2/$3")" >/dev/null 2>"${err}" ||
    { LAST_OBSERVED="DELETE export $1/$2/$3 failed: $(head -c 200 "${err}")"; return 1; }
}

# registry_clear_endpoints CLUSTER NS NAME: replaces the endpoint set of an
# export with the empty set at the next generation, leaving the export record.
registry_clear_endpoints() {
  local err="${E2E_WORK_DIR}/rest.err" gen
  registry_get "${CS_PATH}/clusters/$1/exports"
  gen="$(jq -r --arg ns "$2" --arg n "$3" \
    '[.exports[]? | select(.namespace == $ns and .name == $n) | (.endpointsGeneration // "0" | tonumber)][0] // ""' <<<"${REST_BODY}" 2>/dev/null)" || gen=""
  case "${gen}" in
    '' | *[!0-9]*) LAST_OBSERVED="no export $1/$2/$3 in the registry: $(head -c 200 <<<"${REST_BODY}")"; return 1 ;;
  esac
  printf '{"endpoints":[],"generation":"%s"}' "$((gen + 1))" |
    kc hub --request-timeout=15s replace --raw "$(registry_path "${CS_PATH}/clusters/$1/exports/$2/$3/endpoints")" -f - >/dev/null 2>"${err}" ||
    { LAST_OBSERVED="PUT endpoints $1/$2/$3 failed: $(head -c 200 "${err}")"; return 1; }
}

# ---------------------------------------------------------------------------

p4_slices() {  # p4_slices SERVICE SOURCE SELECTOR COUNT: every cluster imports them
  local key
  for key in ${CLUSTER_KEYS}; do
    slices_match "${key}" "${P4_NS}" "$1" "$2" "$3" "$4" || return 1
  done
}

p4_imports_ok() {  # every cluster has the ServiceImport and its derived Service
  local key
  for key in ${CLUSTER_KEYS}; do
    serviceimport_ok "${key}" "${P4_NS}" "$1" || return 1
  done
}

p4_start() {
  apply a "${P4_NS}" "${MANIFESTS}/echo.yaml" || return 1
  apply b "${P4_NS}" "${MANIFESTS}/bonly.yaml" || return 1
  apply b "${P4_NS}" "${MANIFESTS}/client.yaml" || return 1
  rollout_ready a "${P4_NS}" deploy/echo || return 1
  rollout_ready b "${P4_NS}" deploy/bonly || return 1
  kc b -n "${P4_NS}" wait --for=condition=Ready pod/client --timeout=120s >/dev/null 2>&1 ||
    { LAST_OBSERVED="client pod in b is not Ready"; return 1; }
  label_service a "${P4_NS}" echo dynamo.nvidia.com/mcs-export=true
}

p4_edit_slice() {  # edits the address of the first endpoint of an imported echo slice in b
  local slice
  kc_names b -n "${P4_NS}" endpointslices -l "multicluster.kubernetes.io/service-name=echo,multicluster.kubernetes.io/source-cluster=a" || return 1
  slice="$(head -n 1 <<<"${KC_NAMES}")"
  [ -n "${slice}" ] || { LAST_OBSERVED="no imported slice of echo in b"; return 1; }
  P4_EDIT_RV="$(kc b -n "${P4_NS}" patch "${slice}" --type=json -o jsonpath='{.metadata.resourceVersion}' \
    -p '[{"op":"replace","path":"/endpoints/0/addresses/0","value":"192.0.2.1"}]' 2>"${E2E_WORK_DIR}/kubectl.err")" ||
    { LAST_OBSERVED="$(cat "${E2E_WORK_DIR}/kubectl.err")"; return 1; }
  [ -n "${P4_EDIT_RV}" ] || { LAST_OBSERVED="the patch returned no resourceVersion"; return 1; }
  P4_EDIT_SLICE="${slice}"
}

# p4_slice_restored: the edited slice holds the ready pod addresses again and
# was written after the edit (its resourceVersion moved on).
p4_slice_restored() {
  local rv
  slices_match b "${P4_NS}" echo a app=echo 2 || return 1
  rv="$(kc b --request-timeout="${E2E_REQUEST_TIMEOUT}" -n "${P4_NS}" get "${P4_EDIT_SLICE}" -o jsonpath='{.metadata.resourceVersion}' 2>/dev/null)" || rv=""
  LAST_OBSERVED="${P4_EDIT_SLICE} in b: resourceVersion ${rv:-none} after the edit at ${P4_EDIT_RV}"
  [ -n "${rv}" ] && [ "${rv}" != "${P4_EDIT_RV}" ]
}

p4_delete_derived() {  # deletes the derived Service of echo in b
  serviceimport_ok b "${P4_NS}" echo || return 1
  P4_DELETED_UID="$(kc b -n "${P4_NS}" get service "${DERIVED_NAME}" -o jsonpath='{.metadata.uid}' 2>/dev/null)" || P4_DELETED_UID=""
  [ -n "${P4_DELETED_UID}" ] || { LAST_OBSERVED="cannot read the UID of ${DERIVED_NAME}"; return 1; }
  kc b -n "${P4_NS}" delete service "${DERIVED_NAME}" --wait=true --timeout=60s >/dev/null 2>"${E2E_WORK_DIR}/kubectl.err" ||
    { LAST_OBSERVED="$(cat "${E2E_WORK_DIR}/kubectl.err")"; return 1; }
}

# p4_service_recreated: a derived Service with another UID than the deleted
# one, and the ServiceImport points at its ClusterIP.
p4_service_recreated() {
  local uid
  serviceimport_ok b "${P4_NS}" echo || return 1
  uid="$(kc b --request-timeout="${E2E_REQUEST_TIMEOUT}" -n "${P4_NS}" get service "${DERIVED_NAME}" -o jsonpath='{.metadata.uid}' 2>/dev/null)" || uid=""
  LAST_OBSERVED="${LAST_OBSERVED}; UID ${uid:-none}, deleted UID ${P4_DELETED_UID}"
  [ -n "${uid}" ] && [ "${uid}" != "${P4_DELETED_UID}" ]
}

# p4_scale KEY DEPLOYMENT SELECTOR REPLICAS: scales and waits until exactly
# that many pods are ready, so the expected counts are exact.
p4_scale() {
  scale "$1" "${P4_NS}" "deploy/$2" "$4" || return 1
  rollout_ready "$1" "${P4_NS}" "deploy/$2" || return 1
  poll 60 p4_pods_are "$1" "$3" "$4"
}

p4_pods_are() {
  local n
  n="$(pods_json "$1" "${P4_NS}" "$2" | jq 'length')"
  LAST_OBSERVED="${n} ready pods of $2 in $1"
  [ "${n}" = "$3" ]
}

p4_scale_both() {  # p4_scale_both ECHO_REPLICAS BONLY_REPLICAS
  p4_scale a echo app=echo "$1" || return 1
  p4_scale b bonly app=bonly "$2"
}

run_phase_4() {
  section "Phase 4: continuous repair and leader takeover"
  if ! has_cluster hub || ! has_cluster a || ! has_cluster b; then
    skip "phase 4 needs the clusters hub, a and b"
    return 0
  fi
  reset_namespace "${P4_NS}"

  once "echo (2 replicas) and Service echo in a, bonly (1 pod) in b, a client pod in b; both Services exported by their label" \
    p4_start || return 0
  expect "every cluster imports the 2 ready echo pods of a" p4_slices echo a app=echo 2 || true
  expect "every cluster imports the pod of bonly in b" p4_slices bonly b app=bonly 1 || true
  expect "registry import echo: cluster a with 2 ready endpoints" registry_import_ready "${P4_NS}" echo "a:2" || true
  expect "cluster a runs 2 controller replicas, one of them holds the Lease" p4_replicas || return 0
  expect "the standby made no registry write and holds no registration; the leader registered a and wrote the exports" p4_standby_idle || true

  section "Phase 4: continuous repair (resync period ${E2E_RESYNC_PERIOD}s, repair bound ${E2E_REPAIR_TIMEOUT}s)"
  if ! P4_STATE_BEFORE="$(p4_state_all)"; then
    fail "read the state of the controllers of a and b" "kubectl or the metrics of a controller pod failed"
    return 0
  fi

  once "edit an imported EndpointSlice of echo in b: its first endpoint gets the address 192.0.2.1" p4_edit_slice || true
  expect_repair "the EndpointSlices of echo from a in b are restored (addresses, labels, targetRefs; the edited slice was written again)" p4_slice_restored || true

  once "delete the derived Service of echo in b" p4_delete_derived || true
  expect_repair "the derived Service of echo in b is re-created (new UID) and spec.ips of the ServiceImport is its ClusterIP" p4_service_recreated || true
  expect "a pod in b reaches echo in a through the new derived ClusterIP" curl_derived b "${P4_NS}" "${DERIVED_IP}" 8080 || true

  once "delete the registry export record a/${P4_NS}/echo (the registration of a stays)" registry_delete_export a "${P4_NS}" echo || true
  expect "cluster a still holds its ACTIVE lease" cluster_lease_is a LEASE_STATE_ACTIVE "$(region_of a)" || true
  expect_repair "registry import echo: the record and the 2 ready endpoints of a are back" registry_import_ready "${P4_NS}" echo "a:2" || true
  expect_repair "every cluster imports the 2 ready echo pods of a again" p4_slices echo a app=echo 2 || true
  expect_repair "every cluster has the ServiceImport echo and its derived Service again" p4_imports_ok echo || true

  once "replace the endpoint set of a/${P4_NS}/echo in the registry with the empty set (the record stays)" registry_clear_endpoints a "${P4_NS}" echo || true
  expect_repair "registry import echo: the 2 ready endpoints of a are back" registry_import_ready "${P4_NS}" echo "a:2" || true
  expect_repair "every cluster imports the 2 ready echo pods of a again (endpoint set)" p4_slices echo a app=echo 2 || true

  expect "the controllers of a and b were not restarted, did not reconnect and did not register again" p4_undisturbed || true

  section "Phase 4: leader takeover in cluster a"
  expect "cluster a: two controller pods, the Lease names one" p4_replicas || return 0
  once "delete the leader pod ${P4_LEADER}" \
    kc a -n "${MCS_NAMESPACE}" delete pod "${P4_LEADER}" --wait=false >/dev/null || return 0
  expect "the Lease moves to another running controller pod" p4_lease_moved || return 0
  if [ "${P4_NEW_LEADER}" = "${P4_STANDBY}" ]; then
    log "the standby ${P4_STANDBY} took over"
  else
    log "the replacement pod ${P4_NEW_LEADER} took over; the standby was ${P4_STANDBY}"
  fi
  expect "the new leader holds a registered cluster lease (cluster_registered 1)" p4_new_leader_registered || true
  expect "registry: cluster a holds an ACTIVE lease and region $(region_of a)" cluster_lease_is a LEASE_STATE_ACTIVE "$(region_of a)" || true
  expect "registry import echo: cluster a with 2 ready endpoints" registry_import_ready "${P4_NS}" echo "a:2" || true
  expect "every cluster still imports the 2 ready echo pods of a" p4_slices echo a app=echo 2 || true
  expect "every cluster still imports the pod of bonly in b (a imports it through its new leader)" p4_slices bonly b app=bonly 1 || true
  once "scale echo in a to 3 and bonly in b to 2 replicas" p4_scale_both 3 2 || true
  expect "the new leader exports the 3 echo pods of a: registry import and every cluster" p4_converged echo a app=echo 3 || true
  expect "the new leader applies the 2 bonly pods of b: registry import and every cluster" p4_converged bonly b app=bonly 2 || true
  expect "the takeover was one startup resync of the new leader, with no reconnect or reregister resync" p4_takeover_resyncs || true
  expect "cluster a runs 2 controller replicas again, one of them holds the Lease" p4_replicas || true
}

# p4_converged SERVICE SOURCE SELECTOR COUNT: the registry's import lists COUNT
# ready endpoints of SOURCE, and every cluster imports them.
p4_converged() {
  registry_import_ready "${P4_NS}" "$1" "$2:$4" || return 1
  p4_slices "$1" "$2" "$3" "$4"
}
