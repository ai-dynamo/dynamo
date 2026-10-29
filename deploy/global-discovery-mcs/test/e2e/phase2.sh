# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# shellcheck shell=bash
# shellcheck disable=SC2034  # variables are shared with the other sourced files
#
# Phase 2: the life of an export. Endpoint updates when the backing Deployment
# scales, a ServiceExport of the v1alpha1 version, a conflicting export from b
# (the oldest export wins), the lease expiry of b and its recovery, and the
# deletion of exports. Namespace e2e-p2.

P2_NS="e2e-p2"

# p2_slices COUNT: every cluster imports the COUNT ready echo pods of a.
p2_slices() {
  local key count="$1"
  for key in ${CLUSTER_KEYS}; do
    slices_match "${key}" "${P2_NS}" "${2:-echo}" a app=echo "${count}" || return 1
  done
}

p2_apply_echo() {
  apply a "${P2_NS}" "${MANIFESTS}/echo.yaml" || return 1
  rollout_ready a "${P2_NS}" deploy/echo
}

# p2_scale_echo REPLICAS: scales the Deployment and waits until it has only
# ready pods of that number, so the expected counts are exact.
p2_scale_echo() {
  scale a "${P2_NS}" deploy/echo "$1" || return 1
  rollout_ready a "${P2_NS}" deploy/echo || return 1
  poll 60 p2_pods_are "$1"
}

p2_pods_are() {
  local n
  n="$(pods_json a "${P2_NS}" app=echo | jq 'length')"
  LAST_OBSERVED="${n} ready echo pods in a"
  [ "${n}" = "$1" ]
}

run_phase_2() {
  local key derived
  section "Phase 2: the life of an export"
  reset_namespace "${P2_NS}"

  once "echo (2 replicas) and Service echo in a, exported by its label" \
    p2_start || return 0
  expect "every cluster imports the 2 ready echo pods of a" p2_slices 2 || true

  section "Phase 2: endpoint updates"
  once "scale echo in a to 1 replica" p2_scale_echo 1 || true
  expect "every cluster imports 1 endpoint of a" p2_slices 1 || true
  expect "registry import: cluster a with 1 ready endpoint" registry_import_ready "${P2_NS}" echo "a:1" || true
  once "scale echo in a to 3 replicas" p2_scale_echo 3 || true
  expect "every cluster imports 3 endpoints of a" p2_slices 3 || true
  expect "registry import: cluster a with 3 ready endpoints" registry_import_ready "${P2_NS}" echo "a:3" || true
  once "scale echo in a back to 2 replicas" p2_scale_echo 2 || true
  expect "every cluster imports 2 endpoints of a" p2_slices 2 || true

  section "Phase 2: ServiceExport of the v1alpha1 version"
  once "create Service echo-alpha and its ServiceExport as multicluster.x-k8s.io/v1alpha1 in a" \
    apply a "${P2_NS}" "${MANIFESTS}/alpha.yaml" || true
  expect "ServiceExport echo-alpha in a is Valid=True and Ready=True" export_ok a "${P2_NS}" echo-alpha || true
  for key in ${CLUSTER_KEYS}; do
    expect "ServiceImport echo-alpha in ${key}: ClusterSetIP with a derived Service" serviceimport_ok "${key}" "${P2_NS}" echo-alpha || true
    expect "EndpointSlices of echo-alpha in ${key}: the 2 echo pods of a" slices_match "${key}" "${P2_NS}" echo-alpha a app=echo 2 || true
  done

  section "Phase 2: conflicting exports"
  if has_cluster b; then
    once "export Service shared (port http/8080) from a" apply a "${P2_NS}" "${MANIFESTS}/shared-a.yaml" || true
    expect "ServiceExport shared in a is Valid=True and Ready=True" export_ok a "${P2_NS}" shared || true
    expect "registry import shared: no conflict, port http/8080" registry_import_conflict "${P2_NS}" shared none - http/8080 || true
    once "export Service shared (port http/9090) from b, later than a" apply b "${P2_NS}" "${MANIFESTS}/shared-b.yaml" || true
    expect "the ServiceExport shared in a is older than the one in b" export_older a b "${P2_NS}" shared || true
    expect "ServiceExport shared in b: Conflict=True, reason PortConflict" export_condition b "${P2_NS}" shared Conflict True PortConflict || true
    expect "registry import shared: PortConflict, the oldest export (a) wins, port http/8080 only" \
      registry_import_conflict "${P2_NS}" shared PortConflict a http/8080 || true
    expect "ServiceExport shared in a: Conflict=True, reason PortConflict (every participating cluster, as KEP-1645 asks)" \
      export_condition a "${P2_NS}" shared Conflict True PortConflict || true
    for key in ${CLUSTER_KEYS}; do
      expect "ServiceImport shared in ${key} keeps the winner's port http/8080" import_ports_are "${key}" "${P2_NS}" shared http/8080 || true
    done
    once "delete Service shared in b" kc b -n "${P2_NS}" delete service shared --wait=true >/dev/null || true
    expect "ServiceExport shared in b is deleted with its Service" object_absent b "${P2_NS}" serviceexport/shared || true
    expect "registry import shared: conflict cleared" registry_import_conflict "${P2_NS}" shared none - http/8080 || true
    expect "ServiceExport shared in a: Conflict=False" export_condition a "${P2_NS}" shared Conflict False || true
  else
    skip "conflicting export from b: cluster b is not part of this run"
  fi

  section "Phase 2: lease expiry"
  if has_cluster b; then
    once "Deployment bonly (1 pod) and Service bonly in b, exported by its label" p2_start_bonly || true
    for key in ${CLUSTER_KEYS}; do
      [ "${key}" = b ] && continue
      expect "EndpointSlices of bonly in ${key}: the pod of b, labelled region-b" slices_match "${key}" "${P2_NS}" bonly b app=bonly 1 || true
    done
    once "stop the controller of b (scale to 0)" scale b "${MCS_NAMESPACE}" deploy/mcs-controller 0 || true
    expect "registry: cluster b's lease is EXPIRED once TTL (10s) and skew pass" cluster_lease_is b LEASE_STATE_EXPIRED || true
    expect "registry import bonly: cluster b no longer participates" registry_cluster_participates "${P2_NS}" bonly b false || true
    for key in ${CLUSTER_KEYS}; do
      [ "${key}" = b ] && continue
      expect "EndpointSlices of bonly from b are removed in ${key}" no_slices_from "${key}" "${P2_NS}" bonly b || true
    done
    once "start the controller of b again (scale to 1)" p2_start_controller_b || true
    expect "registry: cluster b registers again with an ACTIVE lease" cluster_lease_is b LEASE_STATE_ACTIVE region-b || true
    expect "registry import bonly: cluster b participates again" registry_cluster_participates "${P2_NS}" bonly b true || true
    for key in ${CLUSTER_KEYS}; do
      [ "${key}" = b ] && continue
      expect "EndpointSlices of bonly from b are back in ${key}" slices_match "${key}" "${P2_NS}" bonly b app=bonly 1 || true
    done
  else
    skip "lease expiry of b: cluster b is not part of this run"
  fi

  section "Phase 2: deleting exports"
  serviceimport_ok b "${P2_NS}" echo >/dev/null 2>&1 || serviceimport_ok hub "${P2_NS}" echo >/dev/null 2>&1 || true
  derived="${DERIVED_NAME}"
  once "remove the export label from Service echo in a" label_service a "${P2_NS}" echo dynamo.nvidia.com/mcs-export- || true
  expect "ServiceExport echo in a is deleted" object_absent a "${P2_NS}" serviceexport/echo || true
  for key in ${CLUSTER_KEYS}; do
    expect "ServiceImport echo, its derived Service and its slices are gone from ${key}" import_gone "${key}" "${P2_NS}" echo "${derived}" || true
  done
  expect "registry no longer has the import echo" registry_import_absent "${P2_NS}" echo || true
  serviceimport_ok hub "${P2_NS}" echo-alpha >/dev/null 2>&1 || true
  derived="${DERIVED_NAME}"
  once "delete the v1alpha1 ServiceExport echo-alpha in a" kc a -n "${P2_NS}" delete serviceexport echo-alpha --wait=true --timeout=60s >/dev/null || true
  for key in ${CLUSTER_KEYS}; do
    expect "ServiceImport echo-alpha, its derived Service and its slices are gone from ${key}" import_gone "${key}" "${P2_NS}" echo-alpha "${derived}" || true
  done
  expect "registry no longer has the import echo-alpha" registry_import_absent "${P2_NS}" echo-alpha || true
}

p2_start() {
  p2_apply_echo || return 1
  label_service a "${P2_NS}" echo dynamo.nvidia.com/mcs-export=true
}

p2_start_bonly() {
  apply b "${P2_NS}" "${MANIFESTS}/bonly.yaml" || return 1
  rollout_ready b "${P2_NS}" deploy/bonly
}

p2_start_controller_b() {
  scale b "${MCS_NAMESPACE}" deploy/mcs-controller 1 || return 1
  rollout_ready b "${MCS_NAMESPACE}" deploy/mcs-controller
}
