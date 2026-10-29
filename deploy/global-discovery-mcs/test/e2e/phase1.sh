# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# shellcheck shell=bash
# shellcheck disable=SC2034  # variables are shared with the other sourced files
#
# Phase 1: registration, export and import. The registry in the hub, the
# controllers of hub, a and b registered with it, and a Service of a exported
# by its label and imported everywhere as a ServiceImport, a derived Service
# and EndpointSlices. Namespace e2e-p1.

P1_NS="e2e-p1"

run_phase_1() {
  local key src_selector="app=echo"
  section "Phase 1: registration, export and import"
  reset_namespace "${P1_NS}"

  expect "registry /readyz is ok" registry_ready || true
  expect "registry lists the clusters $(cluster_list) of clusterset ${E2E_CLUSTERSET}" clusters_are "$(cluster_list)" || true
  for key in ${CLUSTER_KEYS}; do
    expect "cluster ${key} holds an ACTIVE lease and registered region $(region_of "${key}")" \
      cluster_lease_is "${key}" LEASE_STATE_ACTIVE "$(region_of "${key}")" || true
    expect "controller of ${key} holds its leader election Lease" leader_held "${key}" || true
  done

  once "echo (2 replicas) and Service echo in a, a client pod in every other cluster, namespace ${P1_NS} in all" \
    p1_apply_workload || return 0
  once "label Service echo in a dynamo.nvidia.com/mcs-export=true" \
    label_service a "${P1_NS}" echo dynamo.nvidia.com/mcs-export=true || return 0
  expect "ServiceExport echo in a is created by the controller" export_managed a "${P1_NS}" echo || true
  expect "ServiceExport echo in a is Valid=True and Ready=True" export_ok a "${P1_NS}" echo || true
  expect "registry import ${P1_NS}/echo: cluster a with 2 ready endpoints" \
    registry_import_ready "${P1_NS}" echo "a:2" || true

  for key in ${CLUSTER_KEYS}; do
    expect "ServiceImport echo in ${key}: ClusterSetIP, spec.ips is the ClusterIP of its derived-<hash> Service" \
      serviceimport_ok "${key}" "${P1_NS}" echo || true
    expect "EndpointSlices in ${key}: the 2 ready echo pods of a by address and targetRef name, KEP labels, region-a" \
      slices_match "${key}" "${P1_NS}" echo a "${src_selector}" 2 || true
  done
  # Every cluster allocates its own ClusterIP for the derived Service, so
  # read it per cluster before probing from that cluster's client pod.
  for key in ${CLUSTER_KEYS}; do
    [ "${key}" = a ] && continue
    serviceimport_ok "${key}" "${P1_NS}" echo || true
    if [ -n "${DERIVED_IP}" ]; then
      expect "a pod in ${key} reaches echo in a through the derived Service ClusterIP" \
        curl_derived "${key}" "${P1_NS}" "${DERIVED_IP}" 8080 || true
    else
      skip "curl through the derived Service in ${key}: the ServiceImport has no ClusterIP"
    fi
  done
}

p1_apply_workload() {
  apply a "${P1_NS}" "${MANIFESTS}/echo.yaml" || return 1
  rollout_ready a "${P1_NS}" deploy/echo || return 1
  local key
  for key in ${CLUSTER_KEYS}; do
    [ "${key}" = a ] && continue
    apply "${key}" "${P1_NS}" "${MANIFESTS}/client.yaml" || return 1
    kc "${key}" -n "${P1_NS}" wait --for=condition=Ready pod/client --timeout=120s >/dev/null 2>&1 ||
      { LAST_OBSERVED="client pod in ${key} is not Ready"; return 1; }
  done
}
