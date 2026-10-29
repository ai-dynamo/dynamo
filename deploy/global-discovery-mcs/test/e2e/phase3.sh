# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# shellcheck shell=bash
# shellcheck disable=SC2034  # variables are shared with the other sourced files
#
# Phase 3: the Global Discovery Service reads what mcs-controller imports. It
# runs in the hub with --pool-discovery mcs; cluster a runs the pool demo (two
# pods, the Services demo-pool-relay and demo-frontend, exported) and the hub
# runs the router Service gr/global-router (two pods, exported). Namespaces
# demo and gr.

GDS_NS="global-discovery"
P3_POOL_NS="demo"
P3_ROUTER_NS="gr"
GDS_BODY=""

# gds_get PATH: GET against the service through the API server's Service
# proxy of the hub. Sets GDS_BODY; fails with the error text as the body.
gds_get() {
  local err="${E2E_WORK_DIR}/gds.err"
  if GDS_BODY="$(kc hub --request-timeout=15s get --raw "/api/v1/namespaces/${GDS_NS}/services/global-discovery-service:http/proxy$1" 2>"${err}")"; then
    return 0
  fi
  GDS_BODY="$(cat "${err}")"
  return 1
}

# gds_json FILTER: jq on GDS_BODY, raw output.
gds_json() {
  printf '%s' "${GDS_BODY}" | jq -r "$1" 2>/dev/null
}

# pool_listed: GET /pools lists exactly one pool, and the list is complete.
pool_listed() {
  gds_get /pools || { LAST_OBSERVED="GET /pools failed: $(head -c 200 <<<"${GDS_BODY}")"; return 1; }
  LAST_OBSERVED="GET /pools: $(jq -c . <<<"${GDS_BODY}" 2>/dev/null | head -c 600)"
  jq -e '.complete == true and (.pools | length) == 1 and .pools[0].key == {cluster_id: "a", namespace: "demo", dgd_name: "demo"}' \
    >/dev/null 2>&1 <<<"${GDS_BODY}"
}

# no_pools: GET /pools is complete and lists nothing.
no_pools() {
  gds_get /pools || { LAST_OBSERVED="GET /pools failed: $(head -c 200 <<<"${GDS_BODY}")"; return 1; }
  LAST_OBSERVED="GET /pools: $(jq -c . <<<"${GDS_BODY}" 2>/dev/null | head -c 600)"
  jq -e '.complete == true and (.pools | length) == 0' >/dev/null 2>&1 <<<"${GDS_BODY}"
}

# lowest_relay_ip: the lowest address of the ready relay pods in a. The
# service binds the relay pod with the lowest address.
lowest_relay_ip() {
  pods_json a "${P3_POOL_NS}" app=demo | jq -r '.[].ip' | sort -t. -k1,1n -k2,2n -k3,3n -k4,4n | head -n 1
}

# relay_identities_are_bound: relay_identities of the pool is exactly
# ["ip:<lowest relay pod address>"]. The service returns that form and the
# Global Router matches relays on it, so a change of the identity contract
# (ai-dynamo/dynamo#17371) must update this assertion.
relay_identities_are_bound() {
  local got ip
  pool_listed || return 1
  got="$(gds_json '.pools[0].relay_identities | sort | join(",")')"
  ip="$(lowest_relay_ip)"
  LAST_OBSERVED="relay_identities [${got}]; lowest relay pod address ${ip:-none}"
  [ -n "${ip}" ] && [ "${got}" = "ip:${ip}" ]
}

# pool_region_seen: the pool has a location.region. Sets POOL_REGION.
POOL_REGION=""
pool_region_seen() {
  pool_listed || return 1
  POOL_REGION="$(gds_json '.pools[0].location.region // ""')"
  LAST_OBSERVED="location.region [${POOL_REGION}]"
  [ -n "${POOL_REGION}" ]
}

# routers_are: GET /routers lists the ready global-router pods of the hub as
# hub/<pod> with http://<pod IP>:<port> URLs, the port of the slice (the
# container port).
routers_are() {
  local want got
  want="$(pods_json hub "${P3_ROUTER_NS}" app=global-router | jq -c 'map({replica: ("hub/" + .name), url: ("http://" + .ip + ":8080")}) | sort_by(.replica)')"
  gds_get /routers || { LAST_OBSERVED="GET /routers failed: $(head -c 200 <<<"${GDS_BODY}")"; return 1; }
  got="$(jq -c '[.routers[]? | {replica, url: (.url | sub("/$"; ""))}] | sort_by(.replica)' <<<"${GDS_BODY}" 2>/dev/null)" || got="<not JSON>"
  LAST_OBSERVED="GET /routers: ${got}; ready router pods: ${want}"
  [ "${want}" != "[]" ] && [ "$(jq 'length' <<<"${want}")" = 2 ] && [ "${got}" = "${want}" ]
}

p3_setup() {
  reset_namespace "${P3_POOL_NS}"
  reset_namespace "${P3_ROUTER_NS}"
  apply hub "${GDS_NS}" "${MANIFESTS}/gds.yaml" || { LAST_OBSERVED="kubectl apply of gds.yaml failed"; return 1; }
  apply hub "${P3_ROUTER_NS}" "${MANIFESTS}/router.yaml" || { LAST_OBSERVED="kubectl apply of router.yaml failed"; return 1; }
  apply a "${P3_POOL_NS}" "${MANIFESTS}/pool.yaml" || { LAST_OBSERVED="kubectl apply of pool.yaml failed"; return 1; }
  rollout_ready hub "${GDS_NS}" deploy/global-discovery-service || return 1
  rollout_ready hub "${P3_ROUTER_NS}" deploy/global-router || return 1
  rollout_ready a "${P3_POOL_NS}" deploy/demo || return 1
}

p3_export() {
  apply a "${P3_POOL_NS}" "${MANIFESTS}/pool-exports.yaml" || { LAST_OBSERVED="kubectl apply of pool-exports.yaml failed"; return 1; }
}

run_phase_3() {
  section "Phase 3: the Global Discovery Service on the imports"
  if ! has_cluster a || ! has_cluster hub; then
    skip "phase 3 needs the clusters hub and a"
    return 0
  fi
  once "Global Discovery Service (--pool-discovery mcs) in the hub, router gr/global-router (2 pods) in the hub, pool demo (2 pods) in a" \
    p3_setup || return 0
  expect "ServiceExport global-router in the hub: Valid=True and Ready=True" \
    p3_export_router || true
  once "ServiceExports demo-pool-relay (v1alpha1) and demo-frontend (v1beta1) in a" p3_export || return 0
  expect "ServiceExport demo-pool-relay in a (applied as v1alpha1) is Valid=True and Ready=True" export_ok a "${P3_POOL_NS}" demo-pool-relay || true
  expect "ServiceExport demo-frontend in a is Valid=True and Ready=True" export_ok a "${P3_POOL_NS}" demo-frontend || true
  expect "ServiceImports demo-pool-relay and demo-frontend in the hub" p3_imports_in_hub || true

  expect "GET /pools lists one complete pool: cluster_id a, namespace demo, dgd_name demo (demo-frontend is not a pool)" pool_listed || true
  if expect "the pool has a location.region" pool_region_seen; then
    case "${POOL_REGION}" in
      region-a) pass "pool location.region is region-a, the topology.kubernetes.io/region of its slices" ;;
      a) gap "pool location.region is region-a" "the service reports the source cluster id (a): it does not read the topology.kubernetes.io/region label that mcs-controller sets on the slices; ai-dynamo/dynamo#17604 adds that read" ;;
      *) fail "pool location.region is region-a" "got ${POOL_REGION}" ;;
    esac
  fi
  expect "pool relay_identities is [ip:<address of the relay pod with the lowest address>]" relay_identities_are_bound || true
  expect "GET /routers lists hub/<pod> for the 2 global-router pods, with http://<pod IP>:8080 URLs" routers_are || true

  once "scale the relay Deployment demo in a to 0" scale a "${P3_POOL_NS}" deploy/demo 0 || true
  expect "GET /pools no longer lists the pool" no_pools || true
  expect "GET /routers still lists the 2 routers" routers_are || true
  once "scale the relay Deployment demo in a back to 2" scale a "${P3_POOL_NS}" deploy/demo 2 || true
  expect "GET /pools lists the pool again" pool_listed || true
  once "delete the ServiceExport demo-pool-relay in a" \
    kc a -n "${P3_POOL_NS}" delete serviceexport demo-pool-relay --wait=true --timeout=60s >/dev/null || true
  expect "GET /pools no longer lists the pool" no_pools || true
  expect "the ServiceImport demo-pool-relay is gone from the hub" object_absent hub "${P3_POOL_NS}" serviceimport/demo-pool-relay || true
  expect "GET /routers still lists the 2 routers" routers_are || true
}

p3_export_router() {
  apply hub "${P3_ROUTER_NS}" "${MANIFESTS}/router.yaml" >/dev/null 2>&1 || true
  export_ok hub "${P3_ROUTER_NS}" global-router
}

p3_imports_in_hub() {
  object_present hub "${P3_POOL_NS}" serviceimport/demo-pool-relay &&
    object_present hub "${P3_POOL_NS}" serviceimport/demo-frontend
}
