# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# shellcheck shell=bash
# shellcheck disable=SC2034  # variables are shared with the other sourced files
#
# Check functions on MCS objects, shared by the phases. Each returns 0 when
# satisfied and sets LAST_OBSERVED; run them through expect or once.

MANAGED_BY_SLICES="mcs-controller.dynamo.nvidia.com"
LEADER_LEASE="mcs-controller.dynamo.nvidia.com"

# registry_ready: /readyz of the registry answers ok.
registry_ready() {
  local got
  got="$(registry_admin /readyz)" || got="failed: ${got}"
  LAST_OBSERVED="GET /readyz: $(head -c 200 <<<"${got}")"
  grep -qi 'ok\|ready' <<<"${got}" && ! grep -qi 'failed\|error' <<<"${got}"
}

# clusters_are EXPECTED: the clusters of the clusterset, comma-separated.
clusters_are() {
  registry_get "${CS_PATH}/clusters"
  local got
  got="$(rest_json '[.clusters[]?.cluster] | sort | join(",")')" || got="<not JSON>"
  LAST_OBSERVED="clusters: ${got:-none}$([ "${REST_STATUS}" = 0 ] || printf ' (%s)' "$(head -c 200 <<<"${REST_BODY}")")"
  [ "${REST_STATUS}" = 0 ] && [ "${got}" = "$1" ]
}

# cluster_lease_is CLUSTER STATE [REGION]: the registry's view of a cluster.
cluster_lease_is() {
  registry_get "${CS_PATH}/clusters/$1"
  local got want="$2"
  got="$(rest_json '.cluster.lease.state')" || got="<not JSON>"
  if [ -n "${3:-}" ]; then
    got="${got} $(rest_json '.cluster.region')"
    want="$2 $3"
  fi
  LAST_OBSERVED="cluster $1: ${got:-none}"
  [ "${REST_STATUS}" = 0 ] && [ "${got}" = "${want}" ]
}

# leader_held KEY: the leader election Lease names a running controller pod.
leader_held() {
  local holder pods
  holder="$(kc "$1" -n "${MCS_NAMESPACE}" get lease "${LEADER_LEASE}" -o jsonpath='{.spec.holderIdentity}' 2>/dev/null)" || holder=""
  pods="$(kc "$1" -n "${MCS_NAMESPACE}" get pods -l app.kubernetes.io/component=controller \
    --field-selector=status.phase=Running -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null)" || pods=""
  LAST_OBSERVED="holder ${holder:-none}; running controller pods $(tr '\n' ' ' <<<"${pods}")"
  [ -n "${holder}" ] || return 1
  # controller-runtime names the holder <pod>_<uuid>.
  grep -qx -- "${holder%%_*}" <<<"${pods}"
}

# export_condition KEY NS NAME TYPE STATUS [REASON]: a condition of the
# ServiceExport.
export_condition() {
  local got
  got="$(jq -r --arg t "$4" '[.status.conditions[]? | select(.type == $t) | "\(.status)/\(.reason)"][0] // "absent"' \
    <<<"$(kjson "$1" -n "$2" serviceexport "$3")" 2>/dev/null)" || got="absent"
  LAST_OBSERVED="ServiceExport $2/$3 in $1: $4=${got}"
  if [ -n "${6:-}" ]; then
    [ "${got}" = "$5/$6" ]
  else
    [ "${got%%/*}" = "$5" ]
  fi
}

# export_ok KEY NS NAME: managed or not, Valid=True and Ready=True.
export_ok() {
  export_condition "$1" "$2" "$3" Valid True || return 1
  export_condition "$1" "$2" "$3" Ready True
}

# export_managed KEY NS NAME: the controller created the ServiceExport from
# the Service's label.
export_managed() {
  local got
  got="$(kc "$1" -n "$2" get serviceexport "$3" -o jsonpath='{.metadata.labels.dynamo\.nvidia\.com/mcs-managed-by}' 2>/dev/null)" || got=""
  LAST_OBSERVED="ServiceExport $2/$3 in $1: managed-by ${got:-none}"
  [ "${got}" = "mcs-controller" ]
}

# Set by serviceimport_ok.
DERIVED_NAME=""
DERIVED_IP=""

# serviceimport_ok KEY NS NAME: a ClusterSetIP ServiceImport whose spec.ips is
# the ClusterIP of its derived-<hash> Service.
serviceimport_ok() {
  local si svc_ip type
  si="$(kjson "$1" -n "$2" serviceimport "$3")"
  [ -n "${si}" ] || si='{}'
  DERIVED_NAME="$(jq -r '.metadata.annotations["multicluster.kubernetes.io/derived-service"] // ""' <<<"${si}")"
  DERIVED_IP="$(jq -r '.spec.ips[0] // ""' <<<"${si}")"
  type="$(jq -r '.spec.type // ""' <<<"${si}")"
  svc_ip=""
  if [ -n "${DERIVED_NAME}" ]; then
    svc_ip="$(kc "$1" -n "$2" get service "${DERIVED_NAME}" -o jsonpath='{.spec.clusterIP}' 2>/dev/null)" || svc_ip=""
  fi
  LAST_OBSERVED="ServiceImport $3 in $1: type ${type:-none}, ips ${DERIVED_IP:-none}, derived ${DERIVED_NAME:-none}, ClusterIP ${svc_ip:-none}"
  [ "${type}" = "ClusterSetIP" ] && [ -n "${DERIVED_IP}" ] && [ "${DERIVED_IP}" = "${svc_ip}" ] &&
    [[ "${DERIVED_NAME}" == derived-* ]]
}

# import_ports_are KEY NS NAME PORTS: the ports of the ServiceImport, as
# name/port,... sorted.
import_ports_are() {
  local got
  got="$(jq -r '[.spec.ports[]? | "\(.name)/\(.port)"] | sort | join(",")' <<<"$(kjson "$1" -n "$2" serviceimport "$3")" 2>/dev/null)" || got=""
  LAST_OBSERVED="ServiceImport $3 in $1: ports ${got:-none}"
  [ "${got}" = "$4" ]
}

# pods_json KEY NS SELECTOR: name and IP of the ready, not terminating pods.
pods_json() {
  jq -c '[.items[]
      | select(.metadata.deletionTimestamp == null)
      | select(any(.status.conditions[]?; .type == "Ready" and .status == "True"))
      | {name: .metadata.name, ip: .status.podIP}] | sort_by(.name)' \
    <<<"$(kjson "$1" -n "$2" pods -l "$3")" 2>/dev/null
}

# slices_match KEY NS SERVICE SOURCE_KEY POD_SELECTOR COUNT: the imported
# EndpointSlices in KEY of SERVICE from SOURCE_KEY hold exactly the COUNT
# ready pods of SOURCE_KEY, by address and by targetRef pod name, carry the
# KEP labels and the source region, and no other slice labelled with the
# source cluster is left.
slices_match() {
  local key="$1" ns="$2" svc="$3" src="$4" selector="$5" count="$6" slices pods got want
  slices="$(kjson "${key}" -n "${ns}" endpointslices \
    -l "multicluster.kubernetes.io/service-name=${svc},multicluster.kubernetes.io/source-cluster=${src}")"
  pods="$(pods_json "${src}" "${ns}" "${selector}")"
  # Name and address of every ready imported endpoint.
  got="$(jq -c '[.items[].endpoints[]? | select(.conditions.ready != false)
      | {name: .targetRef.name, ip: .addresses[0]}] | sort_by(.name)' <<<"${slices}" 2>/dev/null)" || got="[]"
  want="${pods:-[]}"
  LAST_OBSERVED="slices from ${src} in ${key}: ${got}; ready pods in ${src}: ${want}"
  [ "${got}" = "${want}" ] && [ "$(jq 'length' <<<"${want}")" = "${count}" ] || return 1
  # Labels and targetRef kind on every slice.
  if ! jq -e --arg region "$(region_of "${src}")" --arg svc "${svc}" --arg src "${src}" --arg mgr "${MANAGED_BY_SLICES}" '
      (.items | length) >= 1 and all(.items[];
        .metadata.labels["multicluster.kubernetes.io/service-name"] == $svc
        and .metadata.labels["multicluster.kubernetes.io/source-cluster"] == $src
        and .metadata.labels["topology.kubernetes.io/region"] == $region
        and .metadata.labels["endpointslice.kubernetes.io/managed-by"] == $mgr
        and .metadata.labels["dynamo.nvidia.com/mcs-managed-by"] == "mcs-controller"
        and all(.endpoints[]?; .targetRef.kind == "Pod"))' >/dev/null 2>&1 <<<"${slices}"; then
    LAST_OBSERVED="slices from ${src} in ${key} miss a KEP label or the region ($(region_of "${src}")): $(jq -c '[.items[] | .metadata.labels]' <<<"${slices}" | head -c 600)"
    return 1
  fi
}

# registry_import_ready NS NAME EXPECTED: "<cluster>:<ready endpoints> ..." of
# the participating clusters of the registry's import.
registry_import_ready() {
  registry_get "${CS_PATH}/imports/$1/$2?includeEndpoints=true"
  local got
  # $i and $c are jq variables.
  # shellcheck disable=SC2016
  got="$(rest_json '.serviceImport as $i | [$i.clusters[]? | select(.participating) | .cluster as $c
      | $c + ":" + ([$i.endpoints[]? | select(.cluster == $c) | .endpoints[]? | select(.conditions.ready)] | length | tostring)]
      | sort | join(" ")')" || got="<not JSON>"
  LAST_OBSERVED="registry import $1/$2: ${got:-none}$([ "${REST_STATUS}" = 0 ] || printf ' (%s)' "$(head -c 200 <<<"${REST_BODY}")")"
  [ "${REST_STATUS}" = 0 ] && [ "${got}" = "$3" ]
}

# registry_import_absent NS NAME: the registry answers NotFound.
registry_import_absent() {
  registry_get "${CS_PATH}/imports/$1/$2"
  LAST_OBSERVED="registry import $1/$2: $(head -c 200 <<<"${REST_BODY}")"
  [ "${REST_STATUS}" = 1 ] && grep -qi 'notfound\|not found' <<<"${REST_BODY}"
}

# import_gone KEY NS NAME DERIVED: no ServiceImport, derived Service or
# imported slice of NAME in KEY.
import_gone() {
  local slices
  object_absent "$1" "$2" "serviceimport/$3" || return 1
  if [ -n "$4" ]; then
    object_absent "$1" "$2" "service/$4" || return 1
  fi
  slices="$(kc "$1" -n "$2" get endpointslices -l "multicluster.kubernetes.io/service-name=$3" -o name 2>/dev/null)" || slices=""
  LAST_OBSERVED="imported slices of $3 left in $1: ${slices:-none}"
  [ -z "${slices}" ]
}

# curl_derived FROM_KEY NS IP PORT: a request from the client pod through the
# derived ClusterIP answers with the name of an echo pod.
curl_derived() {
  local got
  got="$(kc "$1" -n "$2" exec client -- wget -qO- -T 5 "http://$3:$4/" 2>&1)" || got="failed: ${got}"
  LAST_OBSERVED="wget http://$3:$4/ from client in $1: $(head -c 200 <<<"${got}")"
  grep -q '^echo-' <<<"${got}"
}

# registry_import_conflict NS NAME REASON WINNER: the registry's import has a
# conflict with this reason, won by the oldest export, and the ports of the
# winner. REASON "none" means no conflict.
registry_import_conflict() {
  registry_get "${CS_PATH}/imports/$1/$2"
  local got
  got="$(rest_json '[(.serviceImport.conflict.reason // "none"), (.serviceImport.conflict.winningCluster // "-"), ([.serviceImport.ports[]? | "\(.name)/\(.port)"] | join(","))] | join(" ")')" || got="<not JSON>"
  LAST_OBSERVED="registry import $1/$2: conflict, winner, ports: ${got:-none}"
  [ "${REST_STATUS}" = 0 ] && [ "${got}" = "$3 $4 $5" ]
}

# registry_cluster_participates NS NAME CLUSTER BOOL: whether the cluster's
# export takes part in the import.
registry_cluster_participates() {
  registry_get "${CS_PATH}/imports/$1/$2"
  local got
  got="$(printf '%s' "${REST_BODY}" | jq -r --arg c "$3" '[.serviceImport.clusters[]? | select(.cluster == $c) | (.participating == true)] | if length == 0 then "absent" else (.[0] | tostring) end' 2>/dev/null)" || got="<not JSON>"
  LAST_OBSERVED="registry import $1/$2: cluster $3 participating=${got}"
  [ "${REST_STATUS}" = 0 ] && [ "${got}" = "$4" ]
}

# no_slices_from KEY NS NAME SOURCE: no imported slice of NAME from SOURCE.
no_slices_from() {
  local got
  got="$(kc "$1" -n "$2" get endpointslices \
    -l "multicluster.kubernetes.io/service-name=$3,multicluster.kubernetes.io/source-cluster=$4" -o name 2>/dev/null)" || got=""
  LAST_OBSERVED="slices of $3 from $4 in $1: ${got:-none}"
  [ -z "${got}" ]
}

# export_older KEY1 KEY2 NS NAME: the ServiceExport in KEY1 was created before
# the one in KEY2, so the registry's oldest-export-wins rule, not its
# tie-break by cluster name, decides.
export_older() {
  local one two
  one="$(kc "$1" -n "$3" get serviceexport "$4" -o jsonpath='{.metadata.creationTimestamp}' 2>/dev/null)" || one=""
  two="$(kc "$2" -n "$3" get serviceexport "$4" -o jsonpath='{.metadata.creationTimestamp}' 2>/dev/null)" || two=""
  LAST_OBSERVED="ServiceExport $4 created at ${one:-none} in $1 and ${two:-none} in $2"
  [ -n "${one}" ] && [ -n "${two}" ] && [[ "${one}" < "${two}" ]]
}
