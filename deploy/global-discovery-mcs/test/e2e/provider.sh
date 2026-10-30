# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# shellcheck shell=bash
# shellcheck disable=SC2034  # settings are shared with the runner

CILIUM_VERSION="${CILIUM_VERSION:-1.20.4}"

install_provider() {
  if [ "${E2E_PROVIDER}" = dynamo ]; then
    install_releases
    return
  fi

  local key peer id=0 values
  for key in ${CLUSTER_KEYS}; do
    id=$((id + 1))
    values="${E2E_WORK_DIR}/cilium-${key}.yaml"
    cat >"${values}" <<YAML
cluster:
  name: ${key}
  id: ${id}
ipam:
  mode: kubernetes
kubeProxyReplacement: false
socketLB:
  enabled: false
operator:
  replicas: 1
clustermesh:
  useAPIServer: true
  enableEndpointSliceSynchronization: true
  mcsapi:
    enabled: true
  apiserver:
    service:
      type: NodePort
      nodePort: 32379
    tls:
      auto:
        method: helm
  config:
    enabled: true
    clusters:
YAML
    for peer in ${CLUSTER_KEYS}; do
      [ "${peer}" != "${key}" ] || continue
      cat >>"${values}" <<YAML
      ${peer}:
        ips: ["$(node_ip "${peer}")"]
        port: 32379
YAML
    done

    # Share the test CA; fresh Helm ownership metadata lets the chart adopt it.
    if [ "${key}" != hub ]; then
      kc hub -n kube-system get secret cilium-ca -o json |
        jq '{apiVersion, kind, type, data, metadata: {name: "cilium-ca", namespace: "kube-system",
          labels: {"app.kubernetes.io/managed-by": "Helm"},
          annotations: {"meta.helm.sh/release-name": "cilium", "meta.helm.sh/release-namespace": "kube-system"}}}' |
        kc "${key}" apply -f - >/dev/null
    fi
    log "installing Cilium ${CILIUM_VERSION} in ${key}, with MCS and EndpointSlice synchronization"
    hm "${key}" upgrade --install cilium cilium --repo https://helm.cilium.io \
      --version "${CILIUM_VERSION}" --namespace kube-system --values "${values}" \
      --wait --timeout "${E2E_HELM_TIMEOUT}" >"${E2E_WORK_DIR}/helm-cilium-${key}.log" 2>&1 ||
      { tail -n 30 "${E2E_WORK_DIR}/helm-cilium-${key}.log" >&2; die "Cilium installation failed in ${key}"; }
    rollout_ready "${key}" kube-system daemonset/cilium || return 1
    rollout_ready "${key}" kube-system deployment/cilium-operator || return 1
    rollout_ready "${key}" kube-system deployment/clustermesh-apiserver || return 1
  done
  pass "Cilium ${CILIUM_VERSION} installed in ${CLUSTER_KEYS}; bundled MCS image and chart unused"
  once "no bundled mcs-registry or mcs-controller in any test cluster" bundled_mcs_absent
}

bundled_mcs_absent() {
  local key workloads
  for key in ${CLUSTER_KEYS}; do
    workloads="$(kc "${key}" --request-timeout=10s get deployments,statefulsets,daemonsets -A -o json)" || return 1
    LAST_OBSERVED="bundled MCS workloads in ${key}"
    jq -e '[.items[] | select(
      .metadata.labels["app.kubernetes.io/name"] == "global-discovery-mcs"
      or .metadata.name == "mcs-registry" or .metadata.name == "mcs-controller"
      or any(.spec.template.spec.containers[]?; .image | contains("global-discovery-mcs")))] | length == 0' \
      >/dev/null <<<"${workloads}" || return 1
  done
}
