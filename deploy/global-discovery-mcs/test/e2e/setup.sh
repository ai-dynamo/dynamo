# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# shellcheck shell=bash
# shellcheck disable=SC2034  # variables are shared with the other sourced files
#
# Setup and teardown of the end-to-end test: prerequisites, images, the three
# kind clusters, the static routes between their nodes and the Helm releases.
# Sourced by run.sh after lib.sh.

preflight() {
  local tool
  for tool in docker kubectl helm jq curl; do
    command -v "${tool}" >/dev/null 2>&1 || die "${tool} is required"
  done
  [ -x "${KIND}" ] || die "${KIND} is missing; run make kind (make e2e does)"
  docker info >/dev/null 2>&1 || die "Docker is not reachable (with colima: colima start)"
  mkdir -p "${E2E_WORK_DIR}" "${KUBECONFIG_DIR}"
  chmod 700 "${E2E_WORK_DIR}"
  log "work dir ${E2E_WORK_DIR}, phases ${E2E_PHASES}, clusters ${CLUSTER_KEYS} (prefix ${E2E_PREFIX}), kind $("${KIND}" version | cut -d' ' -f2)"
}

pod_subnet_of() {
  case "$1" in
    hub) printf '%s' "${POD_SUBNET_HUB}" ;;
    a) printf '%s' "${POD_SUBNET_A}" ;;
    b) printf '%s' "${POD_SUBNET_B}" ;;
  esac
}

service_subnet_of() {
  case "$1" in
    hub) printf '%s' "${SERVICE_SUBNET_HUB}" ;;
    a) printf '%s' "${SERVICE_SUBNET_A}" ;;
    b) printf '%s' "${SERVICE_SUBNET_B}" ;;
  esac
}

region_of() {
  printf 'region-%s' "$1"
}

phase_selected() {  # phase_selected N
  case " ${E2E_PHASES//,/ } " in
    *" $1 "*) return 0 ;;
  esac
  return 1
}

# ---------------------------------------------------------------------------
# Images

# image_tag_of IMAGE: e2e-<12 hex of the image id>, so a rebuilt image always
# gets a new tag and a reused cluster runs it.
image_tag_of() {
  local id
  id="$(docker image inspect --format '{{.Id}}' "$1" 2>/dev/null)" || return 1
  id="${id#sha256:}"
  printf 'e2e-%s' "$(printf '%s' "${id}" | cut -c1-12)"
}

# build_image REPO CONTEXT [DOCKERFILE]: builds REPO:e2e-latest for
# E2E_PLATFORM (unless E2E_SKIP_BUILD=1), tags it by id and prints the tag.
build_image() {
  local repo="$1" context="$2" dockerfile="${3:-}" tag
  if [ "${E2E_SKIP_BUILD}" != "1" ]; then
    log "docker build ${repo} (${E2E_PLATFORM}) from ${context}" >&2
    if [ -n "${dockerfile}" ]; then
      docker build --platform "${E2E_PLATFORM}" -f "${dockerfile}" -t "${repo}:e2e-latest" "${context}" \
        >"${E2E_WORK_DIR}/build-${repo}.log" 2>&1 ||
        { tail -n 30 "${E2E_WORK_DIR}/build-${repo}.log" >&2; die "docker build of ${repo} failed; log ${E2E_WORK_DIR}/build-${repo}.log"; }
    else
      docker build --platform "${E2E_PLATFORM}" -t "${repo}:e2e-latest" "${context}" \
        >"${E2E_WORK_DIR}/build-${repo}.log" 2>&1 ||
        { tail -n 30 "${E2E_WORK_DIR}/build-${repo}.log" >&2; die "docker build of ${repo} failed; log ${E2E_WORK_DIR}/build-${repo}.log"; }
    fi
  fi
  docker image inspect "${repo}:e2e-latest" >/dev/null 2>&1 ||
    die "image ${repo}:e2e-latest is not in Docker; run without E2E_SKIP_BUILD=1"
  tag="$(image_tag_of "${repo}:e2e-latest")"
  docker tag "${repo}:e2e-latest" "${repo}:${tag}"
  log "image ${repo}:${tag}" >&2
  printf '%s' "${tag}"
}

build_images() {
  MCS_IMAGE_TAG="$(build_image "${MCS_IMAGE_REPO}" "${MODULE_DIR}")"
  if phase_selected 3; then
    GDS_IMAGE_TAG="$(build_image "${GDS_IMAGE_REPO}" "${REPO_ROOT}" "${REPO_ROOT}/container/Dockerfile.global-discovery-service")"
  fi
  if ! docker image inspect "${BUSYBOX_IMAGE}" >/dev/null 2>&1; then
    log "docker pull ${BUSYBOX_IMAGE}"
    docker pull --platform "${E2E_PLATFORM}" -q "${BUSYBOX_IMAGE}" >/dev/null || die "docker pull ${BUSYBOX_IMAGE} failed"
  fi
}

# image_in_node KEY IMAGE: the node's containerd has IMAGE (repo:tag).
image_in_node() {
  docker exec "$(node_container "$1")" crictl images 2>/dev/null |
    awk -v r="$2" '{ if (($1 ":" $2) == r || ("docker.io/library/" $1 ":" $2) == r || ("docker.io/library/" r) == ($1 ":" $2)) found = 1 } END { exit !found }'
}

# load_image KEY IMAGE: loads a Docker image into a kind node. The Docker
# image store may keep a multi-platform index that kind load cannot read, so
# a failure falls back to an archive of E2E_PLATFORM only.
load_image() {
  local key="$1" image="$2" archive="${E2E_WORK_DIR}/image.tar"
  if image_in_node "${key}" "${image}"; then
    return 0
  fi
  if "${KIND}" load docker-image --name "$(cluster_name "${key}")" "${image}" >"${E2E_WORK_DIR}/kind-load.log" 2>&1 &&
    image_in_node "${key}" "${image}"; then
    return 0
  fi
  log "kind load docker-image ${image} into ${key} failed ($(tail -n 1 "${E2E_WORK_DIR}/kind-load.log")); loading an archive"
  docker save --platform "${E2E_PLATFORM}" -o "${archive}" "${image}" >"${E2E_WORK_DIR}/kind-load.log" 2>&1 ||
    docker save -o "${archive}" "${image}" >>"${E2E_WORK_DIR}/kind-load.log" 2>&1 ||
    die "docker save ${image} failed: $(tail -n 3 "${E2E_WORK_DIR}/kind-load.log")"
  "${KIND}" load image-archive --name "$(cluster_name "${key}")" "${archive}" >>"${E2E_WORK_DIR}/kind-load.log" 2>&1 || true
  rm -f "${archive}"
  image_in_node "${key}" "${image}" ||
    die "image ${image} is not in the node of ${key} after loading: $(tail -n 3 "${E2E_WORK_DIR}/kind-load.log")"
}

# ---------------------------------------------------------------------------
# Clusters

create_cluster() {  # create_cluster KEY
  local key="$1" name config
  name="$(cluster_name "${key}")"
  if cluster_exists "${key}"; then
    log "reusing kind cluster ${name}"
  else
    config="${E2E_WORK_DIR}/kind-${key}.yaml"
    cat >"${config}" <<YAML
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
networking:
  podSubnet: "$(pod_subnet_of "${key}")"
  serviceSubnet: "$(service_subnet_of "${key}")"
nodes:
  - role: control-plane
YAML
    log "creating kind cluster ${name} (pods $(pod_subnet_of "${key}"), services $(service_subnet_of "${key}"))"
    "${KIND}" create cluster --name "${name}" --image "${KIND_NODE_IMAGE}" --config "${config}" \
      --kubeconfig "$(kubeconfig_of "${key}")" --wait 180s >"${E2E_WORK_DIR}/kind-create-${key}.log" 2>&1 ||
      { tail -n 20 "${E2E_WORK_DIR}/kind-create-${key}.log" >&2; die "kind create cluster ${name} failed; log ${E2E_WORK_DIR}/kind-create-${key}.log"; }
  fi
  "${KIND}" get kubeconfig --name "${name}" >"$(kubeconfig_of "${key}")" || die "cannot read the kubeconfig of ${name}"
  chmod 600 "$(kubeconfig_of "${key}")"
}

# route_pod_subnets: on every node, a static route to the pod subnet of each
# other cluster through that cluster's node, as VPC peering does on EKS.
# Idempotent.
route_pod_subnets() {
  local from to via
  for from in ${CLUSTER_KEYS}; do
    for to in ${CLUSTER_KEYS}; do
      [ "${from}" = "${to}" ] && continue
      via="$(node_ip "${to}")"
      [ -n "${via}" ] || die "node of ${to} has no address on Docker network ${KIND_NETWORK}"
      docker exec "$(node_container "${from}")" ip route replace "$(pod_subnet_of "${to}")" via "${via}" ||
        die "cannot add the route to $(pod_subnet_of "${to}") on ${from}"
    done
  done
}

# check_inotify: every kind node is a systemd container, and each of them
# holds dozens of inotify instances of the Docker host's per-user pool
# (fs.inotify.max_user_instances, 128 by default in colima). Three clusters
# need about 150, and the node that starts last fails with "too many open
# files".
check_inotify() {
  local limit
  limit="$(docker exec "$(node_container hub)" cat /proc/sys/fs/inotify/max_user_instances 2>/dev/null)" || return 0
  [ -n "${limit}" ] || return 0
  LAST_OBSERVED="fs.inotify.max_user_instances is ${limit}"
  if [ "${limit}" -lt 256 ] && [ "$(wc -w <<<"${CLUSTER_KEYS}")" -ge 3 ] && [ "${E2E_SKIP_INOTIFY_CHECK:-0}" != "1" ]; then
    die "fs.inotify.max_user_instances is ${limit} on the Docker host; three kind clusters need at least 256. With colima run: colima ssh -- sudo sysctl -w fs.inotify.max_user_instances=512 (or set E2E_SKIP_INOTIFY_CHECK=1)"
  fi
}

create_clusters() {
  local key
  for key in ${CLUSTER_KEYS}; do
    create_cluster "${key}"
    [ "${key}" = hub ] && check_inotify
  done
  route_pod_subnets
  for key in ${CLUSTER_KEYS}; do
    load_image "${key}" "${MCS_IMAGE_REPO}:${MCS_IMAGE_TAG}"
    load_image "${key}" "${BUSYBOX_IMAGE}"
  done
  if phase_selected 3; then
    load_image hub "${GDS_IMAGE_REPO}:${GDS_IMAGE_TAG}"
  fi
}

delete_clusters() {
  local key
  for key in ${CLUSTER_KEYS}; do
    if cluster_exists "${key}"; then
      log "deleting kind cluster $(cluster_name "${key}")"
      "${KIND}" delete cluster --name "$(cluster_name "${key}")" --kubeconfig "${E2E_WORK_DIR}/kind-delete.kubeconfig" >/dev/null 2>&1 ||
        log "kind delete cluster $(cluster_name "${key}") failed"
    fi
    rm -f "$(kubeconfig_of "${key}")"
  done
}

# ---------------------------------------------------------------------------
# Helm releases

# install_chart KEY ARGS...: installs or upgrades the release in the cluster
# with the image of this run and waits for it.
install_chart() {
  local key="$1"
  shift
  hm "${key}" upgrade --install "${MCS_RELEASE}" "${CHART_DIR}" \
    --namespace "${MCS_NAMESPACE}" --create-namespace \
    --set-string "image.repository=${MCS_IMAGE_REPO}" \
    --set-string "image.tag=${MCS_IMAGE_TAG}" \
    --set-string image.pullPolicy=Never \
    --set controller.replicas=1 \
    --set controller.clusterIdentitySource=flags \
    --set-string "controller.clustersetId=${E2E_CLUSTERSET}" \
    --set-string "controller.clusterId=$key" \
    --set-string "controller.region=$(region_of "${key}")" \
    --set-string controller.leaseTtl=10s \
    --set controller.registryAllowInsecureHttp=true \
    --set-string controller.log.level=debug \
    "$@" \
    --wait --timeout "${E2E_HELM_TIMEOUT}" >"${E2E_WORK_DIR}/helm-${key}.log" 2>&1 ||
    { tail -n 20 "${E2E_WORK_DIR}/helm-${key}.log" >&2; die "helm upgrade --install in ${key} failed; log ${E2E_WORK_DIR}/helm-${key}.log"; }
}

# registry_url: the gRPC NodePort of the hub registry on the hub node's
# address on KIND_NETWORK, which a and b dial.
registry_url() {
  local ip port
  ip="$(node_ip hub)"
  port="$(kc hub -n "${MCS_NAMESPACE}" get svc mcs-registry -o jsonpath='{.spec.ports[?(@.name=="grpc")].nodePort}' 2>/dev/null)" || port=""
  LAST_OBSERVED="hub node address ${ip:-none}, gRPC NodePort ${port:-none}"
  [ -n "${ip}" ] && [ -n "${port}" ] || return 1
  REGISTRY_URL="http://${ip}:${port}"
}

install_releases() {
  install_chart hub \
    --set registry.enabled=true \
    --set registry.store=memory \
    --set registry.replicas=1 \
    --set registry.service.type=NodePort \
    --set controller.enabled=true
  once "registry NodePort of the hub is reachable on the hub node address" registry_url || return 1
  log "registry URL of the member clusters: ${REGISTRY_URL}"
  local key
  for key in ${CLUSTER_KEYS}; do
    [ "${key}" = hub ] && continue
    install_chart "${key}" \
      --set controller.enabled=true \
      --set-string "controller.registryUrls[0]=${REGISTRY_URL}"
  done
  pass "helm releases installed: registry and controller in the hub, controllers in the member clusters (${CLUSTER_KEYS}; 1 replica each)"
}
