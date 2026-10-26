#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install the Dynamo Kubernetes platform 1.4.2 (operator) WITH etcd and NATS.
#
# Why etcd + NATS: platform chart 1.4.x installs without them by default and the
# operator injects DYN_DISCOVERY_BACKEND=kubernetes. With the
# vllm-runtime:1.3.0 image used in these experiments, the frontend never surfaces
# models published through Kubernetes discovery, so /v1/models stays empty and chat
# requests return 404. The fix is to install etcd + NATS here (note the keys are
# global.etcd.install / global.nats.install, not etcd.enabled) and set
# DYN_DISCOVERY_BACKEND=etcd on both components in every DGD (gen-dgds.sh does this).
#
# Use `helm repo add` + install by chart name: direct chart URLs under
# helm.ngc.nvidia.com return 404. The separate dynamo-crds chart is not needed; the
# operator applies its CRDs.
#
# Environment:
#   PLATFORM_NAMESPACE  namespace for the platform           (default: dynamo-system)
#   PLATFORM_VERSION    dynamo-platform chart version         (default: 1.4.2)
#   HELM                helm command, e.g. "microk8s helm3"   (default: helm)
#   KUBECTL             kubectl command, e.g. "microk8s kubectl" (default: kubectl)
set -euo pipefail

PLATFORM_NAMESPACE="${PLATFORM_NAMESPACE:-dynamo-system}"
PLATFORM_VERSION="${PLATFORM_VERSION:-1.4.2}"
read -r -a HELM_CMD <<< "${HELM:-helm}"
read -r -a KUBECTL_CMD <<< "${KUBECTL:-kubectl}"

"${HELM_CMD[@]}" repo add dynamo https://helm.ngc.nvidia.com/nvidia/ai-dynamo 2>/dev/null || true
"${HELM_CMD[@]}" repo update >/dev/null

"${HELM_CMD[@]}" upgrade --install dynamo-platform dynamo/dynamo-platform \
  --version "${PLATFORM_VERSION}" \
  --namespace "${PLATFORM_NAMESPACE}" --create-namespace \
  --set global.etcd.install=true \
  --set global.nats.install=true \
  --wait --timeout 10m

echo "--- waiting for dynamo-platform pods ---"
"${KUBECTL_CMD[@]}" -n "${PLATFORM_NAMESPACE}" wait --for=condition=Ready pods --all --timeout=300s
"${KUBECTL_CMD[@]}" -n "${PLATFORM_NAMESPACE}" get pods
