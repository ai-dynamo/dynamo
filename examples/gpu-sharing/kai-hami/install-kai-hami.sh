#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install upstream KAI-Scheduler v0.17.0 (GPU sharing + hamicore binder plugin) and
# the HAMi resource isolator (kai-resource-isolator 1.1.0-chart), then open the KAI
# default queues so 16 fractional workers can be admitted on 8 GPUs.
#
# Prerequisites (not installed here):
#   - A Kubernetes cluster with the NVIDIA GPU Operator, so the `nvidia` RuntimeClass
#     exists and the node advertises nvidia.com/gpu.
#   - An NVIDIA driver supported by the GPU Operator. The published KAI + HAMi run
#     used driver 595.91.07 with GPU Operator v25.10.1.
#   - KAI-Scheduler >= v0.17.0 is required by the HAMi isolator.
#
# Install the Dynamo platform separately with ../common/install-dynamo-platform.sh.
#
# Environment:
#   HELM     helm command, e.g. "microk8s helm3"       (default: helm)
#   KUBECTL  kubectl command, e.g. "microk8s kubectl"  (default: kubectl)
set -euo pipefail

read -r -a HELM_CMD <<< "${HELM:-helm}"
read -r -a KUBECTL_CMD <<< "${KUBECTL:-kubectl}"

echo "=== KAI-Scheduler v0.17.0 + hamicore ==="
"${HELM_CMD[@]}" upgrade --install kai-scheduler oci://ghcr.io/kai-scheduler/kai-scheduler/kai-scheduler \
  -n kai-scheduler --create-namespace \
  --version v0.17.0 \
  --set "global.gpuSharing=true" \
  --set "binder.plugins.hamicore.enabled=true"

echo "--- waiting for kai-scheduler pods ---"
"${KUBECTL_CMD[@]}" -n kai-scheduler wait --for=condition=Ready pods --all --timeout=180s

echo "=== HAMi resource isolator 1.1.0-chart ==="
# The chart lives on Docker Hub under projecthami; chart versions carry a "-chart"
# suffix. It adds LD_PRELOAD=/usr/local/vgpu/libvgpu.so to fractional pods, which
# intercepts CUDA allocations and enforces the memory cap (memory only, not compute).
"${HELM_CMD[@]}" upgrade --install kai-resource-isolator oci://docker.io/projecthami/kai-resource-isolator \
  --namespace kai-resource-isolator --create-namespace \
  --set monitor.enabled=true \
  --set monitor.runtimeClassName=nvidia \
  --version 1.1.0-chart

echo "--- waiting for isolator pods ---"
"${KUBECTL_CMD[@]}" -n kai-resource-isolator wait --for=condition=Ready pods --all --timeout=180s

echo "=== open default queue quotas ==="
# KAI creates the default queues asynchronously after install.
for q in default-parent-queue default-queue; do
  for _ in $(seq 1 30); do
    "${KUBECTL_CMD[@]}" get queue "$q" >/dev/null 2>&1 && break
    sleep 2
  done
  "${KUBECTL_CMD[@]}" patch queue "$q" --type=merge -p '{
    "spec": {
      "resources": {
        "gpu":    {"quota": -1, "limit": -1, "overQuotaWeight": 1},
        "cpu":    {"quota": -1, "limit": -1, "overQuotaWeight": 1},
        "memory": {"quota": -1, "limit": -1, "overQuotaWeight": 1}
      }
    }
  }'
done
"${KUBECTL_CMD[@]}" get queue default-queue -o jsonpath='{.spec.resources}'
echo
echo "KAI + HAMi stack ready"
