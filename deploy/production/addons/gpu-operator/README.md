<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA GPU Operator - Production Addon

NVIDIA GPU Operator owns the cluster-level NVIDIA stack: driver integration, container runtime setup, device plugin, GPU Feature Discovery, and DCGM exporter. Dynamo relies on these resources but does not install or manage them itself.

## Deployment

| Field | Value |
|---|---|
| Argo CD app | [`gitops/apps/00-gpu-operator.yaml`](../../gitops/apps/00-gpu-operator.yaml) |
| Chart | `gpu-operator` |
| Source | `https://helm.ngc.nvidia.com/nvidia` |
| Version | `v26.3.1` |
| Namespace | `gpu-operator` |
| Values | [`values.yaml`](values.yaml) |
| k3s overlay | [`values-k3s.yaml`](values-k3s.yaml) |
| Sync wave | `0` |
| Baseline | Yes |

## What This Addon Owns

| Capability | Local setting |
|---|---|
| Container runtime | `operator.defaultRuntime: containerd`. |
| GPU metrics | DCGM exporter `ServiceMonitor` enabled with `release: kube-prometheus-stack`. |
| k3s runtime wiring | `values-k3s.yaml` points toolkit at the k3s containerd template and socket. |

## What This Addon Does Not Own

- It does not create Dynamo workloads.
- It does not choose GPU scheduling policy; KAI/Grove and Kubernetes scheduling own placement.
- It does not own model storage or SGLang runtime images.

## k3s Host Driver Path

Use `values-k3s.yaml` through [`gitops/root-app-k3s.yaml`](../../gitops/root-app-k3s.yaml) when validating on k3s. If the host driver must be installed outside the operator, follow [`runbooks/gpu-operator-k3s-host-driver.md`](../../runbooks/gpu-operator-k3s-host-driver.md).

## Verify

```bash
kubectl -n gpu-operator get pods
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}'
kubectl get servicemonitor -A | grep -i dcgm
deploy/pre-deployment/pre-deployment-check.sh --require gpu-operator,dcgm-servicemonitor
```

## Upgrade Notes

GPU Operator upgrades can affect drivers, runtime classes, and device plugin behavior. Validate on the exact node OS and kernel before promoting to production.

Upstream reference: [NVIDIA GPU Operator documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/).
