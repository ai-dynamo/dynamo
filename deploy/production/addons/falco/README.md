<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Falco - Production Addon

Falco provides runtime threat detection from kernel and Kubernetes activity. In this profile it is a baseline detection layer for node and container behavior; it does not enforce policy or block pods.

## Deployment

| Field | Value |
|---|---|
| Argo CD app | [`gitops/apps/30-falco.yaml`](../../gitops/apps/30-falco.yaml) |
| Chart | `falco` |
| Source | `https://falcosecurity.github.io/charts` |
| Version | `8.0.2` |
| Namespace | `falco` |
| Values | [`values.yaml`](values.yaml) |
| Sync wave | `3` |
| Baseline | Yes |

## What This Addon Owns

| Capability | Local setting |
|---|---|
| Runtime sensor | `driver.kind: modern_ebpf`. |
| Large-node tuning | `modernEbpf.cpusForEachBuffer: 16` and inotify sysctl init container. |
| Alert formatting | `falco.json_output: true`, `priority: warning`. |
| Kubernetes context | `collectors.kubernetes.enabled: true`. |
| Event fan-out | `falcosidekick.enabled: true`, web UI disabled. |
| Resource cap | 1 CPU / 2 GiB limit, 100m / 512 MiB request. |

## What This Addon Does Not Own

- It does not enforce Kubernetes admission policy.
- It does not replace NetworkPolicy, Pod Security Admission, or image scanning.
- It does not store alerts long term; send sidekick output into the cluster's chosen alert/log pipeline when needed.

## Operating Notes

Falco can be noisy on GPU nodes because driver, container runtime, and model-loading activity are heavy. Tune rules in a separate values overlay rather than editing generated chart resources in-cluster.

## Verify

```bash
kubectl -n falco get pods
kubectl -n falco logs -l app.kubernetes.io/name=falco --tail=50
deploy/pre-deployment/pre-deployment-check.sh --require falco
```

## Upgrade Notes

Review major Falco chart updates with rule and driver changes in mind. Kernel/eBPF compatibility should be checked on the exact GPU node image before promoting.

Upstream reference: [Falco Kubernetes installation documentation](https://falco.org/docs/setup/kubernetes/).
