<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Hardware Fault Injection Service

A FastAPI service that orchestrates simulated hardware failures (GPU faults, network partitions, kernel-level errors) against Dynamo workloads running in a Kubernetes cluster. Used by Dynamo's fault-tolerance test suite.

## ⚠️ Test harness only — NOT for production

This directory deploys infrastructure that intentionally has elevated privileges so it can break things on purpose:

- A **`ClusterRole`** (`fault-injection-api`, in [`deploy/api-service.yaml`](deploy/api-service.yaml)) granting:
  - `get` / `list` / `watch` / `patch` on `nodes`, `pods`, `services` — to mark and disrupt running workloads
  - `get` / `list` / `watch` on `Deployments`, `DaemonSets`, `StatefulSets` (in `apps`)
  - `get` / `list` / `create` / `delete` on `NetworkPolicy` (in `networking.k8s.io`) — to install / remove partitions
  - **Full control (`get`/`list`/`create`/`delete`/`watch`)** on chaos-mesh CRDs: `NetworkChaos`, `PodChaos`, `StressChaos`, `IoChaos`
- A **chaos-mesh dependency** ([`deploy/chaos-mesh-gpu.yaml`](deploy/chaos-mesh-gpu.yaml)) installed cluster-wide.
- A **`DaemonSet`** ([`deploy/gpu-fault-injector-kernel.yaml`](deploy/gpu-fault-injector-kernel.yaml)) running a privileged pod on every node, with a kernel module that can synthesize GPU XID errors. The default `require_confirmation: false` means any caller with `pods/exec` on that pod can trigger node-level GPU faults without a confirmation prompt.

**An attacker with any reachable path to the API service can cause arbitrary pod kills, network partitions, IO faults, and GPU XID errors across the entire cluster.** That is the *point* of this harness — it must only be deployed against test clusters that you are willing to disrupt, and it must never be exposed to untrusted clients.

### Deployment expectations

- Deploy into a dedicated namespace (see [`deploy/namespace.yaml`](deploy/namespace.yaml)) on a test cluster only.
- The API service binds on a `ClusterIP`. Reach it via a port-forward or in-cluster client; never expose externally.
- Tear down with `kubectl delete -f deploy/` after a test run completes — leaving the ClusterRole / DaemonSet running is a long-tail risk.
- Production Dynamo deployments must not deploy any of these manifests.

## Components

| Path | Role |
|---|---|
| [`api_service/`](api_service/) | FastAPI server that orchestrates fault injection. Receives REST calls and dispatches to chaos-mesh / GPU kernel agents. |
| [`agents/`](agents/) | Worker-side agents that execute injected faults (process kills, NATS partitions, etc.). |
| [`cuda_fault_injection/`](cuda_fault_injection/) | CUDA-level fault injection helpers — see [`cuda_fault_injection/README.md`](cuda_fault_injection/README.md). |
| [`helpers/`](helpers/) | Shared utilities for the API service and agents. |
| [`deploy/`](deploy/) | Kubernetes manifests: `namespace.yaml`, `api-service.yaml` (Deployment + Service + ClusterRole), `chaos-mesh-gpu.yaml` (chaos-mesh install), `gpu-fault-injector-kernel.yaml` (privileged DaemonSet). |

## Quick start

Prerequisite: a test Kubernetes cluster with `kubectl` access, [chaos-mesh](https://chaos-mesh.org/) installed (or apply `deploy/chaos-mesh-gpu.yaml`).

```bash
# Deploy the harness (test cluster only — see warning above)
kubectl apply -f deploy/namespace.yaml
kubectl apply -f deploy/chaos-mesh-gpu.yaml         # only if chaos-mesh isn't already installed
kubectl apply -f deploy/gpu-fault-injector-kernel.yaml
kubectl apply -f deploy/api-service.yaml

# Port-forward and exercise the API
kubectl -n fault-injection-system port-forward svc/fault-injection-api 8080:8080
# In another shell:
curl -X POST http://localhost:8080/api/v1/faults/gpu/inject \
  -H "Content-Type: application/json" \
  -d '{"target_pod": "vllm-worker-0", "fault_type": "XID_ERROR", "severity": "HIGH"}'
```

For the full API reference (supported fault types, XID codes, network-partition shapes, recovery calls) see [`docs/design-docs/fault-tolerance-testing.md`](../../../../docs/design-docs/fault-tolerance-testing.md#hardware-fault-injection).

## Cleanup

```bash
# Tear down in reverse order
kubectl delete -f deploy/api-service.yaml
kubectl delete -f deploy/gpu-fault-injector-kernel.yaml
kubectl delete -f deploy/chaos-mesh-gpu.yaml        # only if you applied it from here
kubectl delete -f deploy/namespace.yaml
```

## Related docs

- [Fault Tolerance Testing](../../../../docs/design-docs/fault-tolerance-testing.md) — test infrastructure reference
- [Fault Tolerance README](../../../../docs/fault-tolerance/README.md) — Dynamo's runtime fault-tolerance features (distinct from this test harness)
- [`cuda_fault_injection/README.md`](cuda_fault_injection/README.md) — CUDA-level fault helpers
