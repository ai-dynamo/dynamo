---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Service Discovery
---

Dynamo components (frontends, workers, planner) need to be able to discover each other and their capabilities at runtime. We refer to this as service discovery. There are 2 kinds of service discovery backends supported on Kubernetes.

## Discovery Backends

| Backend | Default | Dependencies | Use Case |
|---------|---------|--------------|----------|
| **Kubernetes** | ✅ Yes | None (native K8s) | Recommended for all Kubernetes deployments |
| **KV Store (etcd)** | No | etcd cluster | Legacy deployments |

## Kubernetes Discovery (Default)

Kubernetes discovery is the default and recommended backend when running on Kubernetes. It uses native Kubernetes primitives to facilitate discovery of components:

- **DynamoWorkerMetadata CRD**: Each worker stores its registered endpoints and model cards in a Custom Resource
- **EndpointSlices**: EndpointSlices signal each component's readiness status

### Discovery Granularity: Pod vs. Container

Kubernetes discovery supports two granularities, selected per DynamoGraphDeployment with the `nvidia.com/dynamo-kube-discovery-mode` annotation:

| Mode | When to use | Granularity |
| --- | --- | --- |
| `pod` (default) | Standard single-container worker pods | One `DynamoWorkerMetadata` CR per pod |
| `container` | Pods with more than one identity-bearing container (e.g. [Engine Failover](failover.md) with `engine-0` and `engine-1`) | One CR per ready container |

In **pod mode**, the discovery daemon watches `EndpointSlices` and emits one CR per pod, named after the pod. This is the behavior assumed in every section below and is correct for any worker whose pod runs a single logical engine.

In **container mode**, the daemon instead watches `Pod` objects directly and emits one CR per ready container. CR names are derived as `{pod_name}-{container_name}`, which lets a single pod expose multiple independently discoverable endpoints.

#### The `main` container special case

To stay backward-compatible with pod-mode consumers, a container named `main` always takes the pod-level identity: its CR is named `{pod_name}` (no suffix) and its instance ID is the hash of the pod name. This means:

- A pod-mode frontend can consume workers from a container-mode pod as long as the worker's container is named `main`.
- Mixing `main` with other identity-bearing containers in the same pod works, but the non-`main` containers must use their own `engine-*` / custom names to avoid colliding with the pod-level CR.

#### Enabling container mode

Set the annotation on the DGD. The setting applies to every service in the deployment:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-deployment
  annotations:
    nvidia.com/dynamo-kube-discovery-mode: container
spec:
  services:
    # ...
```

Container mode is **required** when any service in the DGD has `failover.enabled: true`; the validating webhook rejects the DGD at admission time if the annotation is missing.

### Implementation Details

Each pod runs a **discovery daemon** that watches both EndpointSlices and DynamoWorkerMetadata CRs. A pod is only discoverable when it appears as "ready" in an EndpointSlice AND has a corresponding `DynamoWorkerMetadata` CR. This correlation ensures pods aren't discoverable until they're ready, metadata is immediately available, and stale entries are cleaned up when pods terminate.

#### DynamoWorkerMetadata CRD

Each worker pod creates a `DynamoWorkerMetadata` CR that stores its discovery metadata:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoWorkerMetadata
metadata:
  name: my-worker-pod-abc123
  namespace: dynamo-system
  ownerReferences:
    - apiVersion: v1
      kind: Pod
      name: my-worker-pod-abc123
      uid: <pod-uid>
      controller: true
spec:
  data:
    endpoints:
      "dynamo/backend/generate":
        type: Endpoint
        namespace: dynamo
        component: backend
        endpoint: generate
        instance_id: 12345678901234567890
        transport:
          nats_tcp: "dynamo_backend.generate-abc123"
    model_cards: {}
```

The CR is named after the pod and includes an owner reference for automatic garbage collection when the pod is deleted.

#### EndpointSlices

While DynamoWorkerMetadata resources provide an up-to-date snapshot of a component's capabilities, EndpointSlices give a snapshot of health of the various Dynamo components.

The operator creates a Kubernetes Service targeting the Dynamo components. The Kubernetes controller in turn creates and maintains EndpointSlice resources that keep track of the readiness of the pods targeted by the Service. Watching these slices gives us an up-to-date snapshot of which Dynamo components are ready to serve traffic.

##### Readiness Probes
A pod is marked ready if the readiness probe succeeds. On Dynamo workers, this is when the `generate` endpoint is available and healthy. These probes are configured by the Dynamo operator for each pod/component.

#### RBAC

Each Dynamo component pod is automatically given a ServiceAccount that allows it to watch `EndpointSlice` and `DynamoWorkerMetadata` resources within its namespace.

#### Environment Variables

The following environment variables are automatically injected into pods by the operator to facilitate service discovery:

| Variable | Description |
|----------|-------------|
| `DYN_DISCOVERY_BACKEND` | Set to `kubernetes` |
| `DYN_KUBE_DISCOVERY_MODE` | `pod` or `container`. Injected only when the DGD opts into container mode via annotation. |
| `CONTAINER_NAME` | Name of the container hosting this process. Injected only in container mode. |
| `POD_NAME` | Pod name (via downward API) |
| `POD_NAMESPACE` | Pod namespace (via downward API) |
| `POD_UID` | Pod UID (via downward API) |

The pod's instance ID is deterministically generated by hashing the pod name in pod mode, or hashing `{pod_name}-{container_name}` in container mode (with the `main` container falling back to the pod-level hash). This ensures consistent identity and correlation between EndpointSlices and CRs.

## KV Store Discovery (etcd)

To use etcd-based discovery instead of Kubernetes-native discovery, add the annotation to your DynamoGraphDeployment:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-deployment
  annotations:
    nvidia.com/dynamo-discovery-backend: etcd
spec:
  services:
    # ...
```

This requires an etcd cluster to be available. The etcd connection is configured via the platform Helm chart.
