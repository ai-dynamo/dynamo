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

By default, each Dynamo component pod is given a ServiceAccount that allows it
to watch `EndpointSlice` and `DynamoWorkerMetadata` resources within its
namespace.

When Kubernetes discovery is enabled, the operator creates these RBAC resources
in the `DynamoGraphDeployment` namespace:

- ServiceAccount: `<DGD NAME>-k8s-service-discovery`
- Role: `<DGD NAME>-k8s-service-discovery-role`
- RoleBinding: `<DGD NAME>-k8s-service-discovery-binding`

The Role is scoped to the `DynamoGraphDeployment` and is deleted with it.

##### Use a Pre-Created ServiceAccount

If a normal frontend or worker component needs to run as an existing Kubernetes
ServiceAccount, such as a cloud workload identity ServiceAccount, set
`spec.components[*].podTemplate.spec.serviceAccountName`. The controller uses
that custom ServiceAccount for normal frontend and worker components, including
prefill and decode workers.

Those component pods need the permissions defined in the generated
`<DGD NAME>-k8s-service-discovery-role` for Kubernetes-native service
discovery. Bind your custom ServiceAccount to that Role in the same namespace
as the `DynamoGraphDeployment`.

```yaml
# Relevant fields only.
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: my-dgd
  namespace: dynamo-system
spec:
  components:
    - name: decode
      type: decode
      podTemplate:
        spec:
          serviceAccountName: workload-identity-sa
          containers:
            - name: main
              image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:latest
```

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: workload-identity-discovery
  namespace: dynamo-system
subjects:
  - kind: ServiceAccount
    name: workload-identity-sa
    namespace: dynamo-system
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: my-dgd-k8s-service-discovery-role
```

Keep these caveats in mind:

- Kubernetes discovery must be enabled. It is the default Kubernetes backend, but
  the discovery Role is not reconciled when the deployment selects the `etcd`
  backend.
- You can apply the custom RoleBinding before the generated Role exists. It
  grants permissions only after the operator reconciles the
  `DynamoGraphDeployment` and creates the Role.
- The operator still creates the default discovery ServiceAccount and
  RoleBinding. This is harmless when the component pods use your custom
  ServiceAccount.
- This custom ServiceAccount path applies to normal frontend and worker
  components. Planner and Endpoint Picker Plugin (EPP) components have separate
  service-account and RBAC paths.

#### Environment Variables

The following environment variables are automatically injected into pods by the operator to facilitate service discovery:

| Variable | Description |
|----------|-------------|
| `DYN_DISCOVERY_BACKEND` | Set to `kubernetes` |
| `POD_NAME` | Pod name (via downward API) |
| `POD_NAMESPACE` | Pod namespace (via downward API) |
| `POD_UID` | Pod UID (via downward API) |

The pod's instance ID is deterministically generated by hashing the pod name, ensuring consistent identity and correlation between EndpointSlices and CRs.

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
