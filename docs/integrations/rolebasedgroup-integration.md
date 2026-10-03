<!--
SPDX-FileCopyrightText: Copyright (c) 2026-2027 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# RoleBasedGroup (RBG) Integration

This document provides integration examples for deploying NVIDIA Dynamo with [RoleBasedGroup (RBG)](https://github.com/sgl-project/rbg), demonstrating both aggregated and disaggregated inference architectures for deploying large language models in Kubernetes.

## Table of Contents

- [What is RoleBasedGroup (RBG)?](#what-is-rolebasedgroup-rbg)
- [Deploying Dynamo with RBG](#deploying-dynamo-with-rbg)
    - [Installation](#installation)
- [Example Files](#example-files)
- [Quick Start Example](#quick-start-example)
    - [Prerequisites](#prerequisites)
    - [Deploy Disaggregated Architecture](#deploy-disaggregated-architecture)
    - [Verify Deployment](#verify-deployment)
    - [Test the Deployment](#test-the-deployment)
    - [Understanding the Deployment](#understanding-the-deployment)
- [Advanced Features](#advanced-features)
    - [Rollout Strategy with Partition](#rollout-strategy-with-partition)
    - [In-Place Updates](#in-place-updates)
    - [Coordinated Updates](#coordinated-updates)
    - [Coordinated Scaling](#coordinated-scaling)
    - [Pod Group (Gang Scheduling)](#pod-group-gang-scheduling)
    - [Scaling Adapter (HPA Integration)](#scaling-adapter-hpa-integration)
- [Integrated with Dynamo Ecosystem](#integrated-with-dynamo-ecosystem)
    - [Backend Engines](#backend-engines)
    - [AIConfigurator](#aiconfigurator)
    - [Dynamo SLA Planner](#dynamo-sla-planner)
- [Additional Resources](#additional-resources)

## What is RoleBasedGroup (RBG)?

**RoleBasedGroup (RBG)** is a Kubernetes Custom Resource Definition (CRD) designed for orchestrating distributed, stateful AI inference workloads with multi-role collaboration and built-in service discovery. It addresses the limitations of traditional Kubernetes primitives (StatefulSets, Deployments) when managing complex distributed inference services.

RBG models an inference service as a **role-based group** rather than a collection of independent workloads, treating it as a topological, stateful, and coordinated multi-role system managed as a single unit.

### Why RBG for Dynamo?

Traditional Kubernetes primitives (e.g. plain StatefulSets / LeaderWorkSet) are ill-suited for Dynamo inference services that:

- run as multi-role topologies (gateway / router / prefill / decode),
- are performance-sensitive to GPU / network topology,
- require atomic, cross-role operations (deploy, upgrade, scale, failover).
  RBG treats an inference service as a role-based group, not a loose set of workloads. It models the service as a topologized, stateful, coordinated multi-role organism and manages it as a single unit.

For more information about RBG, see the [RBG documentation](https://github.com/sgl-project/rbg/blob/main/doc/quick_start.md).

## Deploying Dynamo with RBG

This section describes how to deploy NVIDIA Dynamo inference services using RoleBasedGroup (RBG).

### Installation

### Prerequisites

- Kubernetes cluster version >= 1.28
- kubectl command-line tool configured to communicate with your cluster
- At least 1 node with 1+ CPUs and 1GB memory for the RBG controller

### Install RBG Controller

```bash
kubectl apply --server-side -f https://github.com/sgl-project/rbg/blob/main/deploy/kubectl/manifests.yaml
```

Wait for the controller to be ready:

```bash
kubectl wait deploy/rbgs-controller-manager -n rbgs-system --for=condition=available --timeout=5m
```

For more information about installing RBG, see the [Install RBG](https://github.com/sgl-project/rbg/blob/main/doc/install.md).

## Example Files

The following example files demonstrate Dynamo integration with RBG:

| File | Description | Pattern |
|------|-------------|---------|
| [agg.yaml](https://github.com/sgl-project/rbg/tree/main/examples/inference/ecosystem/dynamo/agg.yaml) | Aggregated inference with standalone pattern (Dynamo) | standalonePattern (tp-size=1) |
| [agg-multi-nodes.yaml](https://github.com/sgl-project/rbg/tree/main/examples/inference/ecosystem/dynamo/agg-multi-nodes.yaml) | Aggregated inference with LeaderWorker pattern (Dynamo) | leaderWorkerPattern (tp-size=2) |
| [pd-disagg.yaml](https://github.com/sgl-project/rbg/tree/main/examples/inference/ecosystem/dynamo/pd-disagg.yaml) | PD-disaggregated inference with standalone pattern (Dynamo) | standalonePattern (tp-size=1) |
| [pd-disagg-multi-nodes.yaml](https://github.com/sgl-project/rbg/tree/main/examples/inference/ecosystem/dynamo/pd-disagg-multi-nodes.yaml) | PD-disaggregated inference with LeaderWorker pattern (Dynamo) | leaderWorkerPattern (tp-size=2) |

**File Naming Convention**:
- `agg*`: Aggregated prefill/decode architecture
- `disagg*`: Disaggregated prefill/decode architecture
- `*-multinode`: Multi-node deployment

All examples use:
- **API Version**: `workloads.x-k8s.io/v1alpha2`
- **Runtime**: NVIDIA Dynamo SGLang (`nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.0.1`)
- **Model**: Qwen/Qwen3-0.6B
- **Features**: RoleTemplates, rolloutStrategy with InPlaceIfPossible, scalingAdapter enabled, multi-node tensor parallel support

## Quick Start Example

This section demonstrates how to deploy a disaggregated Dynamo inference service with multi-node workers using RBG.

### Prerequisites

Before deploying, ensure you have:

1. **RBG Controller** installed (see [Installation](#installation) section above)

2. **Dynamo Dependencies** in your Kubernetes cluster:
    - [ETCD service](https://github.com/sgl-project/rbg/tree/main/examples/inference/ecosystem/dynamo/etcd.yaml) for distributed key-value storage (typically `http://etcd:2379`)
    - [NATS service](https://github.com/sgl-project/rbg/tree/main/examples/inference/ecosystem/dynamo/nats.yaml) with JetStream enabled for message streaming (typically `nats://nats:4222`)

3. **Model Storage** configured:
    - PersistentVolume/PersistentVolumeClaim for model files, or
    - HuggingFace token secret for automatic model downloads

4. **GPU Resources**:
    - NVIDIA GPU drivers installed on cluster nodes
    - NVIDIA device plugin for Kubernetes
    - (Optional) RDMA network interface for high-performance disaggregated serving

### Deploy Disaggregated Architecture

Deploy the PD-disaggregated architecture with multi-node workers:

```bash
kubectl apply -f https://github.com/sgl-project/rbg/tree/main/examples/inference/ecosystem/dynamo/pd-disagg-multi-nodes.yaml
```

### Verify Deployment

Check the RoleBasedGroup status:

```bash
kubectl get rolebasedgroup dynamo-pd-disagg-multi-node-inference
```

**Expected output**:
```
NAME                                   READY   AGE
dynamo-pd-disagg-multi-node-inference  True    2m
```

Check all pods are running:

```bash
kubectl get pods -l rolebasedgroup.workloads.x-k8s.io/name=dynamo-pd-disagg-multi-node-inference
```

**Expected output**:
```
NAME                                                 READY   STATUS    RESTARTS   AGE
dynamo-pd-disagg-multi-node-inference-processor-0    1/1     Running   0          2m
dynamo-pd-disagg-multi-node-inference-prefill-0      1/1     Running   0          2m
dynamo-pd-disagg-multi-node-inference-prefill-1      1/1     Running   0          2m
dynamo-pd-disagg-multi-node-inference-decode-0       1/1     Running   0          2m
dynamo-pd-disagg-multi-node-inference-decode-1       1/1     Running   0          2m
```

View detailed deployment status:

```bash
kubectl describe rolebasedgroup dynamo-pd-disagg-multi-node-inference
```

Check role-specific status:

```bash
kubectl get rolebasedgroup dynamo-pd-disagg-multi-node-inference -o jsonpath='{.status.roleStatuses}' | jq
```

**Expected output**:
```json
[
  {
    "name": "processor",
    "replicas": 1,
    "readyReplicas": 1,
    "updatedReplicas": 1
  },
  {
    "name": "prefill",
    "replicas": 1,
    "readyReplicas": 1,
    "updatedReplicas": 1
  },
  {
    "name": "decode",
    "replicas": 1,
    "readyReplicas": 1,
    "updatedReplicas": 1
  }
]
```

### Test the Deployment

Forward the processor service port to localhost:

```bash
kubectl port-forward svc/dynamo-pd-disagg-multi-node-inference-service 8000:8000
```

In another terminal, send a test request:

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ],
    "stream": false,
    "max_tokens": 50
  }'
```

### Understanding the Deployment

The [pd-disagg-multi-nodes.yaml](https://github.com/sgl-project/rbg/tree/main/examples/inference/ecosystem/dynamo/pd-disagg-multi-nodes.yaml) example creates a three-role RoleBasedGroup:

1. **Processor Role** (`processor`):
    - Dynamo frontend for request routing and HTTP API
    - OpenAI-compatible HTTP server listening on port 8000
    - Routes and coordinates requests between prefill and decode workers
    - 1 replica, CPU-only (no GPU required)
    - Exposed via Service `dynamo-pd-disagg-multi-node-inference-service`

2. **Prefill Role** (`prefill`):
    - Specialized worker for prompt processing (prefill phase)
    - Uses `leaderWorkerPattern` with size=2 (1 leader + 1 worker)
    - Tensor parallelism size = 2 (`--tp-size 2`)
    - Disaggregation mode: `--disaggregation-mode prefill`
    - Transfers KV cache to decode workers via NIXL high-speed backend (`--disaggregation-transfer-backend nixl`)
    - Multi-node support via `--dist-init-addr`, `--nnodes`, `--node-rank`

3. **Decode Role** (`decode`):
    - Specialized worker for token generation (decode phase)
    - Uses `leaderWorkerPattern` with size=2 (1 leader + 1 worker)
    - Tensor parallelism size = 2 (`--tp-size 2`)
    - Disaggregation mode: `--disaggregation-mode decode`
    - Multi-node support via `--dist-init-addr`, `--nnodes`, `--node-rank`
    - Receives and processes KV cache from prefill workers

**Key Configuration Highlights**:
- **Shared Template**: Uses `dynamo-base` roleTemplate to avoid duplication across roles
- **Service Discovery**: ETCD and NATS endpoints configured via environment variables (`ETCD_ENDPOINTS`, `NATS_SERVER`)
- **Namespace**: Dynamo namespace configured via `DYN_NAMESPACE`
- **Rollout Strategy**: `InPlaceIfPossible` for minimal disruption during updates
- **Scaling Adapter**: Enabled for prefill and decode roles, allowing independent autoscaling

## Advanced Features

### Rollout Strategy with Partition

RBG supports partition-based rollouts for canary deployments, allowing you to test new versions on specific RoleInstance ordinals before rolling out to all replicas.

**Configure rollout strategy with partition**:

```yaml
roles:
  - name: prefill
    replicas: 10
    restartPolicy: None
    rolloutStrategy:
      type: RollingUpdate
      rollingUpdate:
        type: InPlaceIfPossible
        partition: 5
        maxUnavailable: 2
```

With `partition: 5`, only RoleInstances with ordinal >= 5 (RoleInstances 5-9) will be updated, while RoleInstances 0-4 remain on the old version.

**Trigger update and control partition**:

```bash
  # Update the image - only RoleInstances >= partition (5-9) will update
kubectl patch rolebasedgroup dynamo-pd-disagg-multi-node-inference --type='json' -p='[
  {
    "op": "replace",
    "path": "/spec/roleTemplates/0/template/spec/containers/0/image",
    "value": "nvcr.io/nvidia/ai-dynamo/sglang-runtime:new-version"
  }
]'

  # After validation, lower partition to continue rollout (e.g., update RoleInstances 3-4)
kubectl patch rolebasedgroup dynamo-pd-disagg-multi-node-inference --type='json' -p='[
  {
    "op": "replace",
    "path": "/spec/roles/1/rolloutStrategy/rollingUpdate/partition",
    "value": 3
  }
]'

  # After full validation, remove partition to complete rollout (update RoleInstances 0-2)
kubectl patch rolebasedgroup dynamo-pd-disagg-multi-node-inference --type='json' -p='[
  {
    "op": "remove",
    "path": "/spec/roles/1/rolloutStrategy/rollingUpdate/partition"
  }
]'
```

### In-Place Updates

RBG supports in-place updates for specific container fields (e.g., container image updates) without recreating pods, minimizing service disruption during upgrades.

#### Enable In-Place Updates

To enable in-place updates for a role, configure the `rolloutStrategy` with `InPlaceIfPossible`:

**Configuration Example**:

```yaml
roles:
  - name: prefill
    replicas: 1
    restartPolicy: None
    rolloutStrategy:
      type: RollingUpdate
      rollingUpdate:
        type: InPlaceIfPossible
        maxUnavailable: 1
    leaderWorkerPattern:
      size: 2
```

#### Perform In-Place Update

Once configured, update the container image:

```bash
kubectl patch rolebasedgroup dynamo-pd-disagg-multi-node-inference --type='json' -p='[
  {
    "op": "replace",
    "path": "/spec/roleTemplates/0/template/spec/containers/0/image",
    "value": "nvcr.io/nvidia/ai-dynamo/sglang-runtime:new-version"
  }
]'
```

**Expected output**:
```
NAME                                                 READY   STATUS    RESTARTS   AGE
dynamo-pd-disagg-multi-node-inference-processor-0    1/1     Running   1          16m
dynamo-pd-disagg-multi-node-inference-prefill-0      1/1     Running   1          16m
dynamo-pd-disagg-multi-node-inference-prefill-1      1/1     Running   1          16m
dynamo-pd-disagg-multi-node-inference-decode-0       1/1     Running   1          16m
dynamo-pd-disagg-multi-node-inference-decode-1       1/1     Running   1          16m
```

RBG will perform an in-place update if the change is eligible, avoiding pod reconstruct.

### Coordinated Updates

RBG supports coordinated rolling updates to upgrade multiple roles together with controlled update skew, ensuring prefill and decode workers stay synchronized.

```yaml
apiVersion: workloads.x-k8s.io/v1alpha2
kind: CoordinatedPolicy
metadata:
  name: dynamo-pd-disagg-multi-node-inference #must match the RoleBasedGroup name
  namespace: default
spec:
  policies:
    - name: pd-coordination
      roles: ["prefill", "decode"]
      strategy:
        rollingUpdate:
          maxSkew: 1%
          maxUnavailable: 10%
```

This ensures prefill and decode roles update in sync, with at most 10% difference in update progress between roles.

### Coordinated Scaling

When coordination is enabled, RBG ensures all roles in a coordination group scale proportionally, maintaining balanced capacity across disaggregated components:

```yaml
apiVersion: workloads.x-k8s.io/v1alpha2
kind: CoordinatedPolicy
metadata:
  name: dynamo-pd-disagg
  namespace: default
spec:
  policies:
    - name: pd-coordination
      roles: ["prefill", "decode"]
      strategy:
        scaling:
          maxSkew: "5%"           # Max 5% difference in deployment progress
          progression: OrderScheduled  # Progress based on pod scheduling
```

**Scaling Behaviors**:
- **maxSkew**: Controls the maximum deployment progress difference between roles (e.g., "5%", "10%")
- **progression**:
    - `OrderScheduled`: Progress when pods are scheduled (have nodeName)
    - `OrderReady`: Progress when pods are fully ready

RBG will automatically batch the scaling operations to ensure both prefill and decode roles grow at a similar pace, preventing resource imbalances.

### Pod Group (Gang Scheduling)

RBG supports gang scheduling through Kubernetes scheduler-plugins or Volcano, ensuring all pods in a role are scheduled together. Gang scheduling is enabled via annotations:

```yaml
apiVersion: workloads.x-k8s.io/v1alpha2
kind: RoleBasedGroup
metadata:
  name: dynamo-pd-disagg-lws
  namespace: default
  annotations:
    rbg.workloads.x-k8s.io/group-gang-scheduling: "true"
    # Optional: schedule timeout in seconds (default: 60)
    rbg.workloads.x-k8s.io/group-gang-scheduling-timeout: "120"
spec:
  roles:
    - name: prefill
      replicas: 2
      standalonePattern:
        template:
          spec:
            containers:
              - name: sglang
                image: nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.0.1
                #...
```

### Scaling Adapter (HPA Integration)

RBG provides a `scalingAdapter` feature that enables Autoscaler (HPA, KEDA and DynamoPlanner...) to automatically scale individual roles based on metrics.

#### Enable Scaling Adapter

To enable autoscaling for a role, add the `scalingAdapter` field with `enable: true`:

```yaml
apiVersion: workloads.x-k8s.io/v1alpha2
kind: RoleBasedGroup
metadata:
  name: dynamo-agg-inference
  namespace: default
spec:
  roles:
    - name: backend
      replicas: 1
      scalingAdapter:
        enable: true
      standalonePattern:
        template:
          spec:
            containers:
              - name: sglang
                image: nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.0.1
                #...
```

**Key Points**:
- When `scalingAdapter.enable` is set to `true`, the RBG controller automatically creates a `RoleBasedGroupScalingAdapter` CR for that role
- The ScalingAdapter's lifecycle is bound to the RoleBasedGroup (owner reference)
- You can directly use the generated ScalingAdapter as the target for HPA without manually creating it
- The ScalingAdapter name follows the pattern: `{rbg-name}-{role-name}`

#### Configure HPA

Once the ScalingAdapter is enabled, create an HPA targeting it. Here are examples based on NVIDIA Dynamo's autoscaling best practices:

For LLM-specific metrics like request rate, queue depth, or latency (TTFT), you can use custom metrics via Prometheus Adapter:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: dynamo-backend-hpa-custom
  namespace: default
spec:
  scaleTargetRef:
    apiVersion: workloads.x-k8s.io/v1alpha2
    kind: RoleBasedGroupScalingAdapter
    name: dynamo-agg-inference-backend
  minReplicas: 1
  maxReplicas: 10
  metrics:
    - type: External
      external:
        metric:
          name: dynamo_queued_requests
          selector:
            matchLabels:
              dynamo_namespace: "default"
        target:
          type: Value
          value: "10"
```

**Important Notes**:
- The `scaleTargetRef.name` must match the auto-generated ScalingAdapter name: `{rbg-name}-{role-name}`
- Each role with `scalingAdapter.enable: true` gets its own independent ScalingAdapter
- This allows different roles to have different HPA configurations and scale independently
- When the RoleBasedGroup is deleted, the associated ScalingAdapter is automatically garbage collected
- For custom metrics (like TTFT, queue depth), consider using **KEDA** instead of HPA

#### Manual Scaling

When scalingAdapter is enabled, you should scale via the ScalingAdapter (not directly on the RoleBasedGroup):

```bash
# scale via RoleBasedGroupScalingAdapter
kubectl scale rolebasedgroupscalingadapter dynamo-agg-inference-backend --replicas=3 -n default
```

## Integrated with Dynamo Ecosystem

### Backend Engines

| Backend                     | Feature Support | Documentation |
|----------------------------|-----------------|-------------|
| SGLang Runtime      | ✅              | ✅          |
| vLLM Runtime        | ✅              | 🚧           |
| TensorRT-LLM Runtime | ✅              | 🚧           |

### AIConfigurator

[kubectl rbg-plugin](https://github.com/sgl-project/rbg/blob/main/doc/features/kubectl-rbg-llm-generate.md) is a Kubectl Plugin that generates optimized RBG deployment configurations for LLM inference workloads by [Dynamo AIConfigurator](https://github.com/ai-dynamo/aiconfigurator),
which automatically analyzes model requirements, hardware constraints, and performance characteristics to produce production-ready YAML configurations.

**Key capabilities**:
- Use Dynamo AIConfigurator to perform automatic model profiling and resource estimation, generating optimized tensor parallelism and pipeline parallelism configuration
- Best-practice defaults for rollout strategies and scaling policies
- Support for both aggregated and disaggregated serving architectures

### Dynamo SLA Planner

🚧 **Coming Soon**

The SLA-Based Planner will provide intelligent, SLA-driven autoscaling decisions for RBG workloads, optimizing resource allocation to meet latency targets at minimum cost.

**Legend**: ✅ Supported | 🚧 Under Development


## Additional Resources

### Dynamo Documentation

- [Disaggregated Serving Design](../design-docs/disagg-serving.md)
- [Backend Deployment Examples](../backends/)

### RBG Documentation

- [RBG GitHub Repository](https://github.com/sgl-project/rbg)
- [RBG Documentation](https://github.com/sgl-project/rbg/tree/main/doc)
