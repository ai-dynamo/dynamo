# Dynamo Integration with RoleBasedGroup (RBG)

This directory provides integration examples for deploying NVIDIA Dynamo with [RoleBasedGroup (RBG)](https://github.com/sgl-project/rbg), demonstrating both aggregated and disaggregated inference architectures for large language models.

## Table of Contents

- [What is RoleBasedGroup (RBG)?](#what-is-rolebasedgroup-rbg)
- [Supported Features](#supported-features)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Quick Start Example](#quick-start-example)
- [Advanced Features](#advanced-features)

## What is RoleBasedGroup (RBG)?

**RoleBasedGroup (RBG)** is a Kubernetes Custom Resource Definition (CRD) designed for orchestrating distributed, stateful AI inference workloads with multi-role collaboration and built-in service discovery. It addresses the limitations of traditional Kubernetes primitives (StatefulSets, Deployments) when managing complex distributed inference services.

RBG models an inference service as a **role-based group** rather than a collection of independent workloads, treating it as a topological, stateful, and coordinated multi-role system managed as a single unit.

### Why RBG for Dynamo?

RBG provides a loosely-coupled deployment approach in Kubernetes, enabling users to quickly validate distributed inference services without deep integration with specific platforms. Traditional Kubernetes primitives face the following challenges when managing distributed inference services:

- Run as **multi-role topologies** (frontend, prefill workers, decode workers)
- Are **stability-sensitive**, requiring capabilities such as in-place updates and fault recovery to ensure service availability
- Require **multi-role coordination**, where multiple roles need to stay coordinated and synchronized during deployment, scaling, and updates

RBG is particularly well-suited for Dynamo's **disaggregated serving architectures** (such as prefill/decode separation), where multiple specialized worker roles must work together seamlessly. It provides an alternative deployment model to Dynamo's native **DynamoGraphDeployment** CRD.

For more information about RBG, see the [RBG documentation](https://github.com/sgl-project/rbg/blob/main/docs/README.md).

## Supported Features

### Backend Engines

| Backend                     | Feature Support | Documentation |
|----------------------------|-----------------|-------------|
| SGLang Runtime      | ✅              | ✅          |
| vLLM Runtime        | ✅              | 🚧           |
| TensorRT-LLM Runtime | ✅              | 🚧           |

### Dynamo Platform Features

| Feature | Description | Feature Support |  Documentation |
|---------|-------------|--------| -------------|
| [SLA-Based Planner](../../../docs/planner/sla_planner.md) | Apply SLA-driven autoscaling decisions to RBG workloads | ✅ | 🚧 |
| [KVBM](../../../docs/kvbm/kvbm_architecture.md)  | Efficient KV cache block allocation and management | 🚧 | 🚧 |
| AIConfigurator | Generate optimized RBG deployment configurations automatically | 🚧 | 🚧 |

**Legend**: ✅ Supported | 🚧 Under Development

## Installation

### Prerequisites

- Kubernetes cluster version >= 1.28
- kubectl command-line tool configured to communicate with your cluster
- At least 1 node with 1+ CPUs and 1GB memory for the RBG controller
- (Optional) [LeaderWorkerSet](https://github.com/kubernetes-sigs/lws) >= v0.7.0 for multi-node deployments

### Install RBG Controller

#### Option 1: Install with kubectl

```bash
kubectl apply --server-side -f https://github.com/sgl-project/rbg/releases/latest/download/manifests.yaml
```

Wait for the controller to be ready:

```bash
kubectl wait deploy/rbgs-controller-manager -n rbgs-system --for=condition=available --timeout=5m
```

#### Option 2: Install with Helm

```bash
helm repo add rbgs https://sgl-project.github.io/rbg
helm install rbgs rbgs/rbgs -n rbgs-system --create-namespace
```

### Verify Installation

```bash
kubectl get pods -n rbgs-system
```

You should see the RBG controller manager running.

## Directory Structure

This directory is organized as follows:

```
dynamo/
├── README.md                    # This file
└── sglang/                      # SGLang integration examples
    ├── agg.yaml                 # Aggregated architecture (single-node)
    ├── agg-multinode.yaml       # Aggregated architecture (multi-node)
    ├── disagg.yaml              # Disaggregated architecture (single-node)
    └── disagg-multinode.yaml    # Disaggregated architecture (multi-node)
```

**File Naming Convention**:
- `agg*`: Aggregated prefill/decode architecture
- `disagg*`: Disaggregated prefill/decode architecture
- `*-multinode`: Multi-node deployment

## Quick Start Example

This section demonstrates how to deploy a disaggregated Dynamo inference service using RBG.

### Prerequisites

Before deploying, ensure you have:

1. **RBG Controller** installed (see [Installation](#installation) section above)

2. **Dynamo Dependencies** in your Kubernetes cluster:
   - ETCD service for distributed key-value storage (typically `http://etcd:2379`)
   - NATS service with JetStream enabled for message streaming (typically `nats://nats:4222`)
   - For local development: see [docker-compose.yml](../../../deploy/docker-compose.yml)

3. **Model Storage** configured:
   - PersistentVolume/PersistentVolumeClaim for model files, or
   - HuggingFace token secret for automatic model downloads

4. **GPU Resources**:
   - NVIDIA GPU drivers installed on cluster nodes
   - NVIDIA device plugin for Kubernetes
   - (Optional) RDMA network interface for high-performance disaggregated serving

### Deploy Disaggregated Architecture

Navigate to the SGLang integration examples directory:

```bash
cd examples/deployments/rolebasedgroup/sglang
```

Deploy the disaggregated architecture:

```bash
kubectl apply -f disagg.yaml
```

### Verify Deployment

Check the RoleBasedGroup status:

```bash
kubectl get rolebasedgroup dynamo-pd
```

**Expected output**:
```
NAME        READY   AGE
dynamo-pd   True    2m
```

Check all pods are running:

```bash
kubectl get pods -l rbg.workloads.x-k8s.io/rbg-name=dynamo-pd
```

**Expected output**:
```
NAME                       READY   STATUS    RESTARTS   AGE
dynamo-pd-frontend-0       1/1     Running   0          2m
dynamo-pd-prefill-0        1/1     Running   0          2m
dynamo-pd-decoder-0        1/1     Running   0          2m
```

View detailed deployment status:

```bash
kubectl describe rolebasedgroup dynamo-pd
```

Check role-specific status:

```bash
kubectl get rolebasedgroup dynamo-pd -o jsonpath='{.status.roles}' | jq
```

**Expected output**:
```json
[
  {
    "name": "frontend",
    "replicas": 1,
    "readyReplicas": 1,
    "updatedReplicas": 1
  },
  {
    "name": "prefill",
    "replicas": 2,
    "readyReplicas": 2,
    "updatedReplicas": 2
  },
  {
    "name": "decoder",
    "replicas": 1,
    "readyReplicas": 1,
    "updatedReplicas": 1
  }
]
```

### Test the Deployment

Forward the frontend service port to localhost:

```bash
kubectl port-forward pod/dynamo-pd-frontend-0 8000:8000
```

In another terminal, send a test request:

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-235B",
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

The `disagg.yaml` example creates a three-role RoleBasedGroup:

1. **Frontend Role** (`frontend`):
   - OpenAI-compatible HTTP API server listening on port 8000
   - Routes and coordinates requests between prefill and decode workers
   - 1 replica, CPU-only (no GPU required)

2. **Prefill Role** (`prefill`):
   - Specialized worker for prompt processing (prefill phase)
   - 4 GPUs per replica with tensor parallelism (`--tp 4`)
   - Disaggregation mode: `--disaggregation-mode prefill`
   - Transfers KV cache to decode workers via NIXL high-speed backend

3. **Decode Role** (`decoder`):
   - Specialized worker for token generation (decode phase)
   - 8 GPUs per replica with expert parallelism (`--ep-size 8` for MoE models)
   - Disaggregation mode: `--disaggregation-mode decode`
   - Receives and processes KV cache from prefill workers

**Key Configuration Highlights**:
- **Shared Template**: Uses `dynamo-common` roleTemplate to avoid duplication across roles
- **Service Discovery**: ETCD and NATS endpoints configured via environment variables (`DYN_ETCD_ENDPOINTS`, `DYN_NATS_ENDPOINTS`)
- **Model Storage**: Model loaded from PersistentVolumeClaim mounted at `/models/qwen3-235B`
- **Resource Allocation**: GPU count and RDMA network resources properly specified per role
- **Health Checks**: Readiness probes ensure frontend is healthy before routing traffic

## Advanced Features

### In-Place Updates

RBG supports in-place updates for specific container fields (e.g., container image updates) without recreating pods, minimizing service disruption during upgrades.

#### Enable In-Place Updates

To enable in-place updates for a role, you need to configure two fields:

1. **Workload Type**: Set the workload to `InstanceSet`
2. **Rollout Strategy**: Configure the update type as `InPlaceIfPossible`

**Configuration Example**:

```yaml
roles:
  - name: prefill
    replicas: 1
    workload:
      apiVersion: workloads.x-k8s.io/v1alpha1
      kind: InstanceSet
    rolloutStrategy:
      type: RollingUpdate
      rollingUpdate:
        type: InPlaceIfPossible
```

#### Perform In-Place Update

Once configured, update the container image:

```bash
kubectl patch rolebasedgroup dynamo-pd --type='json' -p='[
  {
    "op": "replace",
    "path": "/spec/roleTemplate/0/template/spec/containers/0/image",
    "value": "nvcr.io/nvidia/ai-dynamo/sglang-runtime:new-version"
  }
]'
```

RBG will perform an in-place update if the change is eligible, avoiding pod restarts.

**KVBM Benefits**: For prefill workers with KVBM enabled using Host as the KV cache StoragePool, in-place updates allow the prefill worker to continue reusing the KV cache from the local host after the update, maintaining cache locality.

### Coordinated Updates

RBG supports coordinated rolling updates to upgrade multiple roles together with controlled update skew, ensuring prefill and decode workers stay synchronized:

```yaml
spec:
  coordination:
    - name: pd-coordination
      roles: ["prefill", "decoder"]
      strategy:
         rollingUpdate:
            maxSkew: 1%
            maxUnavailable: 10%
```

This ensures prefill and decode roles update in sync, with at most 10% difference in update progress between roles.

### Coordinated Scaling

When coordination is enabled, RBG ensures all roles in a coordination group scale proportionally, maintaining balanced capacity across disaggregated components:

```yaml
spec:
  coordination:
    - name: pd-coordination
      roles: ["prefill", "decoder"]
      strategy:
        scaling:
          maxSkew: "5%"           # Max 5% difference in deployment progress
          progressionStrategy: "OrderScheduled"  # Progress based on pod scheduling
```

**Scaling Behaviors**:
- **maxSkew**: Controls the maximum deployment progress difference between roles (e.g., "5%", "10%")
- **progressionStrategy**: 
  - `OrderScheduled`: Progress when pods are scheduled (have nodeName)
  - `OrderReady`: Progress when pods are fully ready

RBG will automatically batch the scaling operations to ensure both prefill and decode roles grow at a similar pace, preventing resource imbalances.

### Rollout Strategy with Partition

RBG supports partition-based rollouts for canary deployments, allowing you to test new versions on specific pod ordinals before rolling out to all replicas.

**Configure rollout strategy with partition**:

```yaml
roles:
  - name: prefill
    replicas: 10
    rolloutStrategy:
      type: RollingUpdate
      rollingUpdate:
        partition: 5
        maxUnavailable: 2
```

With `partition: 5`, only pods with ordinal >= 5 (pods 5-9) will be updated, while pods 0-4 remain on the old version.

**Trigger update and control partition**:

```bash
# Update the image - only pods >= partition (5-9) will update
kubectl patch rolebasedgroup dynamo-pd --type='json' -p='[
  {
    "op": "replace",
    "path": "/spec/roleTemplates/0/template/spec/containers/0/image",
    "value": "nvcr.io/nvidia/ai-dynamo/sglang-runtime:new-version"
  }
]'

# After validation, lower partition to continue rollout (e.g., update pods 3-4)
kubectl patch rolebasedgroup dynamo-pd --type='json' -p='[
  {
    "op": "replace",
    "path": "/spec/roles/1/rolloutStrategy/rollingUpdate/partition",
    "value": 3
  }
]'

# After full validation, remove partition to complete rollout (update pods 0-2)
kubectl patch rolebasedgroup dynamo-pd --type='json' -p='[
  {
    "op": "remove",
    "path": "/spec/roles/1/rolloutStrategy/rollingUpdate/partition"
  }
]'
```

## Additional Resources

### Dynamo Documentation

- [Disaggregated Serving Design](../../../docs/design_docs/disagg_serving.md)
- [Backend Deployment Examples](../../../docs/backends/)

### RBG Documentation

- [RBG GitHub Repository](https://github.com/sgl-project/rbg)
- [RBG Documentation](https://github.com/sgl-project/rbg/tree/main/doc)
