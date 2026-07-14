---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Multinode Deployments
subtitle: Scales Dynamo inference across multiple GPU nodes with Grove, LWS, and DisaggregatedSet.
---

This guide explains how to deploy Dynamo workloads across multiple nodes. Multinode deployments enable you to scale compute-intensive LLM workloads across multiple physical machines, maximizing GPU utilization and supporting larger models.

## Overview

Dynamo supports multinode deployments through the `multinode` section in resource specifications. This allows you to:

- Distribute workloads across multiple physical nodes
- Scale GPU resources beyond a single machine
- Support large models requiring extensive tensor parallelism
- Achieve high availability and fault tolerance

## Basic requirements

- **Kubernetes Cluster**: Version 1.30 or later
- **GPU Nodes**: Multiple nodes with NVIDIA GPUs
- **High-Speed Networking**: InfiniBand, RoCE, or high-bandwidth Ethernet (recommended for optimal performance)


### Advanced Multinode Orchestration

#### Using Grove (default)

For sophisticated multinode deployments, Dynamo integrates with advanced Kubernetes orchestration systems:

- **[Grove](https://github.com/NVIDIA/grove)**: Network topology-aware gang scheduling and auto-scaling for AI workloads
- **[KAI-Scheduler](https://github.com/NVIDIA/KAI-Scheduler)**: Kubernetes native scheduler optimized for AI workloads at scale

These systems provide enhanced scheduling capabilities including topology-aware placement, gang scheduling, and coordinated auto-scaling across multiple nodes.

**Features Enabled with Grove:**
- Declarative composition of AI workloads
- Multi-level horizontal auto-scaling
- Custom startup ordering for components
- Resource-aware rolling updates
- [Topology Aware Scheduling](../topology-aware-scheduling.md) — pack pods within a rack, block, or other topology domain for lower latency


[KAI-Scheduler](https://github.com/NVIDIA/KAI-Scheduler) is a Kubernetes native scheduler optimized for AI workloads at large scale.

**Features Enabled with KAI-Scheduler:**
- Gang scheduling
- Network topology-aware pod placement
- AI workload-optimized scheduling algorithms
- GPU resource awareness and allocation
- Support for complex scheduling constraints
- Integration with Grove for enhanced capabilities
- Performance optimizations for large-scale deployments


##### Prerequisites

- [Grove](https://github.com/NVIDIA/grove/blob/main/docs/installation.md) installed on the cluster
- (Optional) [KAI-Scheduler](https://github.com/NVIDIA/KAI-Scheduler) installed on the cluster with the default queue name `dynamo` created. If no queue annotation is specified on the DGD resource, the operator uses the `dynamo` queue by default. Custom queue names can be specified via the `nvidia.com/kai-scheduler-queue` annotation, but the queue must exist in the cluster before deployment.

KAI-Scheduler is optional but recommended for advanced scheduling capabilities.

#### Using LWS and Volcano

LWS is a simple multinode deployment mechanism that allows you to deploy a workload across multiple nodes.

- **LWS**: [LWS Installation](https://github.com/kubernetes-sigs/lws#installation)
- **Volcano**: [Volcano Installation](https://github.com/volcano-sh/volcano#quick-start-guide)

Volcano is a Kubernetes native scheduler optimized for AI workloads at scale. It is used in conjunction with LWS to provide gang scheduling support.

#### Using DisaggregatedSet

DisaggregatedSet (DS) is an opt-in multinode path for clusters that install the `disaggregatedset.x-k8s.io/v1` API. Dynamo uses it to group multiple eligible multinode worker roles under one DS object instead of creating one LeaderWorkerSet per selected component.


## Core Concepts

### Orchestrator Selection Algorithm

Dynamo automatically selects the best available orchestrator for multinode deployments using the following logic:

#### When Both Grove and LWS are Available:
- **Grove is selected by default** (recommended for advanced AI workloads)
- **LWS is selected** if you explicitly set `nvidia.com/enable-grove: "false"` annotation on your DGD resource

#### When the DisaggregatedSet API is Available:
- **DS is selected** only if you set `nvidia.com/enable-disaggregatedset: "true"`
- **Grove still wins by default** when Grove is enabled, so also set `nvidia.com/enable-grove: "false"` if you want the DS path on clusters that have Grove
- **LWS is used as the fallback** when the DS request cannot be honored

#### When Only One Orchestrator is Available:
- The installed orchestrator (Grove or LWS) is automatically selected

#### Scheduler Integration:
- **With Grove**: Dynamo uses Grove for multinode orchestration when the Grove API is available, unless you set `nvidia.com/enable-grove: "false"` on the DGD resource. Scheduler integration is configured separately:
  - KAI-Scheduler: set `global.kai-scheduler.install=true` to install the bundled KAI-Scheduler chart and enable integration, or set `global.kai-scheduler.enabled=true` when KAI-Scheduler is already installed externally and its API is available. Select queues with `nvidia.com/kai-scheduler-queue`.
  - **EXPERIMENTAL:** Volcano: Dynamo does not install Volcano. Set `global.volcano-scheduler.enabled=true` only when Volcano is already installed and the Volcano API is available. Select queues with `nvidia.com/volcano-queue`.
  - KAI-Scheduler and Volcano scheduler integration are mutually exclusive for a single Dynamo operator configuration because both set pod `schedulerName`. Helm rejects configurations that enable both integrations.
- **With LWS**: Uses Volcano scheduler for gang scheduling and resource coordination
- **With DisaggregatedSet**: Reuses the same multinode pod-template rendering as the LWS path, but groups selected worker roles into one DS. DS API detection is separate from Volcano detection.

> **EXPERIMENTAL:** The Dynamo/Grove Volcano scheduler integration is newly introduced and opt-in. Volcano itself is a mature CNCF scheduler, but this integration is intended for clusters where Volcano is already installed and understood by the platform operator.
>
> Changing `global.volcano-scheduler.enabled` affects how the Dynamo operator reconciles Grove-backed DGD resources. When enabled, Dynamo generates Grove resources that use `schedulerName: volcano` and propagates the Volcano queue annotation. Treat this Helm value as a scheduling-mode change for existing Grove-backed DGDs, not as a metadata-only toggle.

#### Configuration Examples:

**Grove with KAI-Scheduler:**
```yaml
global:
  grove:
    enabled: true
  kai-scheduler:
    enabled: true
```

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: my-multinode-deployment
  annotations:
    nvidia.com/kai-scheduler-queue: "dynamo"
spec:
  # ... your deployment spec
```

> **Note:** The `nvidia.com/kai-scheduler-queue` annotation defaults to `"dynamo"`. If you specify a custom queue name, ensure the queue exists in your cluster before deploying. You can verify available queues with `kubectl get queues`.

**Grove with Volcano:**
```yaml
global:
  grove:
    enabled: true
  volcano-scheduler:
    enabled: true
```

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: my-multinode-deployment
  annotations:
    nvidia.com/volcano-queue: "gpu-training"
spec:
  # ... your deployment spec
```

> **Note:** The `nvidia.com/volcano-queue` annotation is propagated to Grove as `scheduling.grove.io/volcano-queue`. If you specify a custom queue name, ensure the Volcano queue exists and is open before deploying. You can verify available queues with `kubectl get queues`.

**Force LWS usage:**
```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: my-multinode-deployment
  annotations:
    nvidia.com/enable-grove: "false"
spec:
  # ... your deployment spec
```

**Request DisaggregatedSet:**
```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: my-ds-deployment
  annotations:
    nvidia.com/enable-grove: "false"
    nvidia.com/enable-disaggregatedset: "true"
spec:
  # ... your deployment spec
```

DS requests fall back to the standard DCD pathway when any of these conditions apply. Multinode components require LWS + Volcano after fallback; without that pathway, reconciliation reports that no multinode orchestrator is available.

- The `disaggregatedset.x-k8s.io/v1` API is not available.
- Fewer than two eligible multinode worker roles are selected.
- A selected component uses `scalingAdapter`.
- Selected roles mix zero replicas with positive replicas.
- More than 10 eligible multinode worker roles are selected.


### The `multinode` Section

The `multinode` section in a resource specification defines how many physical nodes the workload should span:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-multinode-deployment
spec:
  # ... your deployment spec
  services:
    my-service:
      ...
      multinode:
        nodeCount: 2
      resources:
        limits:
          gpu: "2"            # 2 GPUs per node
```

### GPU Distribution

The relationship between `multinode.nodeCount` and `gpu` is multiplicative:

- **`multinode.nodeCount`**: Number of physical nodes
- **`gpu`**: Number of GPUs per node
- **Total GPUs**: `multinode.nodeCount × gpu`

**Example:**
- `multinode.nodeCount: "2"` + `gpu: "4"` = 8 total GPUs (4 GPUs per node across 2 nodes)
- `multinode.nodeCount: "4"` + `gpu: "8"` = 32 total GPUs (8 GPUs per node across 4 nodes)

### Tensor Parallelism Alignment

The tensor parallelism (`tp-size` or `--tp`) in your command/args must match the total number of GPUs:

```yaml
# Example: 2 multinode.nodeCount × 4 GPUs = 8 total GPUs
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-multinode-deployment
spec:
  # ... your deployment spec
  services:
    my-service:
      ...
      multinode:
        nodeCount: 2
      resources:
        limits:
          gpu: "4"
      extraPodSpec:
        mainContainer:
          ...
          args:
            # Command args must use tp-size=8
            - "--tp-size"
            - "8"  # Must equal multinode.nodeCount × gpu

```


## Backend-Specific Operator Behavior

When you deploy a multinode workload, the Dynamo operator automatically applies backend-specific configurations to enable distributed execution. Understanding these automatic modifications helps troubleshoot issues and optimize your deployments.

### vLLM Backend

For vLLM multinode deployments, the operator automatically selects and configures the appropriate distributed execution mode based on your parallelism settings:

#### Deployment Modes

The operator automatically determines the deployment mode based on your parallelism configuration:

**1. Tensor/Pipeline Parallelism Mode (Single model across nodes)**
- **When used**: When `world_size > GPUs_per_node` where `world_size = tensor_parallel_size × pipeline_parallel_size`
- **Use case**: Distributing a single model instance across multiple nodes using tensor or pipeline parallelism

The operator uses Ray for multi-node tensor/pipeline parallel deployments. Ray provides automatic placement group management and worker spawning across nodes.

**Leader Node:**
- **Command**: `ray start --head --port=6379 && <original-vllm-command> --distributed-executor-backend ray`
- **Behavior**: Starts Ray head node, then runs vLLM which creates a placement group spanning all Ray workers
- **Probes**: All health probes remain active (liveness, readiness, startup)

**Worker Nodes:**
- **Command**: `ray start --address=<leader-hostname>:6379 --block`
- **Behavior**: Joins Ray cluster and blocks; vLLM on leader spawns Ray actors to these workers
- **Probes**: All probes (liveness, readiness, startup) are automatically removed

<Note>
vLLM's Ray executor automatically creates a placement group and spawns workers across the cluster. The `--nnodes` flag is NOT used with Ray - it's only compatible with the `mp` backend.
</Note>

**2. Data Parallel Mode (Multiple model instances across nodes)**
- **When used**: When `world_size × data_parallel_size > GPUs_per_node`
- **Use case**: Running multiple independent model instances across nodes with data parallelism (e.g., MoE models with expert parallelism)

**All Nodes (Leader and Workers):**
- **Injected Flags**:
  - `--data-parallel-address <leader-hostname>` - Address of the coordination server
  - `--data-parallel-size-local <value>` - Number of data parallel workers per node
  - `--data-parallel-rpc-port 13445` - RPC port for data parallel coordination
  - `--data-parallel-start-rank <value>` - Starting rank for this node (calculated automatically)
- **Probes**: Worker probes are removed; leader probes remain active

**Note**: The operator intelligently injects these flags into your command regardless of command structure (direct Python commands or shell wrappers)

#### Why Ray for Multi-Node TP/PP?

vLLM supports two distributed executor backends: `ray` and `mp`. For multi-node deployments:

- **Ray executor**: vLLM creates a placement group and spawns Ray actors across the cluster. Workers don't run vLLM directly - the leader's vLLM process manages everything.
- **mp executor**: Each node must run its own vLLM process with `--nnodes`, `--node-rank`, `--master-addr`, `--master-port`. This approach is more complex to orchestrate.

The Dynamo operator uses Ray because:
1. It aligns with vLLM's official multi-node documentation (see `multi-node-serving.sh`)
2. Simpler orchestration - only the leader runs vLLM, workers just need Ray agents
3. vLLM automatically handles placement group creation and worker management

#### Compilation Cache Support
When a volume mount is configured with `useAsCompilationCache: true`, the operator automatically sets:
- **`VLLM_CACHE_ROOT`**: Environment variable pointing to the cache mount point

### SGLang Backend

For SGLang multinode deployments, the operator injects distributed training parameters:

#### Leader Node
- **Distributed Flags**: Injects `--dist-init-addr <leader-hostname>:29500 --nnodes <count> --node-rank 0`
- **Probes**: All health probes remain active

#### Worker Nodes
- **Distributed Flags**: Injects `--dist-init-addr <leader-hostname>:29500 --nnodes <count> --node-rank <dynamic-rank>`
  - The `node-rank` is automatically determined from the pod's stateful identity
- **Probes**: All probes (liveness, readiness, startup) are automatically removed

**Note:** The operator intelligently injects these flags regardless of your command structure (direct Python commands or shell wrappers).

### TensorRT-LLM Backend

For TensorRT-LLM multinode deployments, the operator configures MPI-based communication:

#### Leader Node
- **SSH Configuration**: Automatically sets up SSH keys and configuration from a Kubernetes secret
- **MPI Command**: Wraps your command in an `mpirun` command with:
  - Proper host list including all worker nodes
  - SSH configuration for passwordless authentication on port 2222
  - Environment variable propagation to all nodes
  - Activation of the Dynamo virtual environment
- **Probes**: All health probes remain active

#### Worker Nodes
- **SSH Daemon**: Replaces your command with SSH daemon setup and execution
  - Generates host keys in user-writable directories (non-privileged)
  - Configures SSH daemon to listen on port 2222
  - Sets up authorized keys for leader access
- **Probes**:
  - **Liveness and Startup**: Removed (workers run SSH daemon, not the main application)
  - **Readiness**: Replaced with TCP socket check on SSH port 2222
    - Initial Delay: 20 seconds
    - Period: 20 seconds
    - Timeout: 5 seconds
    - Failure Threshold: 10

#### Additional Configuration
- **Environment Variable**: `OMPI_MCA_orte_keep_fqdn_hostnames=1` is added to all nodes
- **SSH Volume**: Automatically mounts the SSH keypair secret (typically named `mpirun-ssh-key-<deployment-name>`)
- **Automatic SSH key generation**: The operator automatically generates the SSH keypair secret when it detects a multi-node `DynamoGraphDeployment`. No manual secret creation is required.

### Compilation Cache Configuration

The operator supports compilation cache volumes for backend-specific optimization:

| Backend | Support Level | Environment Variables | Default Mount Point |
|---------|--------------|----------------------|---------------------|
| vLLM | Fully Supported | `VLLM_CACHE_ROOT` | User-specified |
| SGLang | Partial Support | _None (pending upstream)_ | User-specified |
| TensorRT-LLM | Partial Support | _None (pending upstream)_ | User-specified |

To enable compilation cache, add a volume mount with `useAsCompilationCache: true` in your component specification. For vLLM, the operator will automatically configure the necessary environment variables. For other backends, volume mounts are created, but additional environment configuration may be required until upstream support is added.

## Next Steps

For additional support and examples, see the working multinode configurations in:

- **SGLang**: [examples/backends/sglang/deploy/](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/sglang/deploy/README.md)
- **TensorRT-LLM**: [examples/backends/trtllm/deploy/](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/trtllm/deploy/README.md)
- **vLLM**: [examples/backends/vllm/deploy/](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/deploy/README.md)

These examples demonstrate proper usage of the `multinode` section with corresponding `gpu` limits and correct `tp-size` configuration.
