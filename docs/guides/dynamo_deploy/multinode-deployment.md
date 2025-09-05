# Multinode Deployment Guide

This guide explains how to deploy Dynamo workloads across multiple nodes. Multinode deployments enable you to scale compute-intensive LLM workloads across multiple physical machines, maximizing GPU utilization and supporting larger models.

## Overview

Dynamo supports multinode deployments through the `multinode` section in resource specifications. This allows you to:

- Distribute workloads across multiple physical nodes
- Scale GPU resources beyond a single machine
- Support large models requiring extensive tensor parallelism
- Achieve high availability and fault tolerance

## Basic requirements

- **Kubernetes Cluster**: Version 1.24 or later
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
- (Optional) [KAI-Scheduler](https://github.com/NVIDIA/KAI-Scheduler) installed on the cluster with default queue name `dynamo` created. You can use a different queue name by setting the `nvidia.com/kai-scheduler-queue` annotation on the DGD resource.

KAI-Scheduler is optional but recommended for advanced scheduling capabilities.

#### Using LWS and Volcano

LWS is a simple multinode deployment mechanism that allows you to deploy a workload across multiple nodes.

- **LWS**: [LWS Installation](https://github.com/kubernetes-sigs/lws#installation)
- **Volcano**: [Volcano Installation](https://volcano.sh/en/docs/installation/)

Volcano is a Kubernetes native scheduler optimized for AI workloads at scale. It is used in conjunction with LWS to provide gang scheduling support.


## Core Concepts

### Orchestrator Selection Algorithm

Dynamo automatically selects the best available orchestrator for multinode deployments using the following logic:

#### When Both Grove and LWS are Available:
- **Grove is selected by default** (recommended for advanced AI workloads)
- **LWS is selected** if you explicitly set `nvidia.com/enable-grove: "false"` annotation on your DGD resource

#### When Only One Orchestrator is Available:
- The installed orchestrator (Grove or LWS) is automatically selected

#### Scheduler Integration:
- **With Grove**: Automatically integrates with [KAI-Scheduler](https://github.com/NVIDIA/KAI-Scheduler) when available, providing:
  - Advanced queue management via `nvidia.com/kai-scheduler-queue` annotation
  - AI-optimized scheduling policies
  - Resource-aware workload placement
- **With LWS**: Uses Volcano scheduler for gang scheduling and resource coordination

#### Configuration Examples:

**Default (Grove with KAI-Scheduler):**
```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-multinode-deployment
  annotations:
    nvidia.com/kai-scheduler-queue: "gpu-intensive"  # Optional: defaults to "dynamo"
spec:
  # ... your deployment spec
```

**Force LWS usage:**
```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-multinode-deployment
  annotations:
    nvidia.com/enable-grove: "false"
spec:
  # ... your deployment spec
```


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


## Next Steps

For additional support and examples, see the working multinode configurations in:

- **SGLang**: [components/backends/sglang/deploy/](../../../components/backends/sglang/deploy/)
- **TensorRT-LLM**: [components/backends/trtllm/deploy/](../../../components/backends/trtllm/deploy/)
- **vLLM**: [components/backends/vllm/deploy/](../../../components/backends/vllm/deploy/)

These examples demonstrate proper usage of the `multinode` section with corresponding `gpu` limits and correct `tp-size` configuration.
