# Multinode Deployment Guide

This guide explains how to deploy Dynamo workloads across multiple nodes. Multinode deployments enable you to scale compute-intensive LLM workloads across multiple physical machines, maximizing GPU utilization and supporting larger models.

## Overview

Dynamo supports multinode deployments through the `numberOfNodes` field in resource specifications. This field allows you to:

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
- (optional) **[KAI-Scheduler](https://github.com/NVIDIA/KAI-Scheduler)**: Kubernetes native scheduler optimized for AI workloads at scale

These systems provide enhanced scheduling capabilities including topology-aware placement, gang scheduling, and coordinated auto-scaling across multiple nodes.

**Features Enabled with Grove:**
- Hierarchical gang scheduling with `PodGangSet` and `PodClique`
- Multi-level horizontal auto-scaling
- Custom startup ordering for components
- Resource-aware rolling updates


[KAI-Scheduler](https://github.com/NVIDIA/KAI-Scheduler) is an optional enhancement that provides a Kubernetes native scheduler optimized for AI workloads at large scale.

**Features Enabled with KAI-Scheduler:**
- Network topology-aware pod placement
- AI workload-optimized scheduling algorithms
- GPU resource awareness and allocation
- Support for complex scheduling constraints
- Integration with Grove for enhanced capabilities
- Performance optimizations for large-scale deployments

#### Using LWS and Volcano

LWS is a simple multinode deployment mechanism that allows you to deploy a workload across multiple nodes.

- **LWS**: [LWS Installation](https://github.com/NVIDIA/LWS#installation)
- **Volcano**: [Volcano Installation](https://volcano.sh/docs/installation/install-volcano/)

Volcano is a Kubernetes native scheduler optimized for AI workloads at scale. It is used in conjunction with LWS to provide gang scheduling support.


## Core Concepts

### The `numberOfNodes` Field

The `numberOfNodes` field in a resource specification defines how many physical nodes the workload should span:

```yaml
numberOfNodes: 2
resources:
  requests:
    cpu: "10"
    memory: "40Gi"
  limits:
    cpu: "10"
    memory: "40Gi"
    gpu: "2"            # 2 GPUs per node
```

### GPU Distribution

The relationship between `numberOfNodes` and `gpu` is multiplicative:

- **`numberOfNodes`**: Number of physical nodes
- **`gpu`**: Number of GPUs per node
- **Total GPUs**: `numberOfNodes × gpu`

**Example:**
- `numberOfNodes: "2"` + `gpu: "4"` = 8 total GPUs (4 GPUs per node across 2 nodes)
- `numberOfNodes: "4"` + `gpu: "8"` = 32 total GPUs (8 GPUs per node across 4 nodes)

### Tensor Parallelism Alignment

The tensor parallelism (`tp-size` or `--tp`) in your command/args must match the total number of GPUs:

```yaml
# Example: 2 numberOfNodes × 4 GPUs = 8 total GPUs
numberOfNodes: 2
resources:
  limits:
    gpu: "4"

# Command args must use tp-size=8
args:
  - "--tp-size"
  - "8"  # Must equal numberOfNodes × gpu
```


## Next Steps

For additional support and examples, see the working multinode configurations in:

- **SGLang**: [components/backends/sglang/deploy/](../../components/backends/sglang/deploy/)
- **TensorRT-LLM**: [components/backends/trtllm/deploy/](../../components/backends/trtllm/deploy/)
- **vLLM**: [components/backends/vllm/deploy/](../../components/backends/vllm/deploy/)

These examples demonstrate proper usage of the `numberOfNodes` field with corresponding `gpu` limits and correct `tp-size` configuration.