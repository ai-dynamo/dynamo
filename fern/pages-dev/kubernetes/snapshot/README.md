---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Checkpointing
---

> ⚠️ **Experimental Feature**: Dynamo Snapshot is currently in **beta/preview**. The Dynamo Snapshot DaemonSet runs in privileged mode to perform CRIU operations. See [Limitations](#limitations) for details.

**Dynamo Snapshot** (Checkpoint/Restore in Kubernetes) is an experimental infrastructure for fast-starting GPU applications using CRIU (Checkpoint/Restore in User-space). Dynamo Snapshot dramatically reduces cold-start times for large models from minutes to seconds by capturing initialized application state and restoring it on-demand.

## What is Dynamo Snapshot?

Dynamo Snapshot provides:
- **Fast cold starts**: Restore GPU-accelerated applications in seconds instead of minutes
- **CUDA state preservation**: Checkpoint and restore GPU memory and CUDA contexts
- **Kubernetes-native**: Integrates seamlessly with Kubernetes primitives
- **Storage flexibility**: PVC-based storage (S3/OCI planned for future releases)
- **Namespace isolation**: Each namespace gets its own checkpoint infrastructure

## Use Cases

### 1. With NVIDIA Dynamo Platform (Recommended)

Use Dynamo Snapshot as part of the Dynamo platform for automatic checkpoint management:
- Automatic checkpoint creation and lifecycle management
- Seamless integration with DynamoGraphDeployment CRDs
- Built-in autoscaling with fast restore

📖 **[Read the Dynamo Integration Guide →](dynamo.md)**

## Architecture

Dynamo Snapshot consists of two main components:

### 1. Dynamo Snapshot Helm Chart
Deploys the checkpoint/restore infrastructure:
- **DaemonSet**: Runs on GPU nodes to perform CRIU checkpoint operations
- **PVC**: Stores checkpoint data (rootfs diffs, CUDA memory state)
- **RBAC**: Namespace-scoped or cluster-wide permissions
- **Seccomp Profile**: Security policies for CRIU syscalls (needs to be injected into workload pods)

### 2. External Restore via DaemonSet
The DaemonSet performs checkpoint/restore externally using `nsenter` to enter pod namespaces:
- **Checkpoint**: Freezes the running process and dumps state (CPU + GPU) to storage
- **Restore**: Enters a placeholder pod's namespaces and restores the checkpointed process via `nsrestore`

## Quick Start

To install the Dynamo Snapshot DaemonSet in your cluster, run the following:

```bash
helm install snapshot nvidia/snapshot \
  --namespace my-team \
  --create-namespace \
  --set storage.pvc.size=100Gi
```

## Key Features

### ✅ Currently Supported
- ✅ **vLLM and SGLang backends** (TensorRT-LLM planned)
- ✅ **LLM decode/prefill workers only** (multimodal, embedding, and diffusion workers are not supported)
- ✅ Cross-node, single-GPU checkpoints
- ✅ PVC storage backend (RWX for multi-node)
- ✅ CUDA checkpoint/restore
- ✅ PyTorch distributed state (with `GLOO_SOCKET_IFNAME=lo`)
- ✅ Namespace-scoped and cluster-wide RBAC
- ✅ Idempotent checkpoint creation
- ✅ Automatic signal-based checkpoint coordination

### 🚧 Planned Features
- 🚧 TensorRT-LLM backend support
- 🚧 S3/MinIO storage backend
- 🚧 OCI registry storage backend
- 🚧 Multi-GPU checkpoints
- 🚧 Multi-node distributed checkpoints

## Limitations

⚠️ **Important**: Dynamo Snapshot has significant limitations that may impact production readiness:

### Security Considerations
- **🔴 Privileged DaemonSet**: The Dynamo Snapshot DaemonSet runs in privileged mode with `hostPID`, `hostIPC`, and `hostNetwork` to perform CRIU operations. Workload pods do **not** need privileged mode — all CRIU privilege lives in the DaemonSet.
- **Security Impact**: The privileged DaemonSet can:
  - Access all host devices and processes
  - Bypass most security restrictions
  - Potentially compromise node security if exploited

### Technical Limitations
- **vLLM and SGLang backends only**: TensorRT-LLM support is planned.
- **LLM workers only**: Checkpoint/restore supports LLM decode and prefill workers. Specialized workers (multimodal, embedding, diffusion) are not supported.
- **Single-node only**: Checkpoints must be created and restored on the same node
- **Single-GPU only**: Multi-GPU configurations not yet supported
- **Network state limitations**: Active TCP connections are closed during restore (use `tcp-close` CRIU option)
- **Storage**: Only PVC storage is currently implemented (S3/OCI planned)

### Recommendation
Dynamo Snapshot is best suited for:
- ✅ Development and testing environments
- ✅ Research and experimentation
- ✅ Controlled production environments with appropriate security controls
- ❌ Security-sensitive production workloads without proper risk assessment

## Documentation

### Getting Started
- [Dynamo Integration Guide](dynamo.md) - Using Dynamo Snapshot with Dynamo Platform
- [Dynamo Snapshot Helm Chart README](https://github.com/ai-dynamo/dynamo/tree/main/deploy/helm/charts/snapshot/README.md) - Helm chart configuration

### Related Documentation
- [CRIU Documentation](https://criu.org/Main_Page) - Upstream CRIU docs

## Prerequisites

- Kubernetes 1.21+
- GPU nodes with NVIDIA runtime (`nvidia` runtime class)
- containerd runtime (for container inspection; CRIU is bundled in Dynamo Snapshot images)
- RWX storage class (for multi-node deployments)
- **Security clearance for privileged DaemonSet** (the Dynamo Snapshot agent runs privileged with hostPID/hostIPC/hostNetwork)

## Contributing

Dynamo Snapshot is part of the NVIDIA Dynamo project. Contributions are welcome!

## License

Apache License 2.0
