---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

> ⚠️ **Experimental Feature**: ChReK is currently in **beta/preview**. It requires privileged mode for restore operations, which may not be suitable for all production environments. See [Limitations](#limitations) for details.

**ChReK** (Checkpoint/Restore in Kubernetes) is an experimental infrastructure for fast-starting GPU applications using CRIU (Checkpoint/Restore in User-space). ChReK dramatically reduces cold-start times for large models from minutes to seconds by capturing initialized application state and restoring it on-demand.

## What is ChReK?

ChReK provides:
- **Fast cold starts**: Restore GPU-accelerated applications in seconds instead of minutes
- **CUDA state preservation**: Checkpoint and restore GPU memory and CUDA contexts
- **Kubernetes-native**: Integrates seamlessly with Kubernetes primitives
- **Storage flexibility**: PVC-based storage (S3/OCI planned for future releases)
- **Namespace isolation**: Each namespace gets its own checkpoint infrastructure

## Use Cases

### 1. With NVIDIA Dynamo Platform (Recommended)

Use ChReK as part of the Dynamo platform for automatic checkpoint management:
- Automatic checkpoint creation and lifecycle management
- Seamless integration with DynamoGraphDeployment CRDs
- Built-in autoscaling with fast restore

📖 **[Read the Dynamo Integration Guide →](dynamo.md)**

### 2. Standalone (Without Dynamo)

Use ChReK independently in your own Kubernetes applications:
- Manual checkpoint job creation
- Build your own restore-enabled container images
- Full control over checkpoint lifecycle

📖 **[Read the Standalone Usage Guide →](standalone.md)**

## Architecture

ChReK consists of two main components:

### 1. ChReK Helm Chart
Deploys the checkpoint/restore infrastructure:
- **DaemonSet**: Runs on GPU nodes to perform CRIU checkpoint operations
- **PVC**: Stores checkpoint data (rootfs diffs, CUDA memory state)
- **RBAC**: Namespace-scoped or cluster-wide permissions
- **Seccomp Profile**: Security policies for CRIU syscalls

### 2. Smart Entrypoint
A wrapper script that intelligently decides between:
- **Cold start**: Normal application startup (when no checkpoint exists)
- **Restore**: CRIU restore from checkpoint (when checkpoint available)

## Quick Start

### Install ChReK Infrastructure

```bash
helm install chrek nvidia/chrek \
  --namespace my-team \
  --create-namespace \
  --set storage.pvc.size=100Gi
```

### Choose Your Integration Path

- **Using Dynamo Platform?** → Follow the [Dynamo Integration Guide](dynamo.md)
- **Using standalone?** → Follow the [Standalone Usage Guide](standalone.md)

## Key Features

### ✅ Currently Supported
- ✅ **vLLM backend only** (SGLang and TensorRT-LLM planned)
- ✅ Single-node, single-GPU checkpoints
- ✅ PVC storage backend (RWX for multi-node)
- ✅ CUDA checkpoint/restore
- ✅ PyTorch distributed state (with `GLOO_SOCKET_IFNAME=lo`)
- ✅ Namespace-scoped and cluster-wide RBAC
- ✅ Idempotent checkpoint creation
- ✅ Automatic signal-based checkpoint coordination

### 🚧 Planned Features
- 🚧 SGLang backend support
- 🚧 TensorRT-LLM backend support
- 🚧 S3/MinIO storage backend
- 🚧 OCI registry storage backend
- 🚧 Multi-GPU checkpoints
- 🚧 Multi-node distributed checkpoints

## Limitations

⚠️ **Important**: ChReK has significant limitations that may impact production readiness:

### Security Considerations
- **🔴 Privileged mode required**: Restore pods **must run in privileged mode** for CRIU to function. This grants containers elevated host access and may violate security policies in many production environments.
- **Security Impact**: Privileged containers can:
  - Access all host devices
  - Bypass most security restrictions
  - Potentially compromise node security if the container is exploited

### Technical Limitations
- **vLLM backend only**: Currently only the vLLM backend supports checkpoint/restore. SGLang and TensorRT-LLM support is planned.
- **Single-node only**: Checkpoints must be created and restored on the same node
- **Single-GPU only**: Multi-GPU configurations not yet supported
- **Network state limitations**: Active TCP connections are closed during restore (use `tcp-close` CRIU option)
- **Storage**: Only PVC storage is currently implemented (S3/OCI planned)

### Recommendation
ChReK is best suited for:
- ✅ Development and testing environments
- ✅ Research and experimentation
- ✅ Controlled production environments with appropriate security controls
- ❌ Security-sensitive production workloads without proper risk assessment

## Documentation

### Getting Started
- [Dynamo Integration Guide](dynamo.md) - Using ChReK with Dynamo Platform
- [Standalone Usage Guide](standalone.md) - Using ChReK independently
- [ChReK Helm Chart README](https://github.com/ai-dynamo/dynamo/tree/main/deploy/helm/charts/chrek/README.md) - Helm chart configuration

### Related Documentation
- [CRIU Documentation](https://criu.org/Main_Page) - Upstream CRIU docs

## Prerequisites

- Kubernetes 1.21+
- GPU nodes with NVIDIA runtime (`nvidia` runtime class)
- CRIU support in container runtime (containerd with CRIU plugin)
- RWX storage class (for multi-node deployments)
- **Security clearance for privileged pods** (required for restore operations)

## Troubleshooting

### Common Issues

**DaemonSet not starting?**
- Check GPU node labels: `kubectl get nodes -l nvidia.com/gpu.present=true`
- Verify NVIDIA runtime is available

**Checkpoint fails?**
- Check DaemonSet logs: `kubectl logs -l app.kubernetes.io/name=chrek -n <namespace>`
- Ensure application properly signals readiness
- Verify CRIU is installed in the runtime

**Restore fails?**
- Ensure restore pod uses the same volumes as checkpoint job
- Verify `hostIPC: true` is set (required for CUDA)
- Check for `PSM3_DISABLED=1` and `GLOO_SOCKET_IFNAME=lo` environment variables

For detailed troubleshooting, see:
- [Dynamo Integration Guide - Troubleshooting](dynamo.md#troubleshooting)
- [Standalone Guide - Troubleshooting](standalone.md#troubleshooting)

## Contributing

ChReK is part of the NVIDIA Dynamo project. Contributions are welcome!

## License

Apache License 2.0
