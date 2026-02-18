---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Disaggregated Inference Communication Guide
subtitle: Best practices for prefill/decode worker communication on Kubernetes
---

# Disaggregated Inference Communication Guide

This guide explains how prefill and decode workers communicate in Dynamo's disaggregated inference architecture on Kubernetes. It answers the frequently asked question: **Why can't prefill and decode workers use NVLink to communicate on the same node?**

## Executive Summary

- **NVLink cannot be used between Kubernetes pods** due to process isolation and GPU partitioning
- **RDMA (InfiniBand/RoCE) is required** for production disaggregated deployments
- **Without RDMA, expect 10-40x performance degradation** in Time To First Token (TTFT)
- **UCX is the communication layer** that NIXL uses to transfer KV cache between workers

---

## Architecture Overview

### Communication Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Dynamo Disaggregated Serving                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐              ┌──────────────────┐                 │
│  │  Prefill Worker  │              │  Decode Worker   │                 │
│  │  (Pod A)         │              │  (Pod B)         │                 │
│  │                  │              │                  │                 │
│  │  ┌────────────┐  │              │  ┌────────────┐  │                 │
│  │  │ KV Cache   │  │   Transfer   │  │ KV Cache   │  │                 │
│  │  │ (GPU VRAM) │──┼──────────────┼─▶│ (GPU VRAM) │  │                 │
│  │  └────────────┘  │              │  └────────────┘  │                 │
│  └────────┬─────────┘              └────────┬─────────┘                 │
│           │                                  │                          │
├───────────┼──────────────────────────────────┼──────────────────────────┤
│           │          NIXL Library            │                          │
│           │    (KV Cache Transfer API)       │                          │
├───────────┼──────────────────────────────────┼──────────────────────────┤
│           │                                  │                          │
│           │              UCX                 │                          │
│           │   (Unified Communication X)      │                          │
│           │                                  │                          │
│  ┌────────┴──────────────────────────────────┴────────┐                 │
│  │                  Transport Layer                    │                 │
│  │                                                     │                 │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │                 │
│  │  │   rc_x/dc_x │  │  cuda_copy  │  │    tcp      │ │                 │
│  │  │   (RDMA)    │  │  (staging)  │  │  (fallback) │ │                 │
│  │  │             │  │             │  │             │ │                 │
│  │  │  InfiniBand │  │ GPU↔Host    │  │  Network    │ │                 │
│  │  │  or RoCE    │  │ memory copy │  │  sockets    │ │                 │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │                 │
│  └─────────────────────────────────────────────────────┘                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Role | Location |
|-----------|------|----------|
| **NIXL** | High-level KV cache transfer API | Dynamo runtime library |
| **UCX** | Low-level communication framework | System library |
| **Transports** | Physical data movement | Hardware/kernel drivers |

---

## Why NVLink Cannot Be Used Between Pods

### The Fundamental Constraint

NVLink is a **direct GPU-to-GPU interconnect** that operates at the hardware level. It requires:

1. **Shared CUDA context** - GPUs must be visible to the same process
2. **Direct memory access** - Process must have permission to access both GPU memory regions
3. **Peer-to-peer mapping** - CUDA runtime must establish memory mappings between GPUs

**Kubernetes pods violate all three requirements:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Physical Node (8× H100 GPUs)                      │
│                                                                          │
│  ┌─────────────────────────────┐    ┌─────────────────────────────┐    │
│  │       Pod A (Prefill)       │    │       Pod B (Decode)        │    │
│  │                             │    │                             │    │
│  │  Process Namespace: PID 1   │    │  Process Namespace: PID 1   │    │
│  │  CUDA_VISIBLE_DEVICES: 0,1  │    │  CUDA_VISIBLE_DEVICES: 2-7  │    │
│  │                             │    │                             │    │
│  │  ┌─────┐  ┌─────┐          │    │  ┌─────┐  ┌─────┐  ...     │    │
│  │  │GPU 0│  │GPU 1│          │    │  │GPU 2│  │GPU 3│          │    │
│  │  └─────┘  └─────┘          │    │  └─────┘  └─────┘          │    │
│  │       ↑ NVLink ↑            │    │       ↑ NVLink ↑            │    │
│  │       (works!)              │    │       (works!)              │    │
│  └─────────────────────────────┘    └─────────────────────────────┘    │
│                                                                          │
│            ╳ NO NVLink possible between pods ╳                          │
│                                                                          │
│  Reason: Separate process namespaces, separate CUDA contexts,           │
│          separate GPU device assignments                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Technical Explanation

1. **Process Isolation**: Kubernetes pods run in separate Linux namespaces. Even on the same node, Pod A cannot directly access Pod B's memory space.

2. **GPU Partitioning**: The Kubernetes device plugin assigns specific GPUs to each pod via `CUDA_VISIBLE_DEVICES`. Pod A's GPU 0 and Pod B's GPU 0 are physically different devices.

3. **CUDA Context Separation**: Each pod initializes its own CUDA context. NVLink peer-to-peer transfers require both GPUs to be within the same CUDA context.

4. **Memory Registration**: NVLink transfers use `cudaMemcpy` with peer access enabled. This requires calling `cudaDeviceEnablePeerAccess()` - impossible across process boundaries.

### Where NVLink DOES Work

NVLink works **within a pod** for tensor parallelism:

```yaml
# Decode worker with TP=4 uses NVLink between its 4 GPUs
VLLMDecodeWorker:
  resources:
    limits:
      gpu: "4"   # All 4 GPUs visible to single process
  args:
    - --tensor-parallel-size
    - "4"        # NVLink used for TP communication
```

---

## Supported Communication Options

### Transport Comparison

| Transport | Bandwidth | Latency | Same-Node | Cross-Node | GPU Direct |
|-----------|-----------|---------|-----------|------------|------------|
| **NVLink** | 450-900 GB/s | ~µs | ✅ (intra-pod only) | ❌ | ✅ |
| **InfiniBand RDMA** | 200-400 GB/s | ~1 µs | ✅ | ✅ | ✅ (with GPUDirect) |
| **RoCE RDMA** | 100-200 GB/s | ~2 µs | ✅ | ✅ | ✅ (with GPUDirect) |
| **TCP** | 10-25 GB/s | ~50 µs | ✅ | ✅ | ❌ (host staging) |

### Same-Node Communication

When prefill and decode workers are on the **same physical node**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                             Same Node                                    │
│                                                                          │
│  ┌────────────────────┐                    ┌────────────────────┐       │
│  │   Prefill Pod      │                    │   Decode Pod       │       │
│  │                    │                    │                    │       │
│  │  ┌──────────────┐  │                    │  ┌──────────────┐  │       │
│  │  │ GPU 0 (VRAM) │  │                    │  │ GPU 2 (VRAM) │  │       │
│  │  └──────┬───────┘  │                    │  └──────▲───────┘  │       │
│  └─────────┼──────────┘                    └─────────┼──────────┘       │
│            │                                         │                   │
│            │         RDMA (InfiniBand/RoCE)          │                   │
│            └─────────────────────────────────────────┘                   │
│                                                                          │
│  Options (best to worst):                                                │
│  1. InfiniBand RDMA with GPUDirect    → GPU-to-GPU, bypasses CPU        │
│  2. RoCE RDMA with GPUDirect          → GPU-to-GPU, bypasses CPU        │
│  3. Host-staged RDMA                  → GPU→CPU→RDMA→CPU→GPU            │
│  4. TCP (fallback)                    → GPU→CPU→TCP→CPU→GPU             │
└─────────────────────────────────────────────────────────────────────────┘
```

**Best Practice**: Use RDMA even for same-node communication. The overhead is minimal and it provides consistent behavior whether pods land on the same or different nodes.

### Cross-Node Communication

When prefill and decode workers are on **different nodes**:

```
┌──────────────────────────────┐         ┌──────────────────────────────┐
│           Node 1             │         │           Node 2             │
│                              │         │                              │
│  ┌────────────────────┐      │         │      ┌────────────────────┐  │
│  │   Prefill Pod      │      │         │      │   Decode Pod       │  │
│  │  ┌──────────────┐  │      │         │      │  ┌──────────────┐  │  │
│  │  │ GPU (VRAM)   │  │      │         │      │  │ GPU (VRAM)   │  │  │
│  │  └──────┬───────┘  │      │         │      │  └──────▲───────┘  │  │
│  └─────────┼──────────┘      │         │      └─────────┼──────────┘  │
│            │                 │         │                │             │
│  ┌─────────▼─────────┐       │         │       ┌────────┴────────┐   │
│  │    RDMA NIC       │       │         │       │    RDMA NIC     │   │
│  │  (InfiniBand/     │◄──────┼─────────┼──────▶│  (InfiniBand/   │   │
│  │   RoCE)           │       │ Network │       │   RoCE)         │   │
│  └───────────────────┘       │         │       └─────────────────┘   │
└──────────────────────────────┘         └──────────────────────────────┘
```

**Requirements for optimal cross-node performance:**
- InfiniBand or RoCE network fabric
- GPUDirect RDMA enabled (GPU memory registered with NIC)
- Proper UCX configuration

---

## UCX Configuration Reference

### Environment Variables

UCX behavior is controlled through environment variables. Set these on both prefill and decode worker pods.

#### Core Transport Selection

```yaml
env:
  - name: UCX_TLS
    value: "rc_x,rc,dc_x,dc,cuda_copy,cuda_ipc"
```

| Transport | Description | When to Use |
|-----------|-------------|-------------|
| `rc_x` | Reliable Connection (accelerated) | Primary RDMA transport |
| `rc` | Reliable Connection (standard) | Fallback RDMA |
| `dc_x` | Dynamically Connected (accelerated) | Scalable RDMA (many endpoints) |
| `dc` | Dynamically Connected (standard) | Fallback scalable RDMA |
| `cuda_copy` | GPU↔Host memory staging | Required for GPU buffers |
| `cuda_ipc` | CUDA IPC (same-node, same-pod) | Intra-pod GPU transfers |
| `tcp` | TCP sockets | Fallback when RDMA unavailable |
| `srd` | Scalable Reliable Datagram (AWS EFA) | AWS-specific (provided by EFA, not core UCX) |

**Excluding transports**: Use `^` prefix to exclude (e.g., `UCX_TLS=^mm` excludes memory mapping).

**Note**: When specifying `UCX_TLS` explicitly with GPU memory, you must include `cuda_copy` or `cuda_ipc` for UCX to recognize GPU buffers.

#### Rendezvous Protocol Settings

```yaml
env:
  - name: UCX_RNDV_SCHEME
    value: "get_zcopy"
  - name: UCX_RNDV_THRESH
    value: "0"
```

| Variable | Value | Description |
|----------|-------|-------------|
| `UCX_RNDV_SCHEME` | `get_zcopy` | Zero-copy RDMA GET (receiver pulls data) |
| `UCX_RNDV_SCHEME` | `put_zcopy` | Zero-copy RDMA PUT (sender pushes data) |
| `UCX_RNDV_SCHEME` | `auto` | Let UCX choose based on message size |
| `UCX_RNDV_THRESH` | `0` | Use rendezvous for all message sizes |
| `UCX_RNDV_THRESH` | `8192` | Use rendezvous for messages ≥8KB |
| `UCX_RNDV_THRESH` | `auto` | Let UCX calculate optimal threshold |

**Recommendation**: Use `get_zcopy` with threshold `0` for KV cache transfers (always large).

#### Memory Registration

```yaml
env:
  - name: UCX_IB_REG_METHODS
    value: "odp,rcache"
```

| Method | Description |
|--------|-------------|
| `odp` | On-Demand Paging (dynamic registration) |
| `rcache` | Registration cache (reuse registrations) |
| `direct` | Direct registration (each transfer) |

#### Debugging and Diagnostics

```yaml
env:
  - name: UCX_LOG_LEVEL
    value: "info"        # Options: fatal, error, warn, info, debug, trace, data, func
  - name: UCX_LOG_FILE
    value: "/tmp/ucx.log" # Optional: log to file instead of stdout
```

**Note**: UCX statistics (`UCX_STATS_DEST`, `UCX_STATS_TRIGGER`) require UCX compiled with `--enable-stats` flag, which is not enabled in default builds.

### Complete Production Configuration

```yaml
env:
  # Transport selection - RDMA with GPU support
  - name: UCX_TLS
    value: "rc_x,rc,dc_x,dc,cuda_copy,cuda_ipc"

  # Rendezvous for large transfers
  - name: UCX_RNDV_SCHEME
    value: "get_zcopy"
  - name: UCX_RNDV_THRESH
    value: "0"

  # Memory registration optimization
  - name: UCX_IB_REG_METHODS
    value: "odp,rcache"

  # RDMA settings
  - name: UCX_IB_GID_INDEX
    value: "3"           # RoCE v2 GID index (cluster-specific)
```

### AWS EFA Configuration

> **Important: NIXL is migrating from UCX to libfabric for AWS**
>
> The Dynamo team is transitioning NIXL to use **libfabric** instead of UCX for AWS EFA deployments. This change is driven by:
> - **Better topology awareness**: libfabric provides hierarchical topology awareness similar to NCCL
> - **Native EFA support**: libfabric is the recommended communication layer for AWS EFA
>
> **Current status**: UCX over EFA works but is not recommended for production. Published AWS examples are functional but not performant. Check with the Dynamo team for libfabric availability timeline.

For AWS with Elastic Fabric Adapter (interim UCX configuration):

```yaml
env:
  - name: UCX_TLS
    value: "srd,cuda_copy,tcp"    # SRD is EFA's RDMA transport
  - name: UCX_RNDV_SCHEME
    value: "auto"                  # Let UCX decide
  - name: UCX_RNDV_THRESH
    value: "auto"                  # Let UCX calculate threshold
```

**Known Limitations**:
- GPU Direct RDMA is non-functional on AWS EFA with Ubuntu 24.04 + kernel ≥6.8
- Expect 3x performance degradation compared to InfiniBand (host-staged transfers)
- For optimal disaggregated performance on AWS, prefer InfiniBand or RoCE environments until libfabric support is available

---

## Deployment Configuration

### Kubernetes Resource Requirements

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
spec:
  services:
    VLLMPrefillWorker:
      resources:
        limits:
          gpu: "2"
      extraPodSpec:
        mainContainer:
          securityContext:
            capabilities:
              add: ["IPC_LOCK"]      # Required for RDMA memory pinning
          resources:
            limits:
              rdma/ib: "2"           # RDMA resources (match TP size)
            requests:
              rdma/ib: "2"
```

### Required Capabilities and Resources

| Setting | Purpose | Notes |
|---------|---------|-------|
| `IPC_LOCK` capability | Pin memory for RDMA | Required by UCX |
| `rdma/ib` resources | RDMA NIC access | Provided by RDMA device plugin |
| `sharedMemory.size` | IPC between processes | 16Gi for vLLM, 80Gi for TRT-LLM |

### Infrastructure Prerequisites

1. **RDMA Device Plugin**: Exposes `rdma/ib` resources to Kubernetes
   ```bash
   kubectl get nodes -o jsonpath='{.items[*].status.allocatable.rdma/ib}'
   ```

2. **InfiniBand/RoCE Network**: Physical RDMA fabric connecting nodes

3. **GPUDirect RDMA** (optional but recommended):
   - NVIDIA driver with GPUDirect enabled
   - `nvidia-peermem` kernel module loaded
   - NIC firmware supporting GPUDirect

---

## Diagnostics and Performance Validation

### Pre-Deployment Validation

#### 1. Verify RDMA Availability

```bash
# Check RDMA devices on node
kubectl debug node/<node-name> -it --image=ubuntu:22.04 -- bash
ibv_devinfo
```

Expected output shows InfiniBand or RoCE devices:
```
hca_id: mlx5_0
        transport:                      InfiniBand (0)
        fw_ver:                         28.35.2000
        ...
```

#### 2. Check UCX Transport Capabilities

```bash
# Inside a Dynamo worker pod
ucx_info -d
```

Look for GPU memory support:
```
# Memory domain: mlx5_0
#     Component: ib
#     memory types: host (access,reg,cache), cuda (access,reg,cache)
#                                            ^^^^ GPU memory supported
```

**If you only see `host`**: GPUDirect RDMA is not working. KV transfers will use host staging.

#### 3. Test UCX Performance

```bash
# Server (on decode worker pod)
ucx_perftest -t tag_bw -n 100 -s 134217728

# Client (on prefill worker pod)
ucx_perftest <server-ip> -t tag_bw -n 100 -s 134217728
```

**Expected bandwidth**:
- InfiniBand HDR: 20-25 GB/s per port
- RoCE 100GbE: 10-12 GB/s
- TCP fallback: 1-2 GB/s

### NIXL Benchmark Tool

Deploy the NIXL benchmark to validate end-to-end KV transfer performance:

```bash
cd deploy/pre-deployment/nixl
./build_and_deploy.sh
```

This deploys a benchmark that measures actual GPU-to-GPU transfer rates through NIXL.

### Runtime Diagnostics

#### Verify NIXL Backend Initialization

```bash
kubectl logs <worker-pod> | grep -i "NIXL\|UCX"
```

**Good output**:
```
NIXL INFO Backend UCX was instantiated
```

**Bad output** (RDMA not working):
```
UCX WARN no RDMA transports available
NIXL INFO falling back to TCP transport
```

#### Monitor Transfer Performance

Check Grafana dashboards for:
- **NIXL transfer bandwidth**: Should show GB/s, not MB/s
- **KV cache transfer latency**: Should be <500ms for typical workloads

**Red flags indicating RDMA issues**:
- Transfer bandwidth < 1 GB/s
- TTFT > 10 seconds
- `Unsupported operation` errors in logs

### Common Diagnostic Commands

```bash
# Check UCX transport selection
kubectl exec <pod> -- env | grep UCX

# Verify RDMA device visibility
kubectl exec <pod> -- ls /dev/infiniband/

# Check GPUDirect RDMA status (on node)
kubectl debug node/<node> -it --image=ubuntu:22.04 -- \
  nsenter -t 1 -m -u -n -p -- dmesg | grep -i "nvidia\|peermem\|gdr"

# Test basic connectivity between pods
kubectl exec <prefill-pod> -- ping -c 3 <decode-pod-ip>
```

---

## Performance Expectations

### KV Cache Transfer Overhead

| Configuration | TTFT Overhead | Notes |
|---------------|---------------|-------|
| Aggregated (baseline) | 0 | No KV transfer needed |
| Disagg + InfiniBand RDMA | +200-500ms | Optimal disaggregated performance |
| Disagg + RoCE RDMA | +300-800ms | Slightly higher than InfiniBand |
| Disagg + Host-staged RDMA | +1-3s | No GPUDirect, CPU bottleneck |
| Disagg + TCP only | +5-15s | **40x degradation** |

### When Disaggregated Makes Sense

**Use disaggregated architecture when:**
- Output sequence length (OSL) > 1000 tokens (overhead amortized)
- You need independent scaling of prefill vs decode capacity
- Prefill and decode have different hardware requirements

**Use aggregated architecture when:**
- Low-latency TTFT is critical
- Short outputs (OSL < 500 tokens)
- RDMA is not available

### Break-Even Analysis

The KV transfer overhead is amortized across output tokens:

```
Total Latency = TTFT + (OSL × ITL)

Example (ISL=4000):
- Aggregated:    218ms + (OSL × 8.0ms)
- Disaggregated: 2400ms + (OSL × 7.8ms)

Break-even: 2400 - 218 = 2182ms overhead
            2182ms / (8.0 - 7.8)ms per token = 10,910 tokens

At OSL=2000: Disagg is 1.1x slower (acceptable)
At OSL=100:  Disagg is 3.7x slower (not recommended)
```

---

## Troubleshooting Guide

### Problem: TTFT is 10+ seconds

**Symptoms**: TTFT degrades from expected 200-500ms to 10+ seconds

**Root Cause**: RDMA not active, falling back to TCP

**Diagnosis**:
```bash
kubectl logs <worker-pod> | grep -i "transport\|UCX\|TCP"
```

**Solutions**:
1. Verify RDMA device plugin is installed
2. Add `rdma/ib` resource requests to pod spec
3. Add `IPC_LOCK` capability
4. Set UCX environment variables

### Problem: "Unsupported operation" errors

**Symptoms**: Logs show `Unexpected UCX error: Unsupported operation`

**Root Cause**: UCX attempting GPU RDMA on hardware that doesn't support it

**Solutions**:
1. Check if GPUDirect RDMA is enabled: `ucx_info -d | grep cuda`
2. If not supported, set `UCX_RNDV_THRESH=inf` to disable GPU RDMA
3. Verify `nvidia-peermem` module is loaded

### Problem: AWS EFA not using GPU Direct

**Symptoms**: 3x performance degradation on AWS despite EFA configured

**Root Cause**: GPU Direct RDMA not functional on kernel ≥6.8 with EFA

**Current Status**: This is a known limitation. Options:
1. Use kernel < 6.8 (Ubuntu 22.04 with kernel 5.15)
2. Accept host-staging performance penalty
3. Wait for AWS to update EFA DMA-BUF support

### Problem: Intermittent transfer failures

**Symptoms**: Sporadic `getXferStatus: backend 'UCX' returned error status`

**Diagnosis**:
```bash
# Enable UCX debug logging
kubectl set env deployment/<worker> UCX_LOG_LEVEL=debug
kubectl logs <worker-pod> | grep -i error
```

**Common causes**:
- Network congestion or packet loss
- Mismatched UCX versions between pods
- RDMA resource exhaustion

---

## Quick Reference

### Minimum Viable RDMA Configuration

```yaml
env:
  - name: UCX_TLS
    value: "rc_x,rc,dc_x,dc,cuda_copy,cuda_ipc"
  - name: UCX_RNDV_SCHEME
    value: "get_zcopy"
  - name: UCX_RNDV_THRESH
    value: "0"

securityContext:
  capabilities:
    add: ["IPC_LOCK"]

resources:
  limits:
    rdma/ib: "2"
  requests:
    rdma/ib: "2"
```

### Diagnostic Checklist

- [ ] `rdma/ib` resources visible: `kubectl get nodes -o jsonpath='{..allocatable.rdma/ib}'`
- [ ] UCX sees RDMA devices: `ucx_info -d | grep "Transport: rc"`
- [ ] UCX sees GPU memory: `ucx_info -d | grep "memory types.*cuda"`
- [ ] NIXL initialized with UCX: `kubectl logs <pod> | grep "Backend UCX"`
- [ ] Transfer bandwidth > 1 GB/s (check Grafana metrics)

---

## Related Documentation

- [Disaggregated Serving Architecture](../design-docs/disagg-serving.md)
- [AIConfigurator Deployment Guide](../features/disaggregated-serving/README.md)
- [NIXL Benchmark Deployment](../../deploy/pre-deployment/nixl/README.md)
- [KV Cache Transfer Methods](../backends/trtllm/kv-cache-transfer.md)
