---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Disagg Communication
subtitle: Best practices for prefill/decode worker communication on Kubernetes
---

# Disaggregated Inference Communication Guide

This guide explains how prefill and decode workers communicate in Dynamo's disaggregated inference architecture on Kubernetes. It answers the frequently asked question: **Why can't prefill and decode workers use NVLink to communicate on the same node?**

## Summary

- **NVLink cannot be used between Kubernetes pods** due to process isolation and GPU partitioning
- **RDMA (InfiniBand, RoCE, or AWS EFA) is required** for production disaggregated deployments
- **Without RDMA, expect 200-500x performance degradation** in Time To First Token (TTFT) — observed ~98s TTFT with TCP vs ~200-500ms with RDMA
- **UCX or libfabric** are the communication layers that NIXL uses to transfer KV cache between workers

---

## Architecture Overview

### Communication Stack

<Frame>
  <img src="../assets/img/disagg-comm-stack.svg" alt="Disaggregated inference communication stack showing NIXL, UCX/libfabric, and transport layers" />
</Frame>

### Component Responsibilities

| Component | Role | Location |
|-----------|------|----------|
| **NIXL** | High-level KV cache transfer API | Dynamo runtime library |
| **UCX or libfabric** | Low-level communication framework | System library |
| **Transports** | Physical data movement | Hardware/kernel drivers |

---

## Why NVLink Cannot Be Used Between Pods

### The Fundamental Constraint

NVLink is a **direct GPU-to-GPU interconnect** that operates at the hardware level. It requires:

1. **Same process** - Both GPUs must be visible to a single process so `cudaDeviceEnablePeerAccess()` can be called
2. **Direct memory access** - Process must have permission to access both GPU memory regions
3. **Peer-to-peer mapping** - CUDA runtime must establish memory mappings between GPUs

**Kubernetes pods violate all three requirements:**

<Frame>
  <img src="../assets/img/disagg-nvlink-limitation.svg" alt="Why NVLink cannot work between Kubernetes pods due to process isolation" />
</Frame>

### Technical Explanation

1. **Process Isolation**: Kubernetes pods run in separate Linux namespaces. Even on the same node, Pod A cannot directly access Pod B's memory space.

2. **GPU Partitioning**: The Kubernetes device plugin assigns specific GPUs to each pod via `CUDA_VISIBLE_DEVICES`. Pod A's GPU 0 and Pod B's GPU 0 are physically different devices.

3. **Process/Namespace Isolation**: Each pod runs in a separate process namespace. NVLink peer-to-peer transfers require both GPUs to be within the same process so `cudaDeviceEnablePeerAccess()` can be called.

4. **Memory Registration**: NVLink transfers use `cudaMemcpy` with peer access enabled. This requires calling `cudaDeviceEnablePeerAccess()` - impossible across process boundaries.

### Where NVLink DOES Work

NVLink works **within a pod** for parallelism strategies (TP, EP) where all GPUs are in the same process:

```yaml
# Decode worker with TP=4 uses NVLink between its 4 GPUs
VLLMDecodeWorker:
  resources:
    limits:
      gpu: "4"   # All 4 GPUs visible to single process
  args:
    - --tensor-parallel-size
    - "4"        # NVLink used for TP/EP communication within pod
```

---

## Supported Communication Options

### Transport Comparison

| Transport | Bandwidth | Latency | Same-Node | Cross-Node | GPU Direct |
|-----------|-----------|---------|-----------|------------|------------|
| **NVLink** | 450-900 GB/s | ~µs | ✅ (intra-pod only) | ❌ | ✅ |
| **InfiniBand RDMA** | 20-50 GB/s | ~1 µs | ✅ | ✅ | ✅ (with GPUDirect) |
| **RoCE RDMA** | 10-25 GB/s | ~2 µs | ✅ | ✅ | ✅ (with GPUDirect) |
| **TCP** | 1-3 GB/s | ~50 µs | ✅ | ✅ | ❌ (host staging) |

### Same-Node Communication

When prefill and decode workers are on the **same physical node**:

<Frame>
  <img src="../assets/img/disagg-same-node.svg" alt="Same-node RDMA communication between prefill and decode pods" />
</Frame>

**Options (best to worst):**
1. InfiniBand RDMA with GPUDirect → GPU-to-GPU, bypasses CPU
2. RoCE RDMA with GPUDirect → GPU-to-GPU, bypasses CPU
3. Host-staged RDMA → GPU→CPU→RDMA→CPU→GPU
4. TCP (fallback) → GPU→CPU→TCP→CPU→GPU

**Best Practice**: Use RDMA even for same-node communication. The overhead is minimal and it provides consistent behavior whether pods land on the same or different nodes.

### Cross-Node Communication

When prefill and decode workers are on **different nodes**:

<Frame>
  <img src="../assets/img/disagg-cross-node.svg" alt="Cross-node RDMA communication between prefill and decode pods on separate nodes" />
</Frame>

**Requirements for optimal cross-node performance:**
- RDMA network fabric (InfiniBand, RoCE, or AWS EFA)
- GPUDirect RDMA enabled (GPU memory registered with NIC)
- Proper UCX or libfabric configuration

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

> **⚠️ AWS EFA Exception**: Do NOT use `get_zcopy` on AWS with Ubuntu 24.04 + Kernel ≥6.8. See [AWS EFA Configuration](#aws-efa-configuration) for required settings.

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

NIXL supports **libfabric** as the backend for AWS EFA deployments. This is the **recommended approach** for disaggregated inference on AWS, achieving ~9.6 GB/s KV transfer bandwidth. See the [AWS EFA with NIXL documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nixl.html) for complete setup instructions.

**Requirements:**
- EFA installer version **1.47.0** or later
- Libfabric (installed via EFA installer at `/opt/amazon/efa`)
- GDRCopy for GPU Direct RDMA operations (GPU Operator v26.x installs this automatically)
- EFA-enabled container image (e.g., `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1-efa-amd64`)

**Kernel Compatibility:**

GDRCopy v2.5.1 has a build failure on kernel 6.15+ due to a `vm_flags_set` redefinition. Pin your Ubuntu EKS AMI to kernel 6.14 or earlier until GDRCopy v2.5.2 is available in GPU Operator.

| Kernel Version | GDRCopy v2.5.1 | GDRCopy v2.5.2 |
|----------------|----------------|----------------|
| 6.14 and below | ✅ Works | ✅ Works |
| 6.15+ | ❌ Build fails | ✅ Works |

**Pod Anti-Affinity (Required):**

EFA is designed for **cross-node** communication. Prefill and decode workers must be scheduled on **different nodes** to avoid EAGAIN errors during KV transfer.

```yaml
VllmDecodeWorker:
  extraPodSpec:
    affinity:
      podAntiAffinity:
        requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
                - key: nvidia.com/dynamo-component
                  operator: In
                  values:
                    - VllmPrefillWorker
            topologyKey: kubernetes.io/hostname
```

**EFA Resource Requests:**

Request EFA interfaces in your pod spec. The p5.48xlarge instance has **32 EFA interfaces** (8 network cards × 4 interfaces each) with 3200 Gbps total bandwidth. The number of interfaces to allocate per worker depends on your deployment:

| Deployment | EFA per Worker | Rationale |
|------------|----------------|-----------|
| 1P + 1D per node pair | 4 | Achieved ~9.6 GB/s; leaves 24 interfaces for other pods |
| Multi-worker per node | 2-4 | Balance between workers sharing the node |
| Maximum bandwidth | 8-16 | For very large KV cache transfers or TP>1 |

Example with 4 EFA interfaces (validated configuration):

```yaml
extraPodSpec:
  mainContainer:
    securityContext:
      capabilities:
        add: ["IPC_LOCK"]
    resources:
      limits:
        vpc.amazonaws.com/efa: "4"
      requests:
        vpc.amazonaws.com/efa: "4"
```

> **Note**: NIXL/libfabric automatically stripes traffic across all allocated EFA interfaces. The 4-interface configuration achieved ~9.6 GB/s in testing, which is sufficient for Llama-3.1-8B KV cache transfers at ISL=8000. Increase the count if your workload requires higher bandwidth (e.g., larger models or higher TP).

**Environment Variables:**

```yaml
env:
  - name: NIXL_LOG_LEVEL
    value: "INFO"
  - name: LD_LIBRARY_PATH
    value: "/usr/local/nixl/lib/x86_64-linux-gnu:/opt/amazon/efa/lib64:$(LD_LIBRARY_PATH)"
```

**vLLM Configuration:**

```bash
vllm serve <your-model> \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"cuda","kv_connector_extra_config":{"backends":["LIBFABRIC"]}}'
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `kv_connector` | `NixlConnector` | Enables NIXL for KV-cache transfer |
| `kv_role` | `kv_both` | Symmetric functionality (producer and consumer) |
| `kv_buffer_device` | `cuda` | Uses GPU memory for KV-cache buffer |
| `backends` | `["LIBFABRIC"]` | Routes NIXL traffic over EFA |

**Verification:**

```bash
# Confirm EFA/libfabric installation
fi_info -p efa -t FI_EP_RDM

# Verify GDRCopy device
ls -la /dev/gdrdrv

# Check NIXL initialization in pod logs (should show 32 EFA devices on p5.48xlarge)
kubectl logs <worker-pod> | grep -i "NIXL\|libfabric\|efa"
```

**Expected Log Output:**

```text
NIXL  INFO Loaded backend plugin: LIBFABRIC
NIXL  INFO Found 32 fabric devices
```

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
| `IPC_LOCK` capability | Pin memory for RDMA | Bypasses RLIMIT_MEMLOCK; required for `ibv_reg_mr()` to pin GPU/host buffers |
| `rdma/ib` resources | RDMA NIC access | Provided by RDMA device plugin |
| `sharedMemory.size` | IPC between processes | 16Gi for vLLM, 80Gi for TRT-LLM |

### Infrastructure Prerequisites

1. **RDMA Device Plugin**: Exposes `rdma/ib` or `vpc.amazonaws.com/efa` resources to Kubernetes
   ```bash
   # InfiniBand/RoCE
   kubectl get nodes -o jsonpath='{.items[*].status.allocatable.rdma/ib}'
   # AWS EFA
   kubectl get nodes -o jsonpath='{.items[*].status.allocatable.vpc\.amazonaws\.com/efa}'
   ```

2. **RDMA Network**: One of:
   - InfiniBand or RoCE fabric
   - AWS EFA (Elastic Fabric Adapter)

3. **GPUDirect RDMA** (optional but recommended):
   - NVIDIA driver with GPUDirect enabled
   - `nvidia-peermem` kernel module loaded (InfiniBand/RoCE)
   - GDRCopy installed (AWS EFA with libfabric)

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
```text
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
```text
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
```text
NIXL INFO Backend UCX was instantiated
```

**Bad output** (RDMA not working):
```text
UCX WARN no RDMA transports available
NIXL INFO falling back to TCP transport
```

#### Monitor Transfer Performance

Check Grafana dashboards for:
- **NIXL transfer bandwidth**: Should show GB/s, not MB/s
- **KV cache transfer latency**: Should be under 500ms for typical workloads

**Red flags indicating RDMA issues**:
- Transfer bandwidth under 1 GB/s
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

| Configuration | TTFT Overhead (avg) | KV Transfer BW | Source |
|---------------|---------------------|----------------|--------|
| Aggregated (baseline) | 0 | N/A | No KV transfer needed |
| Disagg + InfiniBand RDMA with GPUDirect | +200-500ms | 20-50 GB/s | *Expected* based on hardware specs |
| Disagg + RoCE RDMA with GPUDirect | +300-800ms | 10-25 GB/s | *Expected* based on hardware specs |
| Disagg + AWS EFA with libfabric + GDRCopy | **+40ms** | **~9.6 GB/s** | *Measured* on AWS p5.48xlarge (Llama-3.1-8B, ISL=8000) |
| Disagg + Host-staged (no GPUDirect) | +1-3s | 1-3 GB/s | *Expected* - CPU bottleneck |
| Disagg + AWS EFA with UCX (without GPUDirect) | ~3x slower than aggregated | ~1 GB/s | *Measured* on AWS p5.48xlarge |
| Disagg + TCP fallback | **+90-100s** | ~100 MB/s | *Measured* ~98s TTFT on AWS p5.48xlarge |

> **Note**: For AWS EFA deployments, use libfabric with GDRCopy to enable GPUDirect RDMA. UCX on AWS EFA does not support GPUDirect on kernel ≥6.8 and results in severely degraded performance. See [AWS EFA Configuration](#aws-efa-configuration) for setup instructions.

### When Disaggregated Makes Sense

**Use disaggregated architecture when:**
- Input sequence length (ISL) ≥ 4000 tokens (5-18% throughput gain)
- You need independent scaling of prefill vs decode capacity
- Prefill and decode have different hardware requirements

**Use aggregated architecture when:**
- Low-latency TTFT is critical
- Input sequences under 2000 tokens (minimal disagg benefit)
- RDMA is not available

### Break-Even Analysis

The KV transfer overhead is amortized across output tokens. **Measured data from Llama-3.1-8B-Instruct** on AWS p5.48xlarge with NIXL+libfabric:

```text
KV Transfer Overhead (TTFT min, unqueued):
- Aggregated:    ~175ms
- Disaggregated: ~215ms
- KV transfer cost: ~40ms

Performance at ISL=8000, concurrency=10:
- ITL improvement: 27% faster per-token generation
- Throughput gain: 18% higher overall throughput
```

**Key Insight**: The KV transfer overhead via libfabric+EFA is only **~40ms**. Combined with 27% faster decode (ITL), disaggregated inference delivers **18% higher throughput** for long-context workloads.

| Metric | Aggregated | Disaggregated | Difference |
|--------|------------|---------------|------------|
| TTFT (min, unqueued) | 175 ms | 215 ms | +40ms |
| ITL (avg) | 17.9 ms | 13.1 ms | **-27%** |
| Throughput (ISL=8000) | 20,131 tok/s | 24,229 tok/s | **+20%** |

**Disagg advantage scales with input length (ISL):**

| ISL | Throughput Δ | ITL Δ | Recommendation |
|-----|--------------|-------|----------------|
| 1000 | ~0% | -4% | Use aggregated |
| 2000 | +3% | -7% | Either works |
| 4000 | +5% | -12% | Disagg preferred |
| 8000 | **+18%** | **-27%** | **Disagg strongly preferred** |

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

**Root Cause**: GPU Direct RDMA not functional on kernel ≥6.8 with EFA when using UCX

**Solution**: Use libfabric instead of UCX for AWS EFA deployments. Libfabric with GDRCopy provides efficient GPU Direct RDMA operations on AWS. See the [AWS EFA Configuration](#aws-efa-configuration) section for setup instructions.

**Alternative options** (if libfabric is not available):
1. Use kernel before 6.8 (Ubuntu 22.04 with kernel 5.15)
2. Accept host-staging performance penalty

### Problem: EFA EAGAIN errors (fi_read still retrying)

**Symptoms**: Decode worker logs show repeated EAGAIN errors:

```text
fi_read still retrying EAGAIN on rail 0
fi_read still retrying EAGAIN on rail 1
...
```

**Root Cause**: Prefill and decode workers are scheduled on the **same node**. AWS EFA is designed for cross-node communication and does not function correctly for intra-node transfers.

**Diagnosis**:

```bash
# Check if workers are on the same node
kubectl get pods -o wide | grep vllm
```

If both prefill and decode workers show the same NODE, this is the problem.

**Solution**: Add pod anti-affinity rules to ensure workers are scheduled on different nodes:

```yaml
VllmDecodeWorker:
  extraPodSpec:
    affinity:
      podAntiAffinity:
        requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
                - key: nvidia.com/dynamo-component
                  operator: In
                  values:
                    - VllmPrefillWorker
            topologyKey: kubernetes.io/hostname
```

> **Note**: Use `nvidia.com/dynamo-component` as the label key, not `app.kubernetes.io/component`. The Dynamo operator uses this label to identify component types.

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
- [ ] NIXL initialized: `kubectl logs <pod> | grep "Backend"`
- [ ] Transfer bandwidth > 1 GB/s (check Grafana metrics)

**For UCX deployments:**
- [ ] UCX sees RDMA devices: `ucx_info -d | grep "Transport: rc"`
- [ ] UCX sees GPU memory: `ucx_info -d | grep "memory types.*cuda"`

**For libfabric deployments (AWS EFA):**
- [ ] EFA devices available: `fi_info -p efa`
- [ ] GDRCopy installed: `ls /dev/gdrdrv`

---

## Related Documentation

- [Disaggregated Serving Architecture](../design-docs/disagg-serving.md)
- [AIConfigurator Deployment Guide](../features/disaggregated-serving/README.md)
- [NIXL Benchmark Deployment](../../deploy/pre-deployment/nixl/README.md)
- [KV Cache Transfer Methods](../backends/trtllm/trtllm-kv-cache-transfer.md)
