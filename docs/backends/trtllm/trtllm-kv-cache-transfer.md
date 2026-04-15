---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: KV Cache Transfer
---

For general TensorRT-LLM features and configuration, see the [Reference Guide](trtllm-reference-guide.md).

---

In disaggregated serving architectures, KV cache must be transferred between prefill and decode workers. TensorRT-LLM supports two methods for this transfer:

## Using NIXL for KV Cache Transfer

Start the disaggregated service: See [Disaggregated Serving](./trtllm-examples.md#disaggregated) to learn how to start the deployment.

## Default Method: NIXL
By default, TensorRT-LLM uses **NIXL** (NVIDIA Inference Xfer Library) with UCX (Unified Communication X) as backend for KV cache transfer between prefill and decode workers. [NIXL](https://github.com/ai-dynamo/nixl) is NVIDIA's high-performance communication library designed for efficient data transfer in distributed GPU environments.

### Specify Backends for NIXL

TensorRT-LLM supports two NIXL communication backends: UCX and LIBFABRIC. By default, UCX is used if no backend is explicitly specified. Dynamo currently supports both backends. For AWS EFA deployments, UCX with SRD transport is the tested and recommended backend (see [AWS EFA](#aws-efa) below).

## Alternative Method: UCX

TensorRT-LLM can also leverage **UCX** (Unified Communication X) directly for KV cache transfer between prefill and decode workers. To enable UCX as the KV cache transfer backend, set `cache_transceiver_config.backend: UCX` in your engine configuration YAML file.

> [!Note]
> The environment variable `TRTLLM_USE_UCX_KVCACHE=1` with `cache_transceiver_config.backend: DEFAULT` does not enable UCX. You must explicitly set `backend: UCX` in the configuration.

## AWS EFA

On AWS, UCX uses the **SRD (Scalable Reliable Datagram)** transport over EFA devices. NIXL discovers EFA `rdmap*` devices automatically through UCX — no NIXL-level configuration changes are needed.

**Image options:**

- **Pre-built EFA image (AMD64 only):** A dedicated EFA image with the EFA SDK baked in is available on NGC. This is recommended for AMD64 instances (e.g. `p5.48xlarge`):

```
nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.0.1-efa-amd64
```

See [Release Artifacts](../../reference/release-artifacts.md) for all available EFA images.

- **Host-mount approach (ARM64 / GB200):** No pre-built EFA ARM64 image is published. Use the standard `tensorrtllm-runtime` image and mount the EFA SDK from the host node. This is what we tested on GB200 NVL72:

```yaml
volumeMounts:
  - name: efa-sdk
    mountPath: /opt/amazon/efa
volumes:
  - name: efa-sdk
    hostPath:
      path: /opt/amazon/efa
```

**EFA resource requests:**

```yaml
resources:
  requests:
    vpc.amazonaws.com/efa: "4"
  limits:
    vpc.amazonaws.com/efa: "4"
```

**Required environment variables for EFA workers** (set on both prefill and decode):

```yaml
env:
  - name: FI_PROVIDER
    value: "efa"
  - name: FI_EFA_USE_DEVICE_RDMA
    value: "1"
  - name: FI_EFA_ENABLE_SHM_TRANSFER
    value: "0"
  - name: LD_LIBRARY_PATH
    value: "/opt/amazon/efa/lib:/usr/local/lib:/usr/lib"
```

> [!IMPORTANT]
> `FI_EFA_ENABLE_SHM_TRANSFER` must be `0`. SHM transfers break NIXL GPU buffer registrations.

**Security context:** AWS EFA currently requires privileged mode:

```yaml
securityContext:
  privileged: true
```

### NIXL Plugin ABI Mismatch on Decode Multinode

When running multinode decode, the decode leader launches workers via `mpirun -> mgmn_worker_node`, which loads TRT-LLM's bundled NIXL rather than the system `nixl_cu13`. The container's default `NIXL_PLUGIN_DIR` points to system plugins that are ABI-incompatible with TRT-LLM's bundled NIXL. Override this **on the decode service only**:

```yaml
env:
  - name: NIXL_PLUGIN_DIR
    value: "/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/libs/nixl/plugins"
```

Do not set this on prefill workers — they use `nixl_cu13` which is compatible with the system plugins.

### ComputeDomain for GB200 NVL72

On GB200 NVL72 racks, NCCL requires a `ComputeDomain` CR for proper cuMem/NVLS initialization. Without it, workers fail with `NCCL error 'unhandled system error'` during model loading.

```yaml
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: my-compute-domain
spec:
  numNodes: 3    # total nodes across prefill + decode
  channel:
    resourceClaimTemplate:
      name: my-compute-domain-channel
```

Both prefill and decode services must include ResourceClaims:

```yaml
resources:
  claims:
    - name: compute-domain-channel
extraPodSpec:
  resourceClaims:
    - name: compute-domain-channel
      resourceClaimTemplateName: my-compute-domain-channel
```

Required NCCL environment variables for GB200:

```yaml
env:
  - name: NCCL_MNNVL_ENABLE
    value: "1"
  - name: NCCL_CUMEM_ENABLE
    value: "1"
  - name: NCCL_NVLS_ENABLE
    value: "1"
  - name: NVIDIA_GDRCOPY
    value: "1"
```

### Verifying EFA is Active

After deployment, confirm NIXL is using SRD over EFA in the worker logs:

```bash
kubectl logs <prefill-pod> | grep -iE "NixlTransfer|srd|rdmap"
```

Expected output:

```
NixlTransferAgent using NIXL backend: UCX
ucp_context_2 self cfg#1 rma_am(srd/rdmap40s0:1) am(srd/rdmap40s0:1 srd/rdmap62s0:1 ...)
NixlTransferAgent mAddress: 100.x.x.x:32939
```

- `srd/rdmap*` confirms SRD transport over EFA devices
- Multiple `rdmap` entries correspond to one EFA device per GPU
