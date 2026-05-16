---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: EFA (RDMA over AWS Fabric) on EKS
---

# EFA (RDMA over AWS Fabric) on EKS

This guide covers setting up RDMA over AWS Elastic Fabric Adapter (EFA) on EKS for high-performance disaggregated inference with Dynamo. EFA is the only RDMA fabric available on AWS — InfiniBand and RoCE are not offered. With EFA, Dynamo's prefill and decode workers transfer KV cache directly between GPUs across nodes via GPU-Direct RDMA, bypassing CPU and TCP/IP stacks.

Without RDMA, disaggregated inference falls back to TCP with severe performance degradation (~98s TTFT vs ~1s with EFA on Llama-3.1-8B at ISL 8000). See the [Disaggregated Communication Guide](../../disagg-communication-guide.md) for the transport-layer fundamentals.

## Prerequisites

**Recommended GPU EC2 instance types with EFA:**


| Instance family                | GPU                                                     | Aggregate EFA bandwidth                                 | Arch              |
| ------------------------------ | ------------------------------------------------------- | ------------------------------------------------------- | ----------------- |
| `p5.48xlarge` / `p5e.48xlarge` | 8× H100 / H200                                          | 3.2 Tbps                                                | x86_64            |
| `p5en.48xlarge`                | 8× H200                                                 | 3.2 Tbps                                                | x86_64            |
| `p6-b200.48xlarge`             | 8× B200                                                 | 3.2 Tbps                                                | x86_64            |
| P6e-GB200 UltraServer          | GB200 (topology-dependent, up to 72 GPUs / UltraServer) | 400 GB/s EFAv4 per GPU; up to 28.8 Tbps per UltraServer | **arm64 (Grace)** |


This table is not an exhaustive list of all AWS instance types that support EFA. It lists the GPU families most relevant to Dynamo disaggregated inference.

**Cluster setup:**

- **GPU-Direct RDMA enabled on the host** — either kernel ≥ 5.12 (DMA-BUF path; default on current AWS EKS AMIs, typically 6.14+) **or** an older kernel with the `nvidia-peermem` / AWS `efa_nv_peermem` module loaded (legacy peer-memory path; see [Step 2](#step-2-verify-host-kernel-modules) for how to install it).
- **EFA-enabled security group** — VPC security groups must allow all traffic between EFA-attached ENIs. The standard recommendation is a self-referencing security group rule that allows all protocols within the group. See [AWS EFA security group setup](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html#efa-start-security).
- **EKS node groups created with EFA support** — when using `eksctl`, set `efaEnabled: true` on the GPU node group. This attaches the appropriate number of EFA ENIs per instance type.

## Overview

EFA setup involves three pieces:

1. **AWS EFA Kubernetes device plugin** — exposes EFA NICs as the `vpc.amazonaws.com/efa` extended resource (host-level setup, [Step 1](#step-1-install-the-aws-efa-kubernetes-device-plugin)). On modern kernels (≥ 5.12) the DMA-BUF path is used and `efa_nv_peermem` is not required; older kernels need it loaded ([Step 2](#step-2-verify-host-kernel-modules)).
2. **Container image** with libfabric + aws-ofi-nccl + Dynamo ([Step 3](#step-3-build-a-dynamo-efa-image)).
3. **Workload spec** that selects the LIBFABRIC NIXL backend, requests EFA resources, and runs privileged ([Step 4](#step-4-configure-nixl-backend), [Step 5](#step-5-pod-resource-requests)).

## Step 1: Install the AWS EFA Kubernetes Device Plugin

The AWS EFA Kubernetes Device Plugin exposes each node's EFA endpoints as the `vpc.amazonaws.com/efa` extended resource so pods can request them. AWS publishes two install paths — pick one:

**Helm (recommended, from the official `aws/eks-charts` repo):**

```bash
helm repo add eks https://aws.github.io/eks-charts
helm repo update
helm install aws-efa-k8s-device-plugin \
  --namespace kube-system \
  eks/aws-efa-k8s-device-plugin
```

**Or raw manifest (from [aws-samples/aws-efa-eks](https://github.com/aws-samples/aws-efa-eks)):**

```bash
kubectl apply -f https://raw.githubusercontent.com/aws-samples/aws-efa-eks/main/manifest/efa-k8s-device-plugin.yml
```

Wait for the device plugin pods to start on every EFA-capable node:

```bash
kubectl get pods -n kube-system -l name=aws-efa-k8s-device-plugin-daemonset -w
```

Verify EFA resources are advertised by each GPU node:

```bash
kubectl get nodes -o json | jq '.items[] | select(.status.allocatable["vpc.amazonaws.com/efa"] != null) | {name: .metadata.name, efa: .status.allocatable["vpc.amazonaws.com/efa"], gpu: .status.allocatable["nvidia.com/gpu"]}'
```

Each EFA-capable node should report a non-zero `vpc.amazonaws.com/efa` count (e.g., `32` on `p5.48xlarge`, reflecting that instance's EFA endpoint count). The exact count depends on instance type and how the node group's ENIs were configured at launch.

## Step 2: Verify Host Kernel Modules

Modern AWS GPU AMIs (Amazon Linux 2023, Ubuntu 22.04+, kernel ≥ 5.12) use **DMA-BUF** for GPU-Direct RDMA and **do not require** `nvidia-peermem` or `efa_nv_peermem`. The default AMIs for p5/p5e/p5en/p6-b200/GB200 ship with kernels in the 6.x line where DMA-BUF is the active path.

To confirm:

```bash
# On a GPU node (via kubectl debug or SSH):
uname -r
# Expected: 6.x kernel (e.g., 6.14.0-1018-aws)

lsmod | grep -E "^efa|nvidia"
# Expected: efa, nvidia, nvidia_modeset, nvidia_uvm, gdrdrv loaded
# Note: nvidia-peermem / efa_nv_peermem NOT loaded is normal on modern kernels

cat /sys/module/efa/version
# Expected: 3.0.0g or newer
```

If you are on an older kernel (< 5.12) and the host doesn't already have `efa_nv_peermem` loaded, the simplest path is to switch to an AMI that includes EFA host-level components — the EKS-optimized AL2023 NVIDIA AMI and all Bottlerocket AMIs include them. Otherwise, run [`aws-efa-installer`](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html#efa-start-enable) on the host (via a privileged DaemonSet or baked into a custom AMI). See [AWS — Manage EFA devices on Amazon EKS](https://docs.aws.amazon.com/eks/latest/userguide/device-management-efa.html) for the full picture.

## Step 3: Build a Dynamo EFA Image

Dynamo's image build is two steps: `container/render.py` writes a Dockerfile for the chosen framework + target, then `docker build` consumes it. Passing `--make-efa` to `render.py` appends the AWS EFA installer stage from [`container/templates/aws.Dockerfile`](../../../../container/templates/aws.Dockerfile), which defines a stage named `aws` on top of `runtime`. **You must pass `--target aws` to `docker build`** — without it, `docker build` stops at the `runtime` stage and you get an image without EFA. See [`container/README.md`](../../../../container/README.md) for the full build workflow.

```bash
# vLLM EFA image (amd64)
container/render.py --framework=vllm --target=runtime --platform=linux/amd64 \
    --make-efa --output-short-filename
docker build --target aws -t dynamo:latest-vllm-runtime-efa \
    -f container/rendered.Dockerfile .

# SGLang EFA image (amd64 or arm64)
container/render.py --framework=sglang --target=runtime --platform=linux/amd64 \
    --make-efa --output-short-filename
docker build --target aws -t dynamo:latest-sglang-runtime-efa \
    -f container/rendered.Dockerfile .

container/render.py --framework=sglang --target=runtime --platform=linux/arm64 \
    --make-efa --output-short-filename
docker buildx build --platform=linux/arm64 --target aws \
    -t dynamo:latest-sglang-runtime-efa-arm64 -f container/rendered.Dockerfile .

# TRT-LLM EFA image (amd64 only — TRT-LLM base image has no arm64 variant)
container/render.py --framework=trtllm --target=runtime --platform=linux/amd64 \
    --cuda-version=13.1 --make-efa --output-short-filename
docker build --target aws -t dynamo:latest-trtllm-runtime-efa \
    -f container/rendered.Dockerfile .
```

`--output-short-filename` writes to `container/rendered.Dockerfile`; omit it to get the long auto-generated filename (e.g., `vllm-runtime-cuda12.9-amd64-rendered.Dockerfile`) — useful when keeping several rendered Dockerfiles side by side.

> [!IMPORTANT]
> See [Known Issues](#known-issues) below for one case where the default-built image does **not** produce a working EFA deployment out of the box (GB200 / arm64 64K-page kernels). The symptom looks like a working setup but fails at startup during NIXL memory registration.

## Step 4: Configure NIXL Backend

NIXL is the high-level KV transfer API and supports multiple backends. **For EFA, the LIBFABRIC backend must be selected.** UCX is NIXL's default backend, and while it has CUDA-IPC / RDMA transports available in the image, in standard pod-to-pod EFA configurations it lands on a slow transport (effectively TCP-speed at ~1–3 GB/s) instead of EFA's line rate. Empirically, LIBFABRIC is the only backend that reaches full EFA bandwidth on AWS.

Each framework selects the backend differently:


| Framework       | How to select LIBFABRIC                                                                                                                       | Default if unset   |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| **SGLang**      | `SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC` env var                                                                                        | UCX → TCP fallback |
| **vLLM**        | `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_connector_extra_config":{"backends":["LIBFABRIC"]}}'` CLI flag | UCX → TCP fallback |
| **TRT-LLM**     | `TRTLLM_NIXL_KVCACHE_BACKEND=LIBFABRIC` env var                                                                                               | UCX → TCP fallback |
| **KVBM (Rust)** | `DYN_KVBM_NIXL_BACKEND_LIBFABRIC=true` env var                                                                                                | UCX → TCP fallback |


This is a silent-failure path — getting it wrong manifests as ~100 s TTFT instead of a clear error. Always [verify at startup](#verification) that LIBFABRIC is active.

### Required EFA environment variables

In addition to backend selection, set these on every worker pod:

```yaml
env:
  - { name: FI_PROVIDER,                value: efa }
  - { name: FI_EFA_USE_DEVICE_RDMA,     value: "1" }
  - { name: FI_EFA_ENABLE_SHM_TRANSFER, value: "0" }
  - { name: FI_EFA_ENABLE_SHM,          value: "0" }
  # Place Amazon EFA libs first in LD_LIBRARY_PATH
  - name: LD_LIBRARY_PATH
    value: "/opt/amazon/efa/lib:/opt/amazon/efa/lib64:/opt/aws-ofi-nccl/lib:${LD_LIBRARY_PATH}"
```

### Recommended EFA performance tuning

```yaml
env:
  - { name: FI_EFA_FORK_SAFE,           value: "0" }
  - { name: FI_EFA_USE_HUGE_PAGE,       value: "1" }
  - { name: FI_EFA_MR_MAX_CACHED_COUNT, value: "524288" }
  - { name: FI_EFA_MR_MAX_CACHED_SIZE,  value: "0" }
```

When using `FI_EFA_USE_HUGE_PAGE=1`, also add `hugepages-2Mi: 5120Mi` to the pod resource limits.


## Step 5: Pod Resource Requests

Dynamo pods that use EFA must request the resource and run privileged:

```yaml
resources:
  limits:
    nvidia.com/gpu:           "4"          # or your TP
    vpc.amazonaws.com/efa:    "4"          # number of EFA NICs to allocate
    hugepages-2Mi:            5120Mi        # if FI_EFA_USE_HUGE_PAGE=1
securityContext:
  privileged: true                          # REQUIRED — IPC_LOCK alone is insufficient
  capabilities:
    add: [IPC_LOCK]
hostIPC: true                               # required by some EFA setups
volumeMounts:
  - { name: shm, mountPath: /dev/shm }
```

```yaml
volumes:
  - name: shm
    emptyDir: { medium: Memory, sizeLimit: 80Gi }
```

> [!IMPORTANT]
> `privileged: true` is required for NIXL to register CUDA VRAM with the EFA NIC via `fi_mr_reg`. `IPC_LOCK` alone is insufficient.

## Known Issues

One issue currently affects default-built Dynamo EFA images.

### Issue 1: libfabric on GB200 fails `fi_mr_reg` on CUDA VRAM

**Known affected platforms:** GB200.

**Symptom:** Worker pod fails at startup with `fi_mr_reg` returning EFAULT during NIXL initialization. NIXL VRAM registration fails; depending on the framework, the worker either crashes or silently falls back to TCP.

**Root cause:** The libfabric version (versions lower than 2.5.x) bundled with the EFA installer (up to currently latest 1.48.0) lacks a CUDA branch in the dmabuf-eligibility check in `prov/efa/src/efa_mr.c`. On x86_64 hosts the legacy `ibv_reg_mr` path handles CUDA pointers natively, so the bug doesn't surface. On arm64 64K-page kernels (GB200), the legacy path returns EFAULT for CUDA VRAM. Tracked in [ofiwg/libfabric#12019](https://github.com/ofiwg/libfabric/issues/12019).

**Upstream status:** The bug is resolved in `ofiwg/libfabric` main and v2.5.x via a more comprehensive rewrite of `efa_mr_reg_ibv_mr()`. AWS's `aws/libfabric` fork has not picked up the upstream rewrite; the latest EFA installer (1.48.0) still ships `v2.4.0amzn3.0` with the older code path.

**Workarounds:**

1. **Apply the one-line patch to the bundled libfabric.** During image build, replace the `aws.Dockerfile` install step with a custom build:
  ```dockerfile
   RUN git clone --depth 1 --branch v2.4.0amzn3.0 https://github.com/aws/libfabric.git /tmp/libfabric && \
       cd /tmp/libfabric && \
       sed -i 's/efa_mr_is_neuron(efa_mr) || efa_mr_is_rocr(efa_mr)/efa_mr_is_neuron(efa_mr) || efa_mr_is_rocr(efa_mr) || efa_mr_is_cuda(efa_mr)/' prov/efa/src/efa_mr.c && \
       ./autogen.sh && \
       CPPFLAGS="-I/usr/local/cuda/include" \
       LDFLAGS="-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -Wl,-rpath,/usr/local/cuda/lib64" \
         ./configure --prefix=/opt/amazon/efa --enable-efa --with-cuda=/usr/local/cuda --enable-cuda-dlopen && \
       make -j$(nproc) && make install
   # Then rebuild aws-ofi-nccl from source against the patched libfabric (do not mix versions)
  ```
2. **Replace bundled libfabric with `ofiwg/libfabric@v2.5.1`** (or newer). The upstream rewrite is already present; no patch needed. Rebuild `aws-ofi-nccl` against it.


## Verification

After deployment, confirm EFA is actually being used (not silent TCP fallback):

**1. NIXL chose the LIBFABRIC backend** (not UCX):

```bash
kubectl logs <prefill-pod> | grep -iE "NIXL.*backend|Backend.*instantiated"
# Expected: "Backend LIBFABRIC was instantiated"
# WRONG:    "Backend UCX was instantiated"
```

**2. The LIBFABRIC plugin is loaded and executing** (not just opened):

```bash
kubectl exec <pod> -- bash -c '
  grep "libplugin_LIBFABRIC" /proc/$(pgrep -f "dynamo|vllm|sglang" | head -1)/maps | grep "r-xp"
'
# Expected: at least one line ending in "r-xp"  (executable code page mapped)
# If only "r--p" : library opened but never run — config didn't apply, NIXL chose a different backend
```

**3. Registered RDMA memory is GPU VRAM, not CPU pinned memory** (no CPU bounce):

```bash
kubectl logs <pod> | grep "efa_mr_reg_impl" | head -1
# Look for "Registered memory at 0x7d7749bc4000 of size 431767552"
kubectl exec <pod> -- bash -c 'grep "7d7749bc4" /proc/$(pgrep -f "dynamo|vllm|sglang" | head -1)/maps'
# Expected: NO OUTPUT — CUDA VRAM addresses are not in the Linux VMA table.
# If the address IS found: CPU pinned memory was registered — CPU bounce — GPU-Direct NOT working.
```

**4. NIXL transfers are happening, none failing** (via Prometheus metrics endpoint):

NIXL telemetry is off by default. To enable it, set on each worker:

```yaml
env:
  - { name: NIXL_TELEMETRY_ENABLE,            value: "y" }
  - { name: NIXL_TELEMETRY_EXPORTER,          value: prometheus }
  - { name: NIXL_TELEMETRY_PROMETHEUS_PORT,   value: "19090" }   # NIXL's own port — distinct from framework metrics
```

Then query:

```bash
kubectl exec <pod> -- curl -s localhost:19090/metrics | grep -E "nixl_bytes_transferred|nixl_num_failed_transfers"
# Expected: nixl_bytes_transferred_count > 0 and increasing
#           nixl_num_failed_transfers_total stays 0
```

The same metrics with the `vllm:` prefix are also published to vLLM's own metrics endpoint (typically `DYN_SYSTEM_PORT`, e.g. `8081`) when vLLM is the frontend.

**5. Decode side confirms KV receipt**:

```bash
kubectl logs <decode-pod> | grep "External prefix cache hit rate"
# Expected: "External prefix cache hit rate: 100.0%"
```

> [!NOTE]
> Do not use `rdma_write_bytes` or other `/sys/class/infiniband/*/counters/*` checks for EFA verification. EFA SRD uses SEND operations at the hardware level, not RDMA READ/WRITE — `rdma_write_bytes` is always 0 on correctly configured EFA by design. Use the Prometheus + `/proc/<pid>/maps` methodology above instead.

## Common Failure Modes


| Symptom                                                | Likely cause                                                         | Fix                                                                           |
| ------------------------------------------------------ | -------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| TTFT ~100 s, throughput ~MB/s                          | Silent TCP fallback — NIXL backend selection not applied             | Verify Step 4 backend env var; check NIXL startup log                         |
| TTFT ~10 s, throughput 1–5 GB/s                        | UCX host-staged (no GPU-Direct on kernel ≥ 6.8)                      | Switch to LIBFABRIC backend                                                   |
| Pod fails at startup with `fi_mr_reg` EFAULT on GB200  | Issue 1 (libfabric CUDA dmabuf bug)                                  | Apply patch or use ofiwg/libfabric v2.5.1                                     |
| Pod fails at startup with `fi_mr_reg` EFAULT on x86_64 | `privileged: true` missing OR `efa_nv_peermem` missing on old kernel | Verify Step 5 security context                                                |
| Bandwidth halves after image rebuild                   | libfabric / aws-ofi-nccl ABI mismatch                                | Rebuild aws-ofi-nccl from source against the libfabric used in the same image |
| `rdma_write_bytes` shows 0                             | **Not a failure** — EFA SRD uses SEND, not WRITE                     | Use Prometheus `nixl_bytes_transferred` instead                               |


## References

- [Disaggregated Communication Guide](../../disagg-communication-guide.md) — transport-layer fundamentals
- [RDMA / InfiniBand on AKS](../aks/rdma-infiniband.md) — Azure equivalent
- [`container/templates/aws.Dockerfile`](../../../../container/templates/aws.Dockerfile) — EFA installer template
- [AWS — Manage EFA devices on Amazon EKS](https://docs.aws.amazon.com/eks/latest/userguide/device-management-efa.html) — official EKS-side guide (DRA driver + device plugin)
- [AWS EFA documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html) — EC2-side EFA overview
- [`aws/eks-charts` — `aws-efa-k8s-device-plugin`](https://github.com/aws/eks-charts/tree/master/stable/aws-efa-k8s-device-plugin) — Helm chart source
- [ofiwg/libfabric#12019](https://github.com/ofiwg/libfabric/issues/12019) — CUDA dmabuf registration on EFA
