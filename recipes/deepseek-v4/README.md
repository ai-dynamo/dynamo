<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4 Recipes

Dynamo + vLLM serving recipes for **DeepSeek-V4-Pro** and **DeepSeek-V4-Flash**,
tuned for the **agentic** workload (64k ISL / 400 OSL / 90% KV reuse) at a floor of
**≥ 50 output tok/s/user**. Each variant is a `DynamoGraphDeployment` (DGD); a single
shared [`perf/`](perf) Job replays the benchmark traces against any variant.

## Configurations

Per-worker parallelism and speculative decode differ between the prefill and decode
workers in the disaggregated variants, so those are listed as `prefill / decode`.

| Variant (`vllm/…`) | Model | GPUs | Prefill / Decode | MoE backend | Spec. decode | `max_model_len` | Disagg fabric |
|---|---|---|---|---|---|---|---|
| `deepseek-v4-pro/vllm/agg-b200-agentic` | `nvidia/DeepSeek-V4-Pro-NVFP4` | 8× B200 | TP8 + EP | FLASHINFER_TRTLLM | MTP-2 | 1,048,576 | — |
| `deepseek-v4-pro/vllm/agg-h200-agentic` | `deepseek-ai/DeepSeek-V4-Pro` | 8× H200 | TP8 + EP | MARLIN | none | 86,016 ¹ | — |
| `deepseek-v4-pro/vllm/disagg-b200-agentic` | `nvidia/DeepSeek-V4-Pro-NVFP4` | 16× B200 | TP8+EP / TP8+EP | FLASHINFER_TRTLLM | none / MTP-2 | 1,048,576 | NIXL GDR |
| `deepseek-v4-pro/vllm/disagg-h200-agentic` | `deepseek-ai/DeepSeek-V4-Pro` | 32× H200 (1P·3D) | TP8+EP / TP8+EP | MARLIN | none | 86,016 ¹ | NIXL GDR ² |
| `deepseek-v4-flash/vllm/agg-b200-agentic` | `nvidia/DeepSeek-V4-Flash-NVFP4` | 4× B200 | TP4 | FLASHINFER_TRTLLM | none | 1,048,576 | — |
| `deepseek-v4-flash/vllm/agg-h200-agentic` | `deepseek-ai/DeepSeek-V4-Flash` | 4× H200 | DP4 + TP1 + EP | MARLIN (FLASHINFER_MLA attn) | MTP-1 | 1,048,576 | — |
| `deepseek-v4-flash/vllm/disagg-b200-agentic` | `nvidia/DeepSeek-V4-Flash-NVFP4` | 8× B200 | TP4 / TP4 | FLASHINFER_TRTLLM | none | 1,048,576 | NIXL GDR |
| `deepseek-v4-flash/vllm/disagg-h200-agentic` | `deepseek-ai/DeepSeek-V4-Flash` | 28× H200 (4P·3D) | DP4+TP1+EP / DP4+TP1+EP | MARLIN (FLASHINFER_MLA attn) | none / MTP-1 | 1,048,576 | NIXL GDR ² |

Common to all: FP8 KV cache, block size 256, KV-aware routing, prefix caching. B200 variants
serve the NVFP4 checkpoints; H200 variants serve the public checkpoints. Modality: text; reasoning + tool calling supported.

¹ **H200 Pro is capped at `max_model_len=86,016`**: the full-precision weights plus a 1M-token KV
cache do not fit in H200 memory (OOM at load / during `determine_available_memory`). Requests longer
than ~86k tokens are rejected, so the H200 Pro variants cannot serve the longest contexts the B200
(1M) recipes can. B200 and H200-Flash keep the full 1M window.

² **H200 disaggregated uses NIXL over RDMA/GDR.** See [Per-rank NIC mapping](#per-rank-nic-mapping-b200--h200-disaggregated) for what it needs and why.


## Optimization targets

| Workload | Median ISL | Median OSL | KV reuse | User output tok/s |
|---|---:|---:|---:|---:|
| Agentic (coding / tool use) | 64k | 400 | 90% | ≥ 50 |

Benchmarks replay [Mooncake-format](https://github.com/kvcache-ai/Mooncake) traces — see [`perf/README.md`](perf/README.md).

## Performance results

Floor-picks (max system tok/s/GPU at user_p50 ≥ 50), full agentic trace, default temperature.

Variant keys match the [Configurations](#configurations) table (`${MODEL}/vllm/${MODE}-${SKU}-agentic`).

| Variant (`vllm/…`) | Workload | Concurrency | User tok/s | System tok/s/GPU |
|---|---|---:|---:|---:|
| `deepseek-v4-pro/vllm/agg-b200-agentic` | Agentic | 13 | 51.3 | 72.8 |
| `deepseek-v4-pro/vllm/disagg-b200-agentic` | Agentic | 28 | 51.3 | 77.3 |
| `deepseek-v4-pro/vllm/agg-h200-agentic` | Agentic | 8 | 53.2 | 45.9 |
| `deepseek-v4-pro/vllm/disagg-h200-agentic` | Agentic | 32 | 50.5 | 43.9 |
| `deepseek-v4-flash/vllm/agg-b200-agentic` | Agentic | 38 | 50.6 | 362.1 |
| `deepseek-v4-flash/vllm/disagg-b200-agentic` | Agentic | 72 | 70.1 | 340.8 |
| `deepseek-v4-flash/vllm/agg-h200-agentic` | Agentic | 16 | 50.7 | 145.4 |
| `deepseek-v4-flash/vllm/disagg-h200-agentic` | Agentic | 128 | 55.3 | 213.3 |

**AGG figures are single-replica floor-picks** (best tok/s/GPU at user_p50 ≥ 50). AGG scales by deploying
independent replicas; KV-routed *multi*-replica AGG does **not** improve per-GPU throughput — see
[Known limitations](#known-limitations).

## Prerequisites

1. **Dynamo Platform installed** — see the [Kubernetes Deployment Guide](../../docs/kubernetes/README.md).
2. **GPU cluster** matching the variant (8× B200 / 8× H200 for aggregated; 16× / 8× for 1P1D
   disaggregated), nodes labeled `nvidia.com/gpu.product=NVIDIA-B200` or `NVIDIA-H200`.
3. **HuggingFace token** with access to the checkpoints you deploy:
   ```bash
   export NAMESPACE=your-namespace
   kubectl create namespace ${NAMESPACE}
   kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN="your-token" -n ${NAMESPACE}
   ```

## Quick Start

The variant path is `${MODEL}/vllm/${MODE}-${SKU}-agentic` where `MODEL ∈ {deepseek-v4-pro,
deepseek-v4-flash}`, `MODE ∈ {agg, disagg}`, `SKU ∈ {b200, h200}`. Example below deploys
**Pro AGG B200**; change the variables for any other cell.

```bash
export NAMESPACE=your-namespace
MODEL=deepseek-v4-pro
SKU=b200
MODE=agg

# 1) Storage — edit storageClassName (RWX) first.
kubectl apply -f ${MODEL}/model-cache/model-cache.yaml -n ${NAMESPACE}

# 2) Weights — model-download.yaml fetches both the NVFP4 (B200) and public (H200)
#    checkpoints; delete the line that does not apply to your SKU to save disk/time.
kubectl apply -f ${MODEL}/model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=14400s

# 3) Deploy. For ANY disaggregated variant (B200 or H200 — both use NIXL GDR), first set
#    VLLM_GPU_NIC_PCIE_MAPPING in the manifest to your cluster's map (see "Per-rank NIC mapping" below).
kubectl apply -f ${MODEL}/vllm/${MODE}-${SKU}-agentic/deploy.yaml -n ${NAMESPACE}

# 4) Benchmark — see perf/README.md to point ENDPOINT + TRACE_FILE at this DGD.
kubectl apply -f perf/perf.yaml -n ${NAMESPACE}
```

First worker launch loads weights and captures CUDA graphs and can take tens of minutes.

## Per-rank NIC mapping (B200 & H200 disaggregated)

**Why it's needed.** In disaggregated serving the prefill workers stream the KV cache to the decode workers'
GPUs over InfiniBand using **GPU-Direct RDMA (GDR)**. Each tensor-parallel rank should transfer over the RDMA
NIC on **its own PCIe switch** (its affine NIC); if several ranks are forced onto the **same** NIC they
oversubscribe it and GPU-direct registration collapses to slow host-staging — **~0.76 GB/s with 4 ranks on
one NIC vs ~15–25 GB/s with one NIC per rank (~25× slower)**. Our disagg recipe originally pinned every rank
to one device (`UCX_NET_DEVICES=mlx5_0:1`), triggering exactly this. `VLLM_GPU_NIC_PCIE_MAPPING`
(`VLLM_NIC_SELECTION_VARS=UCX_NET_DEVICES:1`, [vLLM #42083](https://github.com/vllm-project/vllm/pull/42083))
assigns each rank its PCIe-affine NIC and restores GDR. **This is a general tensor-parallel NIC-assignment
requirement, not a DeepSeek-V4 property** — any TP>1 NIXL disaggregated worker pinned to a single NIC collapses
the same way, and DSV4's "packed" KV layout (investigated at length) was ruled out as the cause.

**Scope: disaggregated only.** Only recipes that ship **disaggregated** over GDR use this. Aggregated recipes
have no cross-worker KV transfer and ignore `VLLM_GPU_NIC_PCIE_MAPPING` (empty ⇒ no-op). Among the recommended
picks that's **Pro B200 DisAgg** and **Flash H200 4P3D**.

**Providing the affine NIC per rank.** Set `VLLM_GPU_NIC_PCIE_MAPPING` to the node's `GPU_BDF=NIC_BDF` pairs
so each rank uses its PCIe-affine NIC (regenerate per node type — below). This assumes the cluster exposes
each pod the RDMA NICs affine to its GPUs; the map references those NIC BDFs, so it **breaks if the cluster
injects a different NIC set** — derive it from the pod's actual `/sys/class/infiniband` rather than hard-coding.
Note there is no "just auto-select" shortcut: pinning all ranks to one NIC degrades to **single-NIC
host-staging (~20× slower, but still functional)**, while *unsetting* `UCX_NET_DEVICES` to expose all NICs can
**fail NIXL backend creation** — the working fast path is exactly one affine NIC per rank, which this map
constructs. With no RDMA fabric at all, fall back to TCP: `NCCL_IB_DISABLE=1`, `NCCL_NET=Socket`, and drop the
`rdma/ib` resource requests.

The B200 disaggregated recipes request **`rdma/shared_ib: "1"`** (a shared-IB resource that exposes **all** of
the node's RDMA NICs to the pod) so the map can reliably pick each rank's affine NIC even when the cluster's
device plugin would otherwise inject a non-affine set. On a cluster without a shared-IB resource, request the
affine NICs directly instead (e.g. `rdma/ib: "N"`).

Regenerate the map on a target node:

```bash
# 1) GPU PCIe BDFs, in device order:
nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader
#    -> e.g. 0, 00000000:18:00.0

# 2) RDMA NIC PCIe BDF for each mlx5 device:
for d in /sys/class/infiniband/*/device; do
  echo "$(basename "$(dirname "$d")") -> $(basename "$(readlink -f "$d")")"
done
#    -> e.g. mlx5_0 -> 0000:19:00.0
```

Pair each GPU with the NIC sharing its PCIe switch (closest NIC), then join the pairs in GPU order:

```
VLLM_GPU_NIC_PCIE_MAPPING="<gpu0_bdf>=<nic0_bdf>,<gpu1_bdf>=<nic1_bdf>,…"
```

Set the same map on **both** the prefill and decode workers. After deploy, confirm the worker log shows
NIXL registering the CUDA/GDR path (~15–17 GB/s KV transfer). A wrong or missing map degrades to slow
TCP transfer or fails NIXL setup.

## Known limitations

- **Multi-replica AGG does not scale per-GPU.** With `DYN_ROUTER_MODE=kv`, the KV-affinity router
  concentrates the 90%-reuse agentic prefixes onto a few replicas, so KV-routed *multi*-replica AGG lands
  *below* single-replica per-GPU throughput (measured N=7, `router_temperature=1.0`: Flash B200 293 vs 362,
  H200 121 vs 145 tok/s/GPU). Deploy AGG as **independent single-replica DGDs** for linear scaling; the
  disaggregated variants are the multi-worker path.
- **Workers run as `runAsUser: 0`.** FlashInfer's TRT-LLM FP4 MoE JIT writes cubins into a
  root-owned `site-packages` directory, which a non-root user cannot write during
  `determine_available_memory`. The fix (make that directory group-writable in the image) is tracked
  separately; drop `runAsUser: 0` once it lands.
