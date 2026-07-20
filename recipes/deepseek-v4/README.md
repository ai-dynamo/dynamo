<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4 Recipes

Dynamo + vLLM serving recipes for **DeepSeek-V4-Pro** and **DeepSeek-V4-Flash**,
tuned for the **agentic** workload (64k ISL / 400 OSL / 90% KV reuse) at a floor of
**≥ 50 output tok/s/user**. Each variant is a `DynamoGraphDeployment` (DGD); a single
shared [`perf/`](perf) Job replays the benchmark traces against any variant.

## Recipes by model

Each model's README is self-contained — its own **Recipes** (agentic + Day-0), **Performance**,
and **Quick Start**. The sections below this table (Optimization targets, Per-rank NIC mapping,
Known limitations) are **shared** and linked from both.

| Model | Size | Recommended picks | README |
|---|---|---|---|
| **DeepSeek-V4-Pro** | 1.6T / 49B active · 1M ctx | B200 → `disagg-b200-agentic` · H200 → `agg-h200-agentic` | [`deepseek-v4-pro/`](deepseek-v4-pro/) |
| **DeepSeek-V4-Flash** | 284B / 13B active | B200 → `disagg-b200-agentic` (4P2D) · H200 → `disagg-h200-agentic` (4P3D) | [`deepseek-v4-flash/`](deepseek-v4-flash/) |

B200 variants serve the NVFP4 checkpoints (`nvidia/DeepSeek-V4-*-NVFP4`); H200 variants serve the
public checkpoints (`deepseek-ai/DeepSeek-V4-*`). H200 Pro is capped at `max_model_len=86,016` (HBM);
B200 and H200-Flash keep the full 1M window. Modality: text; reasoning + tool calling supported.

## Optimization targets

| Workload | Median ISL | Median OSL | KV reuse | User output tok/s |
|---|---:|---:|---:|---:|
| Agentic (coding / tool use) | 64k | 400 | 90% | ≥ 50 |
| Custom (decode-heavy synthetic) | 10k | 1k | 10% | ≥ 50 |

Benchmarks replay [Mooncake-format](https://github.com/kvcache-ai/Mooncake) traces — see
[`perf/README.md`](perf/README.md). Per-variant floor-picks (max system tok/s/GPU at user_p50 ≥ 50)
live in each model's **Performance** section. Pro is characterized on the Agentic workload only; the
Custom cell is Flash-only.

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
have no cross-worker KV transfer and set neither NIC env var (no GDR needed). Among the recommended
picks that's **Pro B200 DisAgg (1P1D)**, **Flash B200 DisAgg (4P2D)**, and **Flash H200 DisAgg (4P3D)** —
Pro H200 ships AGG, so it needs no NIC map.

**The two NIC env vars are a pair.** `VLLM_GPU_NIC_PCIE_MAPPING` (the node's `GPU_BDF=NIC_BDF` map) and
`VLLM_NIC_SELECTION_VARS` (a fixed selector, `UCX_NET_DEVICES:1`) must be set **together or not at all** —
setting only one fails EngineCore init ([vLLM #42083](https://github.com/vllm-project/vllm/pull/42083)). **The
recipe ships neither** (the map is node-specific and cannot be pre-baked); to enable GDR add **both** per the
steps below. Without them, disagg falls back to single-NIC host-staging (silently under `shared_ib`; see the throughput figures above).

**Providing the affine NIC per rank.** Set `VLLM_GPU_NIC_PCIE_MAPPING` to the node's `GPU_BDF=NIC_BDF` pairs
so each rank uses its PCIe-affine NIC (regenerate per node type — below). This assumes the cluster exposes
each pod the RDMA NICs affine to its GPUs; the map references those NIC BDFs, so it **breaks if the cluster
injects a different NIC set** — derive it from the pod's actual `/sys/class/infiniband` rather than hard-coding.
Note there is no "just auto-select" shortcut: pinning all ranks to one NIC degrades to **single-NIC
host-staging (~20× slower, but still functional)**, while *unsetting* `UCX_NET_DEVICES` to expose all NICs can
**fail NIXL backend creation** — the working fast path is exactly one affine NIC per rank, which this map
constructs. With no RDMA fabric at all, fall back to TCP: `NCCL_IB_DISABLE=1`, `NCCL_NET=Socket`, and drop the
`rdma/ib` resource requests.

**`rdma/ib` vs `rdma/shared_ib` is cluster-plugin-dependent, not GPU-dependent** — which RDMA resource to
request depends on your cluster's RDMA device plugin:
- **`rdma/ib: "N"`** (N exclusive devices) is clean and NIC-isolated but relies on the plugin injecting the
  pod's *affine* NICs. The **H200 disaggregated** recipes (Pro + Flash) ship this (validated where affine
  injection works).
- **`rdma/shared_ib: "1"`** exposes **all** of the node's NICs to the pod, so the per-rank map can always find
  each rank's affine NIC even when the plugin would inject a non-affine set. The **B200 disaggregated** recipes
  (Pro + Flash) ship this (validated on a cluster where affine injection had drifted). Both pair with the same
  `VLLM_GPU_NIC_PCIE_MAPPING`; pick whichever your cluster's plugin exposes.

Caveat for `shared_ib`: because it exposes **all** NICs, a **wrong or stale map fails silently**. Under
exclusive `rdma/ib`, a map that points a rank at a NIC not in the pod hard-fails (`createBackend` errors);
under `shared_ib` that NIC is present, so the rank quietly opens a **non-affine** NIC and GDR degrades to a
cross-socket path with **no error**. So **derive the map from the pod's actual `/sys/class/infiniband` at
deploy time and verify the per-rank NIC selection in the worker log** rather than trusting a hard-coded map.
(Shared NICs are also not bandwidth-isolated from co-tenant pods — a throughput, not a correctness, concern.)

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
VLLM_NIC_SELECTION_VARS="UCX_NET_DEVICES:1"
```

Set **both** env vars — the generated `VLLM_GPU_NIC_PCIE_MAPPING` and the fixed `VLLM_NIC_SELECTION_VARS` (they are a pair, see above) — on **both** the prefill and decode workers. After deploy, confirm the worker log shows
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
