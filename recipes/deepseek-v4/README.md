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
| `deepseek-v4-pro/vllm/agg-h200-agentic` | `deepseek-ai/DeepSeek-V4-Pro` | 8× H200 | TP8 + EP | vLLM default ³ | none | 86,016 ¹ | — |
| `deepseek-v4-pro/vllm/disagg-b200-agentic` | `nvidia/DeepSeek-V4-Pro-NVFP4` | 16× B200 | TP8+EP / TP8+EP | FLASHINFER_TRTLLM | none / MTP-2 | 1,048,576 | NIXL GDR |
| `deepseek-v4-pro/vllm/disagg-h200-agentic` | `deepseek-ai/DeepSeek-V4-Pro` | 32× H200 (1P·3D) | TP8+EP / TP8+EP | vLLM default ³ | none | 86,016 ¹ | NIXL GDR ² |
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

² **H200 disaggregated uses NIXL over RDMA / GDR** (`NCCL_IB_DISABLE=0`, `rdma/ib` resources, per-rank
`VLLM_GPU_NIC_PCIE_MAPPING`) — this is the fabric the shipped **Flash H200 4P3D** (213.3 tps/GPU,
c128 full-run) and **Pro H200 disagg** numbers were measured on. The baked `VLLM_GPU_NIC_PCIE_MAPPING`
is **node-topology-specific** (an 8×H200 + 8×CX-7 node); regenerate it for your hardware (see "RDMA
per-rank NIC mapping" below). For a cluster with no RDMA fabric, fall back to TCP (`NCCL_IB_DISABLE=1`,
`NCCL_NET=Socket`, drop `rdma/ib`).

³ The H200 Pro variants do not pin `--moe-backend`; vLLM auto-selects its default MoE kernel for the
public FP8/BF16 path.

## Optimization targets

| Workload | Median ISL | Median OSL | KV reuse | User output tok/s |
|---|---:|---:|---:|---:|
| Agentic (coding / tool use) | 64k | 400 | 90% | ≥ 50 |

Benchmarks replay [Mooncake-format](https://github.com/kvcache-ai/Mooncake) traces — see [`perf/README.md`](perf/README.md).

## Performance results

Floor-picks (max system tok/s/GPU at user_p50 ≥ 50), full agentic trace, default temperature.

| Variant | Workload | Concurrency | User tok/s | System tok/s/GPU |
|---|---|---:|---:|---:|
| Pro AGG B200 | Agentic | 13 | 51.3 | 72.8 |
| Pro DisAgg B200 | Agentic | 28 | 51.3 | 77.3 |
| Pro AGG H200 | Agentic | 8 | 53.2 | 45.9 |
| Pro DisAgg H200 | Agentic | 16 | 52.6 | 24.3 |
| Flash AGG B200 | Agentic | 38 | 50.6 | 362.1 |
| Flash DisAgg B200 | Agentic | 72 | 70.1 | 340.8 |
| Flash AGG H200 | Agentic | 16 | 50.7 | 145.4 |
| Flash DisAgg H200 | Agentic | 32 | 62.1 | 183.0 |

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

All disaggregated variants (B200 **and** H200) move the KV cache over **NIXL GDR**, which needs each GPU
mapped to its closest RDMA NIC via `VLLM_GPU_NIC_PCIE_MAPPING` (`GPU_BDF=NIC_BDF,…`, [vLLM #42083](https://github.com/vllm-project/vllm/pull/42083)).
**This value is cluster-specific** — the map shipped in the B200 disagg manifests is the node layout we
benchmarked and will not match a different cluster. Regenerate it on a target node:

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

## Known issues

- **Workers run as `runAsUser: 0`.** FlashInfer's TRT-LLM FP4 MoE JIT writes cubins into a
  root-owned `site-packages` directory, which a non-root user cannot write during
  `determine_available_memory`. The fix (make that directory group-writable in the image) is tracked
  separately; drop `runAsUser: 0` once it lands.
- **Image tag is a staging build.** Manifests pin
  `nvcr.io/nvstaging/nim/sungsooh:dsv4-dynamo-vllm-cu130-v0240-dynmain-4a8c55b3-nopatch-20260702`
  (see the `TODO(post-PR)` in each manifest); replace with the Dynamo CI `deepseek-v4` vllm-runtime tag.
