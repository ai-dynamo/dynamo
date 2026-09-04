<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A.X-K2 Recipes

Recipes for [`skt/A.X-K2`](https://huggingface.co/skt/A.X-K2), SK Telecom's 688B-parameter
MoE model (33B active) with DeepSeek-V3-style Multi-head Latent Attention (MLA), a block-scaled
FP8 checkpoint (~647 GiB), and a 256K context window (Apache-2.0), served with NVIDIA Dynamo and
vLLM. Targets are organized per topology and GPU (`vllm/<topology>-<gpu>/`); the current targets
run on 32x H200. The model needs the [SKT-AI vLLM fork](https://github.com/SKT-AI/vllm), so the
workers run a custom runtime image; see [`container/README.md`](container/README.md).

## Deployment Targets

| Target | GPUs | Topology | Parallelism | Routing | Role |
|---|---|---|---|---|---|
| [`disagg-h200`](vllm/disagg-h200/deploy.yaml) | 32x H200 (4 nodes) | 2P2D disaggregated | prefill TP2×DP4 + EP / decode TP8 + EP | KV-aware, NIXL over RDMA | **Recommended** |
| [`agg-h200`](vllm/agg-h200/deploy.yaml) | 8x H200 per worker (4 workers measured) | Aggregated | TP8 + EP | Round-robin | Baseline |

Both serve the model as `A.X-K2` with FP8 MLA KV cache (`fp8_ds_mla`), `--gpu-memory-utilization 0.90`,
`--max-num-seqs 256`, prefix caching, the `deepseek_v3` reasoning and `hermes` tool-call parsers,
and thinking mode on by default. Prefill batch budget is 32K tokens on TP8, the 8K engine default
on TP2×DP4 (32K does not fit), and 2K on decode. KV capacity per worker: 864K tokens on TP8
(49 requests at 16K), 1.77M tokens on TP2×DP4.

## Highlights

### H200 (32 GPUs, 16K input / 1K output)

- **Disaggregation** (same KV-aware routing): +14 to +42 % total throughput on shared-prefix traffic
  and +42 to +98 % on cold traffic between 32 and 128 concurrent requests; decode ITL p99 drops
  from 874–2,174 ms to 80–88 ms.
- **KV-aware routing** on 2P2D: TTFT p50 halved (2.2–3.5 s → 1.1–1.8 s) and no stall at 128
  concurrent cold requests. On aggregated TP8 it adds only +2 to +7 % throughput and makes TTFT p50
  2–4x worse, so `agg-h200` ships with round-robin.
- **Recommended vs baseline** (`agg-h200` → `disagg-h200`): +16 to +51 % shared-prefix and
  +40 to +97 % cold throughput at 32–128 concurrent requests; within 8 % at 8–16.
- **Operating limit:** about 100 concurrent cold 16K requests per 32-GPU unit (decode KV capacity).

## Prerequisites

- Dynamo Platform with the `nvidia.com/v1beta1` `DynamoGraphDeployment` API.
- 8-GPU H200 nodes: one per aggregated worker, four for `disagg-h200`, with RDMA (InfiniBand or
  RoCE) exposed to the pods for NIXL KV transfer.
- A worker image built from the A.X-K2 vLLM fork on `vllm-runtime:1.4.2` ([`container/README.md`](container/README.md)).
  Replace the `your-registry.example.com/...` placeholder before deploying.
- Cluster bindings. The manifests are portable bases per the [contribution guide](../CONTRIBUTING.md):
  add namespace, node selection, tolerations, image pull Secrets, RDMA device resources and network
  attachments, and `UCX_NET_DEVICES` / `NCCL_IB_HCA` through the
  [cluster Kustomization starter](../templates/kustomize/README.md) or a private copy. The measured
  values are in [Cluster bindings](#cluster-bindings-used-in-the-measurement).
- Optional: an `hf-token-secret` for the download Job. The checkpoint is public and ungated; a token only raises rate limits.

## Quick Start

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
# Optional; the download Job runs without it.
kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN="your-token" -n ${NAMESPACE}

# Set storageClassName in model-cache/model-cache.yaml first (RWX, 1000Gi; claim name shared-model-cache).
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=14400s

MODE=disagg   # or agg
kubectl apply -f vllm/${MODE}-h200/deploy.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=axk2-vllm-${MODE}-h200 -n ${NAMESPACE} --timeout=5400s
```

Workers read the checkpoint offline from the PVC and keep their DeepGEMM / FlashInfer / Triton JIT
caches there, so only the first start pays the ~25-minute cold warmup. To change a running
deployment, delete the DGD and re-apply rather than patching it.

Smoke test:

```bash
kubectl port-forward svc/axk2-vllm-${MODE}-h200-frontend 8000:8000 -n ${NAMESPACE}
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"A.X-K2","messages":[{"role":"user","content":"A.X K2를 한 문장으로 소개해줘."}],"max_tokens":256}'
```

Thinking mode is on by default; pass `"chat_template_kwargs": {"enable_thinking": false}` to
disable it per request.

## Benchmark

[`perf/perf.yaml`](perf/perf.yaml) runs an AIPerf closed-loop concurrency sweep at 16,384 input /
1,024 output tokens. Set `ENDPOINT`, `WORKLOAD` (`w1` shared prefix, `w2` cold, `w3` single prefix),
and `CONCURRENCIES` in its `env` block:

```bash
kubectl apply -f perf/perf.yaml -n ${NAMESPACE}
kubectl logs -f job/axk2-bench -n ${NAMESPACE}
```

Artifacts land under `/shared-model-cache/perf/axk2/`. Keep cold-workload concurrency at or below
128 on `disagg-h200` (see [Known issues](#known-issues)).

## Performance

### H200

Measured on 4 nodes x 8 H200, Dynamo 1.4.2, vLLM 0.28.1rc1 (A.X-K2 fork),
NIXL 1.3.2, InfiniBand with four HCAs per node; `vllm bench serve` closed-loop, seed 42, 12 requests
per concurrency slot, single run per cell. Every request is 16,384 input / 1,024 output tokens:

- **W1 (shared prefix):** 32 shared 10,240-token prefixes + 6,144 unique tokens per request.
- **W2 (cold):** 16,384 unique tokens per request.
- **W3 (cache ceiling):** one shared 10,240-token prefix + 6,144 unique tokens.

Arms (all on the same 32 GPUs, image, and engine settings): **B1** agg 4x TP8 round-robin
(= `agg-h200`), **C** = B1 with KV-aware routing (enable the commented lines in `agg-h200`),
**D** 2P2D with KV-aware routing (= `disagg-h200`), **D'** = D with random prefill routing,
**B2** agg 4x TP2×DP4 round-robin.

#### 1. Prefill/decode disaggregation effect (routing held constant: C → D)

| Concurrency | W1 total tok/s | W2 total tok/s | W1 TPOT p50 | W2 TPOT p50 | W1 ITL p99 | W2 ITL p99 |
|---:|---|---|---|---|---|---|
| 8 | 7,173 → 7,002 (−2 %) | 6,900 → 6,674 (−3 %) | 17.3 → 17.2 ms | 17.4 → 16.9 ms | 31 → 18 ms | 31 → 17 ms |
| 16 | 11,624 → 12,145 (+4 %) | 11,173 → 12,530 (+12 %) | 20.9 → 20.2 ms | 22.0 → 19.1 ms | 35 → 21 ms | 35 → 20 ms |
| 32 | 17,523 → 20,003 (+14 %) | 14,484 → 20,599 (+42 %) | 27.7 → 24.6 ms | 34.6 → 23.1 ms | 48 → 79 ms | 41 → 46 ms |
| 64 | 24,832 → 31,280 (+26 %) | 18,618 → 30,154 (+62 %) | 39.5 → 31.6 ms | 53.5 → 29.4 ms | 874 → 82 ms | 2,149 → 87 ms |
| 128 | 33,186 → 47,056 (+42 %) | 22,185 → 43,931 (+98 %) | 60.0 → 42.0 ms | 86.8 → 34.4 ms | 879 → 80 ms | 2,174 → 88 ms |
| 256 | 35,498 → 48,079 (+35 %) | 21,428 → stalled | 79.1 → 56.5 ms | 91.9 → stalled | 2,016 → 330 ms | 4,246 → stalled |

- Gains concentrate at 32–128 concurrent requests; at 8–16 the prefill half of the GPUs idles.
- Decode isolation is the largest effect: prefill chunks no longer block decode steps, so ITL p99
  falls about 11x and TPOT p50 20–60 %.
- W1 TTFT p50 improves (4.25 → 1.82 s at 128); W2 TTFT p50 is 0.6–1.0 s worse at 32–128 because
  TP2×DP4 prefills a 16K prompt in two 8K chunks and decode pulls the KV over NIXL first.
- Each TP8 decode worker holds 49 requests at 16K. At 256, W1 completes with 32 s TTFT p50 and
  heavy preemption; W2 stalls.

#### 2. KV-aware routing effect (topology held constant: B1 → C, D' → D)

| Concurrency | agg: round-robin → KV, W1 tok/s | agg: round-robin → KV, W1 TTFT p50 | disagg: random → KV, W1 tok/s | disagg: random → KV, W1 TTFT p50 | disagg: random → KV, W2 tok/s |
|---:|---|---|---|---|---|
| 8 | 6,984 → 7,173 (+3 %) | 2.20 → 2.20 s | 6,389 → 7,002 (+10 %) | 2.24 → 1.12 s | 6,502 → 6,674 (+3 %) |
| 16 | 11,260 → 11,624 (+3 %) | 1.65 → 1.01 s | 10,808 → 12,145 (+12 %) | 3.04 → 1.10 s | 11,323 → 12,530 (+11 %) |
| 32 | 17,219 → 17,523 (+2 %) | 0.95 → 2.01 s | 18,437 → 20,003 (+8 %) | 2.74 → 1.38 s | 18,755 → 20,599 (+10 %) |
| 64 | 23,730 → 24,832 (+5 %) | 0.96 → 2.22 s | 29,843 → 31,280 (+5 %) | 2.70 → 1.56 s | 30,377 → 30,154 (−1 %) |
| 128 | 31,164 → 33,186 (+6 %) | 1.14 → 4.25 s | 46,556 → 47,056 (+1 %) | 3.51 → 1.82 s | stalled → 43,931 |
| 256 | 33,138 → 35,498 (+7 %) | 37.4 → 30.6 s | 45,478 → 48,079 (+6 %) | 35.3 → 32.3 s | — → stalled |

- Throughput is not where routing pays: +2 to +7 % on aggregated, +1 to +12 % on 2P2D, −1 to −8 %
  on cold traffic, and −1 to −3 % at the W3 cache ceiling (event publishing and router overhead).
- On 2P2D it is essential: requests land on the prefill worker holding their prefix (TTFT p50
  halved), and load awareness kept both decode workers at 64/63 in flight at 128 concurrent cold
  requests where random routing filled one worker to 100 % KV and stalled.
- On aggregated TP8 it raises TTFT p50 2–4x at 32–128 (cause not isolated), so `agg-h200` ships
  with round-robin.

**B2 (agg TP2×DP4, not shipped):** 4x the KV capacity keeps TTFT p50 flat at 2.0–5.0 s through 256
concurrent requests and beats B1 by +28 % (W1) / +7 to +69 % (W2) from 32 upward, at −9 to −18 %
throughput at 8–16. Derive it from `agg-h200` with the prefill worker's parallelism and MoE flags.

**Limits:** single run per cell, closed-loop only (no Poisson goodput), prefill batch budgets differ
by layout (32K vs 8K), W3 measured on B1 and C only.

## Known issues

1. **2P2D decode stall at KV saturation (open).** With cold 16K prompts, once both decode workers
   reach 51 running sequences and 100 % KV with a growing queue, generation stops (D' W2 at 128,
   D W2 at 256). NIXL reported no errors; workers recover when the client aborts. Cap concurrency
   at ~100 per unit or set decode `--max-num-seqs` near the KV limit (49 at 16K).
2. **TP2×DP4 needs the default prefill budget.** `--max-num-batched-tokens 32768` fails at startup
   (`Available KV cache memory: -17.53 GiB`) because dense weights are replicated per GPU.
3. **FlashInfer autotune hangs a cold node** for 25+ minutes until the NCCL watchdog kills the
   worker; both recipes keep it off and persist the JIT caches on the PVC.

## Cluster bindings used in the measurement

### H200

Omitted from the portable manifests; supplied by the cluster Kustomization. Nodes: 4x H200 SXM
(8x 141 GB), one worker pod per node, selected by an InfiniBand-fabric node label plus a
`nvidia.com/gpu` `NoSchedule` toleration. RDMA: four IB physical functions per pod via a
`k8s.v1.cni.cncf.io/networks` annotation (one host-device attachment per HCA) and the matching
extended resource with count 4; `UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1` and
`NCCL_IB_HCA=mlx5`. Two image pull Secrets for the private registry. Probes: operator defaults.
