# Qwen3.5-122B-A10B-FP8 — tp1 + MTP, aggregated, KV-aware routing (H200)

Recipe for [Qwen/Qwen3.5-122B-A10B-FP8](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8),
the FP8 checkpoint of [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)
(122B total / 10B active hybrid MoE — Gated DeltaNet linear attention + MoE with full
attention every 4th layer). The FP8 weights fit a single 143 GB H200 at the full
262,144-token context, which is what this recipe exploits: **one TP1 engine per GPU,
scaled horizontally behind a Dynamo KV-aware router**, with MTP speculative decoding.

## Configuration

Dynamo + vLLM aggregated profile for the agentic workload on **H200**.

|                          | H200 aggregated agentic (tp1 + MTP)              |
| ------------------------ | ------------------------------------------------ |
| **GPU**                  | 1x H200 per worker; scale via `replicas`         |
| **Mode**                 | Aggregated                                       |
| **Framework**            | vLLM (runtime `1.3.0`)                           |
| **Precision**            | FP8 weights + BF16 KV                            |
| **Parallelism**          | TP1                                              |
| **KV cache manager**     | Hybrid (DeltaNet SSM + attention)                |
| **Routing**              | KV-aware (`DYN_ROUTER_MODE=kv`) + worker KV events |
| **Speculative decoding** | MTP, `num_speculative_tokens=3`                  |
| **Context length**       | 262,144 (model default)                          |

### Why TP1 + replicas + KV routing

Every multi-GPU engine layout measured (TP2, TP4, TP8, DP+EP) delivered less output
throughput **per GPU** than independent TP1 replicas at the agentic SLA. The winning
layout is one engine per GPU, scaled horizontally behind the KV-aware router. KV routing
is load-bearing: the replicas are independent engines and agentic requests share
~57k-token prefixes, so the router must land each request on the replica that already
holds its prefix. The DGD ships `replicas: 2` (minimal KV-router validation); a full 8x
H200 node runs `replicas: 8`.

## Supported features

- Modalities: Text
- Reasoning (`--dyn-reasoning-parser qwen3`)
- Tool calling (`--dyn-tool-call-parser qwen3_coder`)

## Prerequisites

1. **Dynamo Platform installed** on the cluster with DGD CRDs served.
2. **NGC/nvcr image pull access** for `nvcr.io/nvidia/ai-dynamo` — create `nvcr-secret`
   and attach it (see Quick Start note).
3. **Hugging Face token** with access to `Qwen/Qwen3.5-122B-A10B-FP8` (public, Apache-2.0),
   stored as `hf-token-secret` — used by the model-download Job.
4. **`model-cache` PVC** (ReadWriteMany), populated via `model-cache/`.

## Quick Start

### 1. Namespace + HF secret
```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN="your-token" -n ${NAMESPACE}
```
> [!NOTE]
> If the namespace lacks nvcr pull access:
> ```bash
> kubectl create secret docker-registry nvcr-secret \
>   --docker-server=nvcr.io --docker-username='$oauthtoken' \
>   --docker-password="<your-NGC-API-key>" -n ${NAMESPACE}
> ```

### 2. Storage
> [!NOTE]
> Edit `model-cache/model-cache.yaml` — set `storageClassName` to a ReadWriteMany class on your cluster.
```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
```

### 3. Download the model
```bash
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=7200s
```

### 4. Deploy the DGD
```bash
kubectl apply -f vllm/agg-h200/deploy.yaml -n ${NAMESPACE}
```
Scale to a full node by editing `spec.components[VllmWorker].replicas` to `8`.

### 5. Benchmark
See [perf/README.md](perf/README.md) — mooncake agentic trace replay, and the
round_robin-vs-kv router comparison.

## Optimization targets

| Workload | Median ISL | Median OSL | KV cache hit rate | User output tok/s |
| -------- | ---------- | ---------- | ----------------- | ----------------- |
| Agentic  | 64k        | 400        | 90%               | 50                |

## Performance results

**KV routing vs round_robin** — tp1 + MTP(nst=3), 2 workers, agentic 15% mooncake trace
(3,541 reqs, block 512, concurrency 8), Dynamo `1.3.0` on H200. MTP forced to the
**measured acceptance length AL=2.925** (see "Speculative decoding" below). Both runs
identical except `DYN_ROUTER_MODE`. Each: 3,411 completed / 130 errors (long-context tail).

| Router       | Output tok/s | Req/s | TTFT mean (ms) | ITL (ms) | KV hit rate |
| ------------ | ------------ | ----- | -------------- | -------- | ----------- |
| round_robin  | 640.3        | 0.28  | 10,499         | 11.2     | ~0 (routing off; workers still cache locally) |
| **kv**       | **764.2**    | 0.33  | **4,248**      | 9.3      | **~59%**    |
| **Δ (kv)**   | **+19.4%**   | +18%  | **−59% (2.5× faster)** | −17% | — |

**KV-aware routing is the recommended configuration** — +19.4% throughput and 2.5× lower
TTFT by landing shared-prefix requests on the replica holding the cache. (Per-user decode
is ~8% lower under kv — it concentrates load on the cached replica — a small trade for the
large system-throughput and first-token-latency wins.)

## Speculative decoding (MTP) — measured, not assumed

The MTP acceptance length was **measured on SpeedBench** (qualitative split, real prompts,
real MTP heads), then forced into the throughput benchmark above via
`synthetic_acceptance_length` — the correct methodology for benchmarking spec-decode on
synthetic trace data.

| nst (draft length) | measured AL | SpeedBench tok/s |
| ------------------ | ----------- | ---------------- |
| 1                  | 1.825       | 681              |
| **3 (recipe)**     | **2.937**   | 895              |
| 5                  | 3.518       | 910              |

- **AL(nst=3) = 2.93** — per-position acceptance 0.80 / 0.63 / 0.50, overall acceptance
  rate ~64%. (Cross-validates the 8k-split value of 2.89 within ~1%.) This is the AL forced
  in the router benchmark.
- **nst=3 is the chosen depth.** nst=5 has higher AL and edges nst=3 by <2% on the
  short-ISL SpeedBench split, but its larger draft head costs more KV pool — on the
  64k-context agentic workload (pool-bound) that reverses the thin gain. nst=1 is clearly
  worse.
- To reproduce the AL measurement, see [perf/README.md](perf/README.md).

### Real vs synthetic spec-config (how to benchmark)

Following the repo convention (see `recipes/nemotron-3-super/.../deploy.yaml`), the worker
sources `--speculative-config` from a **ConfigMap** with two keys:

| ConfigMap key | value | use |
| --- | --- | --- |
| `speculative-config` (shipped active) | `{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}` | **production** — real MTP |
| `speculative-config-synthetic` | above + `"rejection_sample_method":"synthetic","synthetic_acceptance_length":2.925` | **benchmark only** — forced measured AL for synthetic traces (mooncake) |

To reproduce the throughput numbers, set the worker env `SPECULATIVE_CONFIG`
`configMapKeyRef.key` to `speculative-config-synthetic`, then run the mooncake benchmark
(perf/README.md). **Never ship the synthetic key** — production serves real traffic with
real acceptance.

## Notes

- **`--max-num-seqs` must stay ≤ 228.** The Mamba/DeltaNet SSM cache is block-allocated
  at TP1; the vLLM default (`1024`) crashes at startup. The recipe ships `128`.
- **`--kv-cache-dtype auto` (BF16 KV) is intentional** — the FP8 checkpoint ships no KV
  scales, and BF16 KV also measured best for this architecture.
- **MTP needs `--gpu-memory-utilization 0.95`** at the full 262k context (the draft head
  needs the headroom). For a non-speculative deploy, drop `--speculative-config` and set
  utilization back to `0.92`.
- **Benchmarking MTP on synthetic data:** spec-decode performance on synthetic AIPerf
  traces is not representative — use the `speculative-config-synthetic` ConfigMap key
  (measured `synthetic_acceptance_length:2.925`) for representative numbers, per "Real vs
  synthetic spec-config" above. Do **not** ship the synthetic key.
- **ConfigMap/`moe_backend` note.** The tp1+MTP serving config was verified end-to-end;
  the ConfigMap-env indirection and `"moe_backend":"triton"` field follow the Nemotron-3-Super
  reference recipe (TRITON is what tp1 FP8 auto-selects here anyway) — smoke-test on first deploy.
- **Runtime version.** Manifest pins `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0`.
