<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# K-EXAONE 2.0 Recipes

Serving recipes for LG AI Research's **K-EXAONE 2.0 750B-A37B** in NVFP4, on NVIDIA B200
with vLLM via Dynamo.

764.5 B total parameters, ~37 B active per token. 78 layers with **hybrid attention** —
20 `full_attention` and 58 `sliding_attention`. 256 experts, top-8 routing. Native context
262,144 tokens. The NVFP4 (W4A4) checkpoint is ~530 GB on disk.

> [!IMPORTANT]
> **These recipes reference a development image and cannot be run by external users yet.**
> Stock vLLM 0.28.0 cannot load this checkpoint: `exaone_moe.py` omits `prefix=` on
> `ParallelLMHead`, so the checkpoint's quantization exclusion list never matches `lm_head`,
> which is then wrongly quantized and rejects the BF16 weight. The one-line fix is
> [ai-dynamo/dynamo#14599](https://github.com/ai-dynamo/dynamo/pull/14599). Once it merges and
> a public `nvcr.io` tag is cut, the image reference must be swapped to that tag pinned to its
> **multi-arch index digest**, and `imagePullSecrets` plus `runtimeVersionOverride` removed.

## Configurations

| Configuration | GPUs | Shape | File |
|---|---|---|---|
| Aggregated | 4 | TP=4, single worker | [`vllm/agg-b200-chat/deploy.yaml`](vllm/agg-b200-chat/deploy.yaml) |
| Disaggregated | 8 | 1P1D, TP=4 per role | [`vllm/disagg-b200-chat/deploy.yaml`](vllm/disagg-b200-chat/deploy.yaml) |

Both serve the chat workload: ~8K input / 1K output with ~70% prefix reuse.

## Supported features

| Feature | Supported | Notes |
|---|---|---|
| NVFP4 (W4A4) weights | ✅ | FLASHINFER_CUTLASS MoE backend is **mandatory** — see Configuration notes |
| FP8 KV cache | ✅ | the checkpoint declares `kv_cache_quant_algo: FP8`; it is the default path |
| Speculative decoding (MTP) | ✅ | `--spec-method exaone_moe_mtp --spec-tokens 2` |
| Prefix caching | ✅ | on by default |
| Reasoning parser | ✅ | `--dyn-reasoning-parser qwen3` |
| Tool calling | ✅ | `--dyn-tool-call-parser qwen3_coder` |
| Disaggregated serving | ✅ | NIXL over UCX; requires an RDMA device plugin — see Limitations |
| KV-aware routing | ⚠️ | enabled, but no measurable gain on this workload — see Performance results |
| 1M context | ❌ | the checkpoint is 262,144 with unscaled rope |
| Expert parallel | ➖ | measured within noise on TP=4; not enabled |

## Prerequisites

- Kubernetes with the Dynamo operator installed
- **8× NVIDIA B200** (4 for the aggregated recipe)
- A `ReadWriteMany` storage class with ≥700 GB free
- For the disaggregated recipe: an **RDMA device plugin** exposing InfiniBand HCAs as a
  Kubernetes extended resource

## Quick Start

### 1. Create the namespace and HF token secret

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} -n ${NAMESPACE}
```

### 2. Create storage

Edit `storageClassName` in [`model-cache/model-cache.yaml`](model-cache/model-cache.yaml) to
match your cluster, then:

```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
```

### 3. Download the model

```bash
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=complete job/model-download -n ${NAMESPACE} --timeout=6h
```

~530 GB. The Job requests 64 GB of RAM for XET download buffers; set
`HF_XET_HIGH_PERFORMANCE=0` on low-memory nodes.

### 4. Deploy

```bash
# 4-GPU aggregated
kubectl apply -f vllm/agg-b200-chat/deploy.yaml -n ${NAMESPACE}

# 8-GPU disaggregated (1P1D) -- edit the rdma/ resource name first, see Limitations
kubectl apply -f vllm/disagg-b200-chat/deploy.yaml -n ${NAMESPACE}
```

First start takes **40–120 minutes**: 53 shards load, then autotune, then CUDA-graph capture.
Silence is not a hang — watch the worker log for shard progress.

```bash
# aggregated
kubectl wait --for=condition=Ready dgd/k-exaone-2-agg -n ${NAMESPACE} --timeout=7200s
kubectl logs -f -l nvidia.com/dynamo-component=VllmWorker -n ${NAMESPACE}

# disaggregated
kubectl wait --for=condition=Ready dgd/k-exaone-2-disagg -n ${NAMESPACE} --timeout=7200s
kubectl logs -f -l nvidia.com/dynamo-component=DecodeWorker -n ${NAMESPACE}
```

### 5. Smoke test

```bash
# k-exaone-2-disagg-frontend for the disaggregated deployment
kubectl port-forward svc/k-exaone-2-agg-frontend 8000:8000 -n ${NAMESPACE}

curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "LGAI-EXAONE/K-EXAONE-2.0-750B-A37B-NVFP4",
  "messages": [{"role":"user","content":"In 2-3 sentences, explain why the daytime sky is blue."}],
  "max_tokens": 2048, "temperature": 0.6, "top_p": 0.95
}' | jq -r '.choices[0].message.content // .choices[0].message.reasoning_content'
```

> [!NOTE]
> This is a reasoning model. Use **temperature 0.6, not greedy** — `temperature=0` drives it into
> rumination. Give it a real token budget: reasoning consumes the budget before the final answer,
> so a 256-token smoke test prints `null` on a perfectly healthy deployment. Read
> `.content // .reasoning_content`.

## Performance results

8× B200, vLLM 0.28.0 / Dynamo 1.4.1. **Mooncake chat trace replay**, all 12,031 requests.

SLA gate is joint: **E2E ≥ 50 tok/s/user AND TTFT p50 < 5 s**, where
`E2E = OSL / (TTFT_p50 + OSL × ITL)`. Results are at the **highest concurrency meeting both
legs** — not peak throughput.

| Configuration | Concurrency | tok/s/GPU | E2E tok/s/user | TTFT p50 | ITL |
|---|---|---|---|---|---|
| Aggregated (4 GPU) | 8 | **97** | 51.1 | 312 ms | 19.27 ms |

Measured KV reuse on this trace is **8.8%**. That is not a misconfiguration: the trace's working
set is ~42× oversubscribed against TP=4's KV capacity, so blocks are evicted before they can be
hit. It is also why KV-aware routing shows no gain here (9.494% hit rate vs 9.458% for
round-robin) — the router cannot route to a block that is already gone.

Disaggregated trace results are pending re-measurement and will be added before release.

> [!WARNING]
> Synthetic benchmarks with a shared system prompt report ~3× higher throughput for this model
> (68.2% achieved KV reuse vs the trace's 8.8%). Do not compare synthetic and trace numbers.

## Accuracy

Published BF16 reference: GPQA-Diamond **82.2**, IFEval **92.4**.

| Configuration | GPQA-Diamond | IFEval (inst-loose) |
|---|---|---|
| raw `vllm serve`, no spec-dec (control) | 0.7677 ± 0.0301 | 0.9161 |
| **Aggregated** | **0.8131 ± 0.0278** | **0.9113** |
| **Disaggregated** | **0.7778 ± 0.0296** | **0.9293** |

All pairwise comparisons indistinguishable (z ≤ 1.11); the published 82.2 sits inside every
confidence interval. Neither speculative decoding, the Dynamo serving layer, nor the NIXL KV
transfer costs measurable accuracy.

Evaluated at temperature 0.6 / top_p 0.95 with `lm-eval`. GPQA used `--system_instruction` to
repair a broken answer extractor; IFEval deliberately did **not**, because it would collide with
IFEval's own constraints.

## Configuration notes

**`--kernel-config '{"moe_backend":"FLASHINFER_CUTLASS"}'` is mandatory.** vLLM's `auto` selects
`FLASHINFER_TRTLLM`, which on this checkpoint corrupted **32 of 40** long-form generations —
fluent, plausible, wrong, with no error and no crash. Pinning CUTLASS took that to 0/40. The
pinned kernel is slower, and that cost is included in the results above.

**Use the `--dyn-*` parser flags.** Plain `--reasoning-parser` and `--tool-call-parser` configure
the engine only and never reach the Dynamo frontend, so tool calling silently does nothing.

**TP=4 is the floor and the optimum.** ~530 GB of weights does not fit TP=2 on 180 GB B200s. TP=8
has 5.7× the KV capacity (17.05 M vs 2.99 M tokens) and is still ~30% slower per GPU — this model
is communication-bound, not KV-capacity-bound.

**`--block-size 64`.** Block size 16 is measurably worse.

**Disaggregated: prefill and decode must agree** on spec-dec and block size. A mismatch changes KV
block geometry and produces silent garbage output, not an error. `--max-num-seqs` is the exception
and is deliberately different — 32 on prefill, 256 on decode.

**`--enable-prompt-tokens-details` is a `vllm serve` flag** and is rejected by `dynamo.vllm`. Read
achieved KV reuse from the worker log instead:

```bash
kubectl logs <worker> -n ${NAMESPACE} | grep -o 'Prefix cache hit rate: [0-9.]*%'
```

## Limitations

- **Development image only.** See the note at the top.
- **The RDMA resource name is cluster-specific.** The disaggregated recipe requests
  `rdma/shared_ib`; other clusters expose `rdma/ib` or `rdma/rdma_shared_device_a`. Edit it to
  match your device plugin. Prefer a **shared** flavour: with an exclusive-mode resource, two
  co-located workers each claiming HCAs can deadlock NCCL bootstrap in whichever initialises
  second. A Kustomize `provider-networking` Component is the portable answer and is planned.
- **NIXL falls back to TCP silently.** If no fast transport is available, UCX stages GPU memory
  through host RAM at roughly two orders of magnitude lower bandwidth, and the only symptom is a
  large TTFT. Verify before trusting any measurement:
  ```bash
  kubectl exec <decode-worker> -n ${NAMESPACE} -- \
    curl -s localhost:9090/metrics | grep vllm:nixl_xfer_time_seconds
  ```
  A ~1 GB KV transfer should take milliseconds, not seconds.
- **1M context is not supported.** The checkpoint declares `max_position_embeddings = 262144`
  with `rope_type: "default"` (unscaled), and LG's model card states the same. Reaching 1M would
  need a rope-scaling override with unvalidated long-context accuracy.
- **No cache-clearing endpoint.** `VLLM_SERVER_DEV_MODE` is a development flag and is not shipped,
  so `POST /reset_prefix_cache` returns 404. To reproduce the benchmark numbers, restart the
  deployment between measurement points rather than clearing the cache in place.

## Model card

<https://huggingface.co/LGAI-EXAONE/K-EXAONE-2.0-750B-A37B-NVFP4>
