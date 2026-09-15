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

## Configurations

Dynamo + vLLM deployment profiles for the B200 chat workload:

|                          | B200 aggregated chat                         | B200 disaggregated chat                                      |
| ------------------------ | -------------------------------------------- | ------------------------------------------------------------ |
| **Recipe**               | [`vllm/agg-b200-chat`](vllm/agg-b200-chat/deploy-generic.yaml) | [`vllm/disagg-b200-chat`](vllm/disagg-b200-chat/deploy-generic.yaml) |
| **GPU**                  | 4x B200                                      | 4x B200 prefill + 4x B200 decode                             |
| **Mode**                 | Aggregated                                   | Prefill/decode disaggregated, 1P1D                           |
| **Framework**            | vLLM 0.28.0                                  | vLLM 0.28.0                                                  |
| **Precision**            | NVFP4 (W4A4) + FP8 KV                        | NVFP4 (W4A4) + FP8 KV                                        |
| **Parallelism**          | TP4                                          | TP4 prefill / TP4 decode                                     |
| **MoE backend**          | FLASHINFER_CUTLASS (mandatory)               | FLASHINFER_CUTLASS (mandatory)                               |
| **Expert parallel**      | Off (measured within noise at TP4)           | Off                                                          |
| **Speculative decoding** | MTP, `exaone_moe_mtp` DL=2                   | MTP, `exaone_moe_mtp` DL=2 — **both roles**                  |
| **Block size**           | 64                                           | 64                                                           |
| **Max num seqs**         | 32                                           | 32 prefill / **256 decode**                                  |
| **Max batched tokens**   | 8,192                                        | 8,192                                                        |
| **GPU memory util**      | 0.93                                         | 0.93                                                         |
| **Weight loading**       | safetensors (engine default)                 | safetensors (engine default)                                 |
| **Context length**       | 262,144 (model native)                       | 262,144 (model native)                                       |
| **Prefix caching**       | On (vLLM default)                            | On (vLLM default)                                            |
| **Routing**              | KV-aware                                     | KV-aware                                                     |
| **KV transfer**          | N/A                                          | NIXL/UCX over InfiniBand RDMA (`rc_x`/`rc`)                  |
| **KV cache offloading**  | None                                         | None                                                         |

Spec-dec and block size **must match across prefill and decode** — a mismatch changes KV block
geometry and produces silent garbage output rather than an error. `--max-num-seqs` is the
deliberate exception.

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
| KV-aware routing | ✅ | wired end to end (`--router-kv-events` + worker `--kv-events-config`); inert at the shipped `replicas: 1`, correct when you scale out. See Limitations. |
| 262,144-token context | ✅ | the checkpoint's native window; served in full, no rope scaling |
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
kubectl apply -f vllm/agg-b200-chat/deploy-generic.yaml -n ${NAMESPACE}

# 8-GPU disaggregated (1P1D) -- edit the rdma/ resource name first, see Limitations
kubectl apply -f vllm/disagg-b200-chat/deploy-generic.yaml -n ${NAMESPACE}
```

First start takes **40–120 minutes**: 53 shards load, then autotune, then CUDA-graph capture.
Silence is not a hang — watch the worker log for shard progress.

`deploy-generic.yaml` is a **generated** file: it is the render of
`kustomize/base` through the variant matrix in `.kustomize-matrix.yaml`. Apply it directly, or
edit a copy. Contributors edit `kustomize/base/deploy.yaml` and regenerate from the repo root:

```bash
scripts/kustomize-matrix.py unfold recipes/k-exaone-2.0/vllm/<variant>/.kustomize-matrix.yaml
scripts/kustomize-matrix.py render recipes/k-exaone-2.0/vllm/<variant>/.kustomize-matrix.yaml
```

To bind either variant to a specific cluster — a different model-cache claim, node labels and
taints, a scheduler name, or the physical RDMA resource below — copy the cluster scaffold in
[`recipes/templates/kustomize`](../templates/kustomize) rather than editing the recipe. That
layer is where non-portable values belong, and it keeps this base applicable elsewhere.

```bash
# aggregated
kubectl wait --for=condition=Ready dgd/k-exaone-2-agg -n ${NAMESPACE} --timeout=7200s
kubectl logs -f -l nvidia.com/dynamo-component=Worker -n ${NAMESPACE}

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

### 6. Benchmark

```bash
kubectl apply -f perf/perf.yaml -n ${NAMESPACE}
```

See [`perf/README.md`](perf/README.md) for staging the trace, running a concurrency sweep, and
fetching artifacts.

## Performance results

8x B200, vLLM 0.28.0 / Dynamo 1.4.1. **Mooncake chat trace replay**, the 15% subset
(1,805 requests) shipped in [`perf/`](perf/) -- the same file the benchmark recipe runs.

SLA gate is joint: **E2E >= 50 tok/s/user AND TTFT p50 < 5 s**, where
`E2E = OSL / (TTFT_p50 + OSL x ITL)`. Results are at the **highest concurrency meeting both
legs**, not peak throughput.

| Configuration | Concurrency | tok/s/GPU | E2E tok/s/user | TTFT p50 | ITL |
|---|---|---|---|---|---|
| Aggregated (4 GPU) | 7 | **87** | 51.4 | 291 ms | 19.15 ms |
| Disaggregated (8 GPU) | 14 | **85** | 54.9 | 2,230 ms | 15.98 ms |

Each row is that configuration's own operating point, which is why the concurrencies differ;
tok/s/GPU is what makes them comparable, not a matching concurrency.

**The two topologies are level** -- 85 against 87 tok/s/GPU under the same gate. Disaggregation
is an SLA and scaling choice for this model, not a throughput win or loss: it buys per-token
latency (ITL 15.98 ms against 19.15) and lets prefill and decode scale independently, for twice
the GPUs.

Concurrency is the latency/throughput knob within a target. Running the disaggregated recipe at
C=7 instead trades throughput for interactivity: 55 tok/s/GPU, but E2E 70.6 tok/s/user and
TTFT p50 1,018 ms.

Aggregated on the **full** 12,031-request trace, for reference: **97 tok/s/GPU** at C=8,
E2E 51.1, TTFT 312 ms.

Measured KV reuse is **8.8%** on the full trace. That is not a misconfiguration: the working set
is ~42x oversubscribed against TP=4's KV capacity, so blocks are evicted before they can be hit.
It is also why KV-aware routing shows no gain here (9.494% hit rate vs 9.458% round-robin) --
the router cannot route to a block that is already gone.

**38 of the trace's 1,805 requests are rejected, by design, in every run.** The trace is
multi-turn and its accumulated prompts reach 614,440 tokens, while this checkpoint declares
`max_position_embeddings` of 262,144, so those requests return HTTP 400 and are excluded from
the metrics. The count is a fixed property of the trace against this model, identical at every
concurrency, so the rows above remain comparable to each other -- but the throughput figures are
computed over the requests that a 262k-context model can actually serve. The C=14 run completed
1,733 of 1,805: 38 over-length, plus 34 that returned no content under load.

> [!WARNING]
> Synthetic benchmarks with a shared system prompt report ~3x higher throughput for this model
> (68.2% achieved KV reuse vs the trace's 8.8%). Do not compare synthetic and trace numbers.

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

**Weight loading: do not add `--load-format fastsafetensors` without GPUDirect Storage.** It is
tempting on a 53-shard, ~530 GB checkpoint, but fastsafetensors' fast path is GDS, and where the
GDS pieces are missing it does not fall back cleanly — it stalls before opening a single shard.
Observed on a cluster whose `model-cache` PVC is NFS-backed, with no `/dev/nvidia-fs*` device and
no `libcufile.so` in the image: all four ranks spun at ~0.8 core for 13 minutes with zero file
descriptors open on the mount and only the CUDA context resident on the GPUs. Removing the flag —
the only variable changed — restored a normal start. Add it only where the nvidia-fs driver and
cuFile are both present and the filesystem supports GDS.

**Evaluating this model.** Use temperature 0.6 / top_p 0.95, not greedy — `temperature=0`
drives this reasoning model into repetition. On `lm-eval`, GPQA needs `--system_instruction` to
repair an answer extractor that otherwise discards valid responses, and IFEval must **not** get
one: injecting "end your response with X" collides with IFEval's own instructions and corrupts
the measurement. Give a generous token budget — reasoning consumes it before the final answer,
so a short cap reads as a wrong answer rather than a truncated one.

**`--enable-prompt-tokens-details` is a `vllm serve` flag** and is rejected by `dynamo.vllm`. Read
achieved KV reuse from the worker log instead:

```bash
kubectl logs <worker> -n ${NAMESPACE} | grep -o 'Prefix cache hit rate: [0-9.]*%'
```

## Limitations

- **The RDMA resource name is cluster-specific.** The disaggregated recipe requests
  `rdma/shared_ib`; other clusters expose `rdma/ib` or `rdma/rdma_shared_device_a`. Edit it to
  match your device plugin. Prefer a **shared** flavour: with an exclusive-mode resource, two
  co-located workers each claiming HCAs can deadlock NCCL bootstrap in whichever initialises
  second. A Kustomize `provider-networking` Component is the portable answer and is planned.
- **Verify the KV transport before trusting any disaggregated measurement.** `UCX_TLS` here
  excludes `tcp` on purpose, but if the `rdma/` resource name does not match your cluster the
  HCAs are never exposed to the pod and UCX has no fast transport to select. With `tcp` in the
  list it will silently stage GPU memory through host RAM at roughly two orders of magnitude
  lower bandwidth, and the only symptom is a large TTFT. Check:
  ```bash
  kubectl exec <decode-worker> -n ${NAMESPACE} -- \
    curl -s localhost:9090/metrics | grep vllm:nixl_xfer_time_seconds
  ```
  Divide `_sum` by `_count`: a ~1 GB KV transfer should take **milliseconds, not seconds**. To see
  the transport UCX actually chose, redeploy with `UCX_PROTO_INFO=y` and look for `rc_mlx5` rather
  than `tcp/eth0` in the worker log.
- **KV-aware routing does nothing until you scale out — but scaling out is now all it takes.**
  The frontend ships `--router-mode kv --router-kv-events` and the worker (prefill, in the
  disaggregated recipe) ships `--kv-events-config`, so the block index is published and consumed.
  Both halves are still inert at `replicas: 1`, because one worker is one destination — the 8.8%
  figure below is prefix caching, not routing. Raising `replicas` is sufficient; you do not need
  to edit any flag.
  Measured on a 2 x TP4 deployment on this workload, that is not worth doing for throughput: KV
  routing reached a 9.494% prefix hit rate against round-robin's 9.458%, a 0.037 pp difference,
  and aggregate throughput was identical (623 vs 623 tok/s, 847 vs 845). KV routing did show
  8-24% lower TTFT, but with equal hit rates that cannot be cache-driven, so it is not evidence
  of prefix affinity -- and TTFT is not the binding SLA leg here (291 ms against a 5 s gate).
  The working set is ~42x oversubscribed against TP4's KV capacity, so blocks are evicted before a
  router could exploit them — a router cannot route to a block that is already gone.

- **No cache-clearing endpoint.** `VLLM_SERVER_DEV_MODE` is a development flag and is not shipped,
  so `POST /reset_prefix_cache` returns 404. To reproduce the benchmark numbers, restart the
  deployment between measurement points rather than clearing the cache in place.

## Model card

<https://huggingface.co/LGAI-EXAONE/K-EXAONE-2.0-750B-A37B-NVFP4>
