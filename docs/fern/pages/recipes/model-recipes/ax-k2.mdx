---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: A.X-K2
subtitle: Serve A.X-K2 with Dynamo and vLLM on B200 or H200, aggregated or disaggregated.
---
import { RecipeStyles } from "@/components/RecipeStyles";

<RecipeStyles />

Deploy SK Telecom's [A.X-K2](https://huggingface.co/skt/A.X-K2) with
NVIDIA Dynamo and vLLM. B200 targets use NVFP4 weights; H200 targets use FP8
weights. H200 targets use BF16 KV cache; B200 targets use FP8 KV cache.
All use EAGLE3 with three speculative tokens.

<div className="dynamo-target-picker">

<p className="dynamo-target-picker-title">Choose your deployment target</p>

<div className="dynamo-target-picker-row">

<span className="dynamo-target-picker-dim">GPU</span>

<input type="radio" id="recipe-sku-b200" name="recipe-sku" value="b200" defaultChecked />

<label htmlFor="recipe-sku-b200">B200</label>

<input type="radio" id="recipe-sku-h200" name="recipe-sku" value="h200" />

<label htmlFor="recipe-sku-h200">H200</label>

</div>

<div className="dynamo-target-picker-row">

<span className="dynamo-target-picker-dim">Topology</span>

<input type="radio" id="recipe-variant-agg" name="recipe-variant" value="agg" defaultChecked />

<label htmlFor="recipe-variant-agg">Aggregated</label>

<input type="radio" id="recipe-variant-disagg" name="recipe-variant" value="disagg" />

<label htmlFor="recipe-variant-disagg">Disaggregated</label>

</div>

<div className="dynamo-target-picker-summary" data-sku="b200" data-variant="agg">

<span><b>Checkpoint</b> skt/A.X-K2-NVFP4</span>
<span><b>Hardware</b> 8x B200, two TP4 aggregate workers</span>
<span><b>Precision</b> NVFP4 weights, FP8 KV cache</span>
<span><b>Speculation</b> EAGLE3, 3 tokens</span>
<span><b>FlashInfer autotuning</b> Disabled</span>

</div>

<div className="dynamo-target-picker-summary" data-sku="b200" data-variant="disagg">

<span><b>Checkpoint</b> skt/A.X-K2-NVFP4</span>
<span><b>Hardware</b> 12x B200, two TP4 prefill workers and one TP4 decode worker</span>
<span><b>Precision</b> NVFP4 weights, FP8 KV cache</span>
<span><b>Speculation</b> EAGLE3, 3 tokens on both roles</span>
<span><b>FlashInfer autotuning</b> Disabled on prefill and decode</span>

</div>

<div className="dynamo-target-picker-summary" data-sku="h200" data-variant="agg">

<span><b>Checkpoint</b> skt/A.X-K2</span>
<span><b>Hardware</b> 32x H200, four TP8 aggregate workers, expert parallelism enabled</span>
<span><b>Precision</b> FP8 weights, BF16 KV cache</span>
<span><b>Speculation</b> EAGLE3, 3 tokens, draft KV dtype auto</span>

</div>

<div className="dynamo-target-picker-summary" data-sku="h200" data-variant="disagg">

<span><b>Checkpoint</b> skt/A.X-K2</span>
<span><b>Hardware</b> 32x H200, two TP8/DP1 prefill workers and two TP8 decode workers, expert parallelism enabled</span>
<span><b>Precision</b> FP8 weights, BF16 KV cache</span>
<span><b>Speculation</b> EAGLE3, 3 tokens, draft KV dtype auto</span>

</div>

</div>

## Prerequisites

- A Kubernetes cluster with the [Dynamo platform](../../kubernetes/getting-started/quickstart.mdx) installed.
- A ReadWriteMany storage class for model weights.
- A Hugging Face token with access to `skt/A.X-K2`, `skt/A.X-K2-NVFP4`, and `skt/A.X-K2-EAGLE3`.

<div data-sku="b200">

Each worker needs four B200 GPUs and 400 GiB of host memory: eight GPUs total
for aggregated serving or twelve for disaggregated serving.

</div>

<div data-sku="h200">

Each target needs four workers with eight H200 GPUs each (32 GPUs total).
Each worker requests 512 GiB of host memory with a 768 GiB limit.
Use a cluster overlay to bind H200 scheduling and expose your RDMA devices
and network interfaces; the generic manifests do not supply site bindings.

<Note>

For H200 results, use the benchmark configuration described below.

</Note>

</div>

<div data-sku="b200" data-variant="disagg">

Disaggregated serving also requires InfiniBand with one `rdma/shared_ib`
device resource available per worker for NIXL/UCX KV transfer.

</div>

## Deploy

Run commands from the repository root. Set `CONTEXT` and `NAMESPACE` to your
cluster context and namespace, and set `HF_TOKEN` to your Hugging Face token.
Create the namespace and token secret:

```bash
kubectl --context "${CONTEXT}" create namespace "${NAMESPACE}"
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="${HF_TOKEN}"
```

Set `storageClassName` in
[model-cache.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/ax-k2/model-cache/model-cache.yaml)
to your cluster's ReadWriteMany storage class. Create the PVC and download
the three pinned checkpoints:

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply -f recipes/ax-k2/model-cache/model-cache.yaml
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply -f recipes/ax-k2/model-cache/model-download.yaml
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" wait --for=condition=Complete \
  job/axk2-model-download --timeout=14400s
```

All profiles and the download job mount `model-cache`. To use an existing
populated PVC, set `claimName` in the job and selected deployment source to
that PVC and skip creating a new claim. Regenerate the selected manifest
after editing its Kustomize source.

<div data-sku="b200" data-variant="agg">

Deploy two aggregate workers:

```bash
export DGD=axk2-agg-b200-chat
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply \
  -f recipes/ax-k2/vllm/agg-b200-chat/deploy-generic.yaml
```

The manifest is generated from
[Kustomize sources](https://github.com/ai-dynamo/dynamo/tree/main/recipes/ax-k2/vllm/agg-b200-chat/kustomize).
To change a setting, edit `kustomize/base/deploy.yaml` and regenerate:

```bash
python3 scripts/kustomize-matrix.py unfold recipes/ax-k2/vllm/agg-b200-chat/.kustomize-matrix.yaml
python3 scripts/kustomize-matrix.py render recipes/ax-k2/vllm/agg-b200-chat/.kustomize-matrix.yaml
```

</div>

<div data-sku="b200" data-variant="disagg">

Deploy two prefill workers and one decode worker:

```bash
export DGD=axk2-disagg-b200-chat
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply \
  -f recipes/ax-k2/vllm/disagg-b200-chat/deploy-generic.yaml
```

The manifest is generated from
[Kustomize sources](https://github.com/ai-dynamo/dynamo/tree/main/recipes/ax-k2/vllm/disagg-b200-chat/kustomize).
To change a setting, edit `kustomize/base/deploy.yaml` and regenerate:

```bash
python3 scripts/kustomize-matrix.py unfold recipes/ax-k2/vllm/disagg-b200-chat/.kustomize-matrix.yaml
python3 scripts/kustomize-matrix.py render recipes/ax-k2/vllm/disagg-b200-chat/.kustomize-matrix.yaml
```

</div>

<div data-sku="h200" data-variant="agg">

Deploy four aggregate workers after applying your cluster bindings to the
[Kustomize sources](https://github.com/ai-dynamo/dynamo/tree/main/recipes/ax-k2/vllm/agg-h200/kustomize)
and regenerating the manifest:

```bash
python3 scripts/kustomize-matrix.py unfold recipes/ax-k2/vllm/agg-h200/.kustomize-matrix.yaml
python3 scripts/kustomize-matrix.py render recipes/ax-k2/vllm/agg-h200/.kustomize-matrix.yaml
export DGD=axk2-vllm-agg-h200
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply \
  -f recipes/ax-k2/vllm/agg-h200/deploy-generic.yaml
```

</div>

<div data-sku="h200" data-variant="disagg">

Deploy two prefill workers and two decode workers after applying your cluster bindings to the
[Kustomize sources](https://github.com/ai-dynamo/dynamo/tree/main/recipes/ax-k2/vllm/disagg-h200/kustomize)
and regenerating the manifest:

```bash
python3 scripts/kustomize-matrix.py unfold recipes/ax-k2/vllm/disagg-h200/.kustomize-matrix.yaml
python3 scripts/kustomize-matrix.py render recipes/ax-k2/vllm/disagg-h200/.kustomize-matrix.yaml
export DGD=axk2-vllm-disagg-h200
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply \
  -f recipes/ax-k2/vllm/disagg-h200/deploy-generic.yaml
```

</div>

## Smoke Test

Wait for the selected deployment, then forward its frontend port:

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" wait --for=condition=Ready \
  "dynamographdeployment/${DGD}" --timeout=7200s
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" port-forward \
  "service/${DGD}-frontend" 8000:8000
```

<div data-sku="b200">

In another terminal, verify model discovery and a completion:

```bash
curl --fail http://localhost:8000/v1/models
curl --fail http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"skt/A.X-K2-NVFP4","messages":[{"role":"user","content":"What is 2 + 2?"}],"temperature":0,"max_tokens":4096,"stream":false}'
```

Model discovery should list `skt/A.X-K2-NVFP4`; the completion should contain
a `choices` array. Reasoning uses the `deepseek_v3` parser, and tool calling
uses the `hermes` parser.

</div>

<div data-sku="h200">

In another terminal, verify model discovery and a completion:

```bash
curl --fail http://localhost:8000/v1/models
curl --fail http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"A.X-K2","messages":[{"role":"user","content":"What is 2 + 2?"}],"temperature":0,"max_tokens":4096,"stream":false}'
```

Model discovery should list `A.X-K2`; the completion should contain
a `choices` array. Reasoning uses the `deepseek_v3` parser, and tool calling
uses the `hermes` parser.

</div>

## Benchmark

<div data-sku="b200">

The [AIPerf workflow](https://github.com/ai-dynamo/dynamo/blob/main/recipes/ax-k2/perf/README.md)
replays an 8K-input / 1K-output chat trace with 70% KV reuse against the
aggregated profile at concurrency 32. It preserves expanded configurations,
raw reports, and frontend metrics. Keep the default real EAGLE3 acceptance
configuration for accuracy evaluation.

Before starting the benchmark, follow the workflow's
[trace-staging instructions](https://github.com/ai-dynamo/dynamo/blob/main/recipes/ax-k2/perf/README.md#stage-the-trace)
to copy the Git LFS chat trace from your checkout onto the model-cache PVC.
The job reads it through `TRACE_FILE` and verifies its SHA-256 and request
counts before sending traffic.

<div data-variant="disagg">

The checked-in benchmark job targets the aggregate frontend. To benchmark
the disaggregated profile, update its endpoint and frontend selector to this
deployment before applying the job.

</div>

### Performance Results

Benchmarking uses synthetic EAGLE3 acceptance length 2.12.


| Workload          | Framework | Recipe                 | SKU  | Concurrency | System output tok/s/GPU | User output tok/s (mean) | TTFT P50 (seconds) |
| ----------------- | --------- | ---------------------- | ---- | -----------: | -----------------------: | ------------------------: | ------------------: |
| Chat (15% subset) | vLLM      | Aggregated (2 workers) | B200 | 16          | 76.49                   | 55.83                    | 0.525              |
| Chat (15% subset) | vLLM      | Disaggregated (2P1D)   | B200 | 16          | 61.97                   | 83                       | 3.879              |


</div>

<div data-sku="h200">

Configure the [H200 AIPerf Job](https://github.com/ai-dynamo/dynamo/blob/main/recipes/ax-k2/perf/h200/perf.yaml)
for the running deployment using the
[benchmark instructions](https://github.com/ai-dynamo/dynamo/blob/main/recipes/ax-k2/perf/h200/README.md).

Run with 32 H200 GPUs, `1.4.1-a.x-k2-post.1`, FP8 weights, BF16 KV cache,
`flashinfer_cutlass`, `FLASH_ATTN_MLA_SPARSE`, and EAGLE3 k=3 with real
acceptance. The checked-in AIPerf Job uses different sweep settings from the
measurements below.

| Setting | Value |
| --- | --- |
| Topology | 4×TP8 aggregated or 2P2D TP8 (32 H200 GPUs) |
| Workload | W1: 10,240 shared-prefix + 6,144 unique input; 1,024 output; 32 prefixes |
| Cache / Routing | KV-aware; fresh workers, cache retained within each sweep |
| Engine Limits | 262,144 context; batch 8,192; 64 sequences |
| KV Capacity | 64-token blocks; 465,856 tokens per worker |

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply -f recipes/ax-k2/perf/h200/perf.yaml
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" wait \
  --for=condition=Complete job/axk2-h200-perf --timeout=21600s
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" logs job/axk2-h200-perf -c aiperf
```

### H200 Performance Results

| Recipe | Concurrency | System output tok/s | System output tok/s/GPU | User output tok/s (mean) | TPOT p50 (ms) | TPOT p99 (ms) | TTFT p50 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Aggregated (4 workers) | 16 | 896.09 | 28.00 | 60.84 | 16.87 | 20.87 | 655.40 |
| Disaggregated (2P2D) | 16 | 883.40 | 27.61 | 60.73 | 17.09 | 20.73 | 817.80 |
| Aggregated (4 workers) | 32 | 1,388.63 | 43.39 | 46.72 | 22.27 | 27.16 | 652.25 |
| Disaggregated (2P2D) | 32 | 1,483.08 | 46.35 | 51.14 | 20.27 | 23.58 | 902.83 |
| Aggregated (4 workers) | 64 | 1,941.52 | 60.67 | 33.21 | 31.64 | 42.08 | 678.50 |
| Disaggregated (2P2D) | 64 | 2,141.49 | 66.92 | 38.86 | 26.55 | 36.11 | 1,629.23 |
| Aggregated (4 workers) | 128 | 2,646.13 | 82.69 | 22.59 | 46.99 | 61.10 | 707.32 |
| Disaggregated (2P2D) | 128 | 2,034.02 | 63.56 | 30.73 | 34.22 | 47.21 | 28,194.28 |

#### Recommendation

- **Low concurrency or TTFT-first:** prefer aggregated serving; 2P2D has no throughput gain at c16.
- **Throughput/TPOT-first:** consider 2P2D at c64, with TTFT p50 of 1.63 seconds.
- **Higher concurrency:** prefer aggregated at c128; 2P2D has lower throughput and TTFT p50 of 28.19 seconds.

</div>

## Compare All Targets

### B200


| Setting               | Aggregated                            | Disaggregated                      |
| --------------------- | ------------------------------------- | ---------------------------------- |
| Workers               | 2 aggregate                           | 2 prefill + 1 decode               |
| Total GPUs            | 8x B200                               | 12x B200                           |
| Parallelism           | TP4, DP1, expert parallelism disabled | Same on both roles                 |
| Precision             | NVFP4 weights, FP8 KV cache           | Same on both roles                 |
| Attention             | `FLASHINFER_MLA_SPARSE`               | Same on both roles                 |
| FlashInfer autotuning | Disabled                              | Disabled on prefill and decode     |
| EAGLE3                | 3 speculative tokens                  | 3 speculative tokens on both roles |
| Async scheduling      | Enabled                               | Prefill disabled, decode enabled   |
| Context length        | 262,144 tokens                        | 262,144 tokens                     |
| Routing               | KV-aware                              | KV-aware                           |
| KV transfer           | N/A                                   | NIXL/UCX over InfiniBand           |


### H200


| Setting                  | Aggregated                                  | Disaggregated                                                         |
| ------------------------ | ------------------------------------------- | --------------------------------------------------------------------- |
| Workers                  | 4 aggregate                                 | 2 prefill + 2 decode                                                  |
| Total GPUs               | 32x H200                                    | 32x H200                                                              |
| Parallelism              | TP8 + expert parallelism                    | Prefill TP8/DP1 + expert parallelism; decode TP8 + expert parallelism |
| Checkpoint               | `skt/A.X-K2`, FP8 weights                   | Same                                                                  |
| KV Cache                 | `bfloat16`                                | Same on both roles                                                    |
| EAGLE3                   | 3 speculative tokens, draft KV dtype `auto` | Same on both roles                                                    |
| MoE Backend              | `flashinfer_cutlass`                        | Same on both roles                                                    |
| Attention Backend        | `FLASH_ATTN_MLA_SPARSE`                     | Same on both roles                                                    |
| Max Sequences per Engine | 64                                          | 64 on both roles                                                      |
| Context Length           | 262,144 tokens                              | Same                                                                  |
| Routing                  | KV-aware                                    | KV-aware                                                              |
| KV Transfer              | N/A                                         | NIXL/UCX over RDMA                                                    |


If model initialization runs out of memory, lower `--max-model-len` to 32768
on every worker.

## Source

- [Recipe and configuration table](https://github.com/ai-dynamo/dynamo/tree/main/recipes/ax-k2)
- [Aggregated manifest](https://github.com/ai-dynamo/dynamo/blob/main/recipes/ax-k2/vllm/agg-b200-chat/deploy-generic.yaml)
- [Disaggregated manifest](https://github.com/ai-dynamo/dynamo/blob/main/recipes/ax-k2/vllm/disagg-b200-chat/deploy-generic.yaml)
- [H200 aggregated manifest](https://github.com/ai-dynamo/dynamo/blob/main/recipes/ax-k2/vllm/agg-h200/deploy-generic.yaml)
- [H200 disaggregated manifest](https://github.com/ai-dynamo/dynamo/blob/main/recipes/ax-k2/vllm/disagg-h200/deploy-generic.yaml)
