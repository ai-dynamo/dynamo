<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Motif-3 disaggregated chat

**Experimental.** Deploy **2P1D with KV-aware routing** on six NVIDIA B200 GPUs: two prefill
workers and one decode worker, each using TP2 and expert parallelism. NIXL
transfers the KV cache from prefill to decode. The KV router uses worker cache
events and active load to choose a prefill worker for each request.

The recipe keeps the aggregated image and model settings:
`nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0-motif-3-dev.1`, DP1, BF16 activations,
NVFP4 weights, FP8 KV cache, block size 128, a 262,144-token context,
0.85 GPU memory utilization, asynchronous scheduling, and MTP2. The frontend
uses the Motif chat processor and tool and reasoning parsers.

## Deploy

Use a namespace with a populated `shared-model-cache` PVC as described in the
[model recipe](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/recipes/model-recipes/motif-3.mdx).
Set `CONTEXT` and `NAMESPACE` to your Kubernetes context and namespace, then
apply the self-contained manifest from the repository root:

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply \
  -f recipes/motif-3/vllm/disagg-b200-chat/deploy-nscale-2p1d-kv.yaml
```

The Nscale manifest selects B200 nodes and requests one `rdma/shared_ib`
device per worker. It uses UCX with InfiniBand and CUDA transports. For other
clusters, use `deploy-generic-2p1d-kv.yaml` and configure GPU node selection
and fabric resources for that environment. Frontend and workers use Hugging
Face offline mode.

The frontend Service is `motif3-disagg-b200-frontend:8000`. Wait for all three
workers to become ready, then verify model discovery and a chat completion:

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" port-forward \
  svc/motif3-disagg-b200-frontend 8000:8000
```

In another terminal:

```bash
python3 .agents/skills/dynamo-router-starter/scripts/check_router_health.py \
  --base-url http://127.0.0.1:8000
```

## Variants

Each suffix is available as `deploy-generic-<suffix>.yaml` and
`deploy-nscale-<suffix>.yaml`. The variants share the DGD name
`motif3-disagg-b200`; applying another variant changes that deployment.

| Suffix | Prefill workers | Decode workers | Total B200 GPUs | Routing |
|---|---:|---:|---:|---|
| `2p1d-kv` | 2 × TP2 | 1 × TP2 | 6 | KV-aware, with worker KV events |
| `2p1d` | 2 × TP2 | 1 × TP2 | 6 | Round-robin benchmark baseline |
| `1p1d` | 1 × TP2 | 1 × TP2 | 4 | Round-robin benchmark baseline |

The KV variant sets `DYN_ROUTER_MODE=kv` and
`DYN_ROUTER_USE_KV_EVENTS=true` on the frontend. Both worker roles explicitly
publish KV-cache events through `--kv-events-config`; Dynamo forwards those
events over its ZeroMQ event plane. Each worker has its own pod network
namespace, so they can use the same local event port. All processes use
`PYTHONHASHSEED=0`. Prefix caching is enabled on both roles. In 2P1D, the
prefill router chooses between two workers; there is one decode worker.

## Why the runtime patch is required

**The pinned image fails during disaggregated startup without this patch.**
Motif mixes MLA and standard attention, while the image's NIXL connector
assumes one model-wide cache shape and forces an incompatible memory layout.
[`launch_motif_disagg.py`](kustomize/base/launch_motif_disagg.py) makes two
changes before launching each worker:

- Validate cache dimensions using the actual attention backend.
- Keep the NHD cache layout expected by Motif's attention kernels.

The launcher checks the original source hashes and refuses unknown runtime
versions. The manifest includes it in a ConfigMap mounted at `/motif-runtime`;
you do not need a separate local file or a rebuilt image. This workaround is
limited to the pinned image with TP2 on both roles. Remove it only after an
image includes equivalent fixes and passes disaggregated validation.

`NCCL_NET_PLUGIN=none` separately avoids a conflict between the image's HPC-X
UCX library and NIXL's bundled UCX. TP2 collectives stay within each node;
NIXL uses InfiniBand for cross-node KV transfer.

Prefill sets `DYN_HEALTH_CHECK_ENABLED=false` because standalone prefill
canaries retain KV blocks without a decode consumer until the transfer lease
expires. Engine monitoring and Kubernetes probes remain active. Verify
serving through the frontend and allow outstanding transfers to drain before
resetting caches.

## Benchmark

Use AIPerf 0.13.0 with a separate, tokenizer-only Hugging Face cache. The
client must not load Motif's model configuration from the workers' full model
cache. The helper below copies and verifies the three tokenizer files at the
pinned model revision, then exercises AIPerf's tokenizer loader:

```bash
export HF_HOME=/tmp/motif3-disagg-client-hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 recipes/motif-3/perf/prepare_disagg_client.py
```

Run the benchmark client with these same environment variables. The validated
client dependencies are Transformers 5.19.0, Tokenizers 0.23.2, and
huggingface-hub 1.33.0.

Use the round-robin variants for the aggregated comparison baseline. Record
the routing mode with every result; benchmark the KV variant as a separate
series. Replay the complete chat 15% trace at concurrency **8, 16, and 32** for each
layout. The supplied trace contains 1,805 rows; do not take another 15% sample.
Its SHA-256 is
`b1221bca72b69f842897f339624306a84857f1b55ea0d866525f94d9ceb9b871`.
Use the [AIPerf configuration](../../perf/disagg.yaml), one concurrency
per run, and a separate artifact directory for each case. Run a separate
warmup before measurement. Clear all prefill and decode KV caches after
warmup and between independent cases, after requests have drained. For KV
routing, clear the router index as well as worker caches before each
independent run. The round-robin frontend has no KV-placement index.

The recipe defaults to actual MTP verification. To reproduce the aggregated
recipe's **benchmark-only synthetic acceptance length 2.13**, select
`speculative-config-synthetic` in the `SPECULATIVE_CONFIG` ConfigMap reference
for **both** worker roles. Report that setting with results; synthetic
acceptance on a chat trace does not measure real MTP acceptance or output
quality. Report context-limit rejections separately from other errors.

Per-GPU system throughput divides by **all** participating GPUs: four for
1P1D and six for 2P1D. Include TTFT, per-user output rate, total output rate,
request counts, and errors. Deployment and benchmark validation is in progress;
no performance result is claimed by these manifests.

## Edit and render

Edit `kustomize/base/` for serving settings and
`kustomize/components/` for worker counts, KV routing, and Nscale scheduling.
The six checked-in manifests and public overlays are generated from the matrix:

```bash
python3 scripts/kustomize-matrix.py unfold recipes/motif-3/vllm/disagg-b200-chat/.kustomize-matrix.yaml
python3 scripts/kustomize-matrix.py render recipes/motif-3/vllm/disagg-b200-chat/.kustomize-matrix.yaml
python3 scripts/kustomize-matrix.py check recipes/motif-3/vllm/disagg-b200-chat/.kustomize-matrix.yaml
```
