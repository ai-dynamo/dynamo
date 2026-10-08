<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Motif-3 disaggregated chat

**Experimental.** Deploy **2P1D with KV-aware routing** on six NVIDIA B200 GPUs: two prefill
workers and one decode worker, each using TP2 and expert parallelism. NIXL
transfers the KV cache from prefill to decode. The KV router uses worker cache
events and active load to choose a prefill worker for each request.

The recipe uses a runtime image with the Motif NIXL fixes included and keeps
the aggregated model settings: DP1, BF16 activations,
NVFP4 weights, FP8 KV cache, block size 128, a 262,144-token context,
0.85 GPU memory utilization, asynchronous scheduling, and MTP2. The frontend
uses the Motif chat processor and tool and reasoning parsers.

## Deploy

Use a namespace with a populated `shared-model-cache` PVC as described in the
[model recipe](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/recipes/model-recipes/motif-3.mdx).
The image is hosted in the private `nvcr.io/nvstaging/nim` registry. Configure
image-pull credentials with read access through the namespace's ServiceAccount
or your site Kustomization.

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

## Deployment Targets

The generic and Nscale manifests deploy the same 2P1D KV-aware configuration
and share the DGD name `motif3-disagg-b200`. Applying the other target updates
that deployment.

| Manifest | Prefill workers | Decode workers | Total B200 GPUs |
|---|---:|---:|---:|
| `deploy-generic-2p1d-kv.yaml` | 2 × TP2 | 1 × TP2 | 6 |
| `deploy-nscale-2p1d-kv.yaml` | 2 × TP2 | 1 × TP2 | 6 |

The frontend sets `DYN_ROUTER_MODE=kv` and
`DYN_ROUTER_USE_KV_EVENTS=true`. Both worker roles explicitly
publish KV-cache events through `--kv-events-config`; Dynamo forwards those
events over its ZeroMQ event plane. Each worker has its own pod network
namespace, so they can use the same local event port. All processes use
`PYTHONHASHSEED=0`. Prefix caching is enabled on both roles. In 2P1D, the
prefill router chooses between two workers; there is one decode worker.

## Runtime Compatibility

The manifests pin
`nvcr.io/nvstaging/nim/motif-3:dynamo-db6a3705-disagg-41e2d0b33` by digest.
The [runtime branch](https://github.com/ai-dynamo/dynamo/tree/vanshils/motif-3-disagg-runtime)
is based directly on commit
[`db6a3705`](https://github.com/ai-dynamo/dynamo/commit/db6a3705c1d9a9df4f94324636c454e45dc000aa).
The image includes the Motif NIXL compatibility fixes: cache dimensions
are validated using each attention backend, and Motif retains the NHD layout
expected by its attention kernels. Workers start directly with
`python3 -m dynamo.vllm`. The image supports homogeneous TP2 disaggregation.

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

Record KV-aware routing with every result. Replay the complete chat 15% trace
at concurrency **8, 16, and 32**. The supplied trace contains 1,805 rows; do
not take another 15% sample.
Its SHA-256 is
`b1221bca72b69f842897f339624306a84857f1b55ea0d866525f94d9ceb9b871`.
Use the [AIPerf configuration](../../perf/disagg.yaml), one concurrency
per run, and a separate artifact directory for each case. Run a separate
warmup before measurement. Clear all prefill and decode KV caches after
warmup and between independent cases, after requests have drained. Clear the
router index as well as worker caches before each independent run.

The recipe defaults to actual MTP verification. To reproduce the aggregated
recipe's **benchmark-only synthetic acceptance length 2.13**, select
`speculative-config-synthetic` in the `SPECULATIVE_CONFIG` ConfigMap reference
for **both** worker roles. Report that setting with results; synthetic
acceptance on a chat trace does not measure real MTP acceptance or output
quality. Report context-limit rejections separately from other errors.

Per-GPU system throughput divides by **all six** participating GPUs. Include
TTFT, per-user output rate, total output rate,
request counts, and errors. Deployment and benchmark validation is in progress;
no performance result is claimed by these manifests.

## Edit and render

Edit `kustomize/base/` for serving settings, worker counts, and KV routing.
Edit `kustomize/components/nscale/` for Nscale scheduling and fabric resources.
The two checked-in manifests and public overlays are generated from the matrix:

```bash
python3 scripts/kustomize-matrix.py unfold recipes/motif-3/vllm/disagg-b200-chat/.kustomize-matrix.yaml
python3 scripts/kustomize-matrix.py render recipes/motif-3/vllm/disagg-b200-chat/.kustomize-matrix.yaml
python3 scripts/kustomize-matrix.py check recipes/motif-3/vllm/disagg-b200-chat/.kustomize-matrix.yaml
```
