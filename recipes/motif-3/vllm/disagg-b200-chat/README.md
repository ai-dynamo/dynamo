<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Motif-3 disaggregated chat

These experimental variants separate prefill and decode using NIXL. Each worker
uses the aggregated recipe's Motif runtime image, TP2 with expert parallelism,
DP1, BF16 activations, NVFP4 weights, FP8 KV cache, block size 128,
262,144-token context, 0.85 GPU memory utilization, asynchronous scheduler,
and MTP2. The frontend retains the Motif chat processor, tool and reasoning
parsers, and round-robin routing. Prefix caching is enabled on both roles.

| Variant | Prefill workers | Decode workers | Total B200 GPUs |
|---|---:|---:|---:|
| `1p1d` | 1 × TP2 | 1 × TP2 | 4 |
| `2p1d` | 2 × TP2 | 1 × TP2 | 6 |

Both roles use `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0-motif-3-dev.1`.
They must have matching model, block size, KV dtype, and speculative settings.
The prefill worker is a NIXL producer; the decode worker is a NIXL consumer.
The Nscale variants request one `rdma/shared_ib` device per worker and use UCX
with InfiniBand and CUDA transports. Generic variants leave fabric and node
selection to the deployment environment.

`NCCL_NET_PLUGIN=none` prevents the image's HPC-X NCCL plugin from loading
a UCX version that conflicts with NIXL's bundled UCX. TP2 collectives stay
within each node; NIXL independently uses InfiniBand for the KV transfer.

The pinned image needs two guarded NIXL compatibility fixes for Motif: validate
MLA cache dimensions using the attention backend, and preserve NHD for the mixed
MLA/standard-attention model. `kustomize/base/launch_motif_disagg.py` applies
these fixes in each worker before startup and rejects unexpected source hashes.
The patch preserves global TP replication behavior and supports homogeneous TP2.
It is mounted from a ConfigMap; no replacement image is built. Validation of
end-to-end transfer and performance is still in progress.

Prefill sets `DYN_HEALTH_CHECK_ENABLED=false`: standalone prefill canaries
retain KV blocks without a decode consumer until their NIXL lease expires,
which prevents a clean cache reset. Engine monitoring and Kubernetes probes
remain active; verify serving with a completion through the frontend. After
smoke tests or benchmark traffic, retry cache reset until each worker reports
success, allowing pending transfer leases to drain before measurement.

## Deploy

Populate the existing `shared-model-cache` PVC as described in the
[model recipe](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/recipes/model-recipes/motif-3.mdx).
Frontend and workers use Hugging Face offline mode.

From the repository root, choose one variant:

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply \
  -f recipes/motif-3/vllm/disagg-b200-chat/deploy-nscale-1p1d.yaml

# Change the existing deployment to two prefill workers:
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply \
  -f recipes/motif-3/vllm/disagg-b200-chat/deploy-nscale-2p1d.yaml
```

The variants share the DGD name `motif3-disagg-b200`, so applying the second
changes the same deployment. The frontend Service is
`motif3-disagg-b200-frontend:8000`. Wait for every worker and verify a chat
completion through that frontend before benchmarking.

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

Replay the complete chat 15% trace at concurrency **8, 16, and 32** for each
layout. The supplied trace contains 1,805 rows; do not take another 15% sample.
Its SHA-256 is
`b1221bca72b69f842897f339624306a84857f1b55ea0d866525f94d9ceb9b871`.
Use the [AIPerf configuration](../../perf/disagg.yaml), one concurrency
per run, and a separate artifact directory for each case. Run a separate
warmup before measurement. Clear all prefill and decode KV caches after
warmup and between independent cases, after requests have drained. The
round-robin frontend does not maintain a KV-placement index.

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
`kustomize/components/` for worker counts and Nscale scheduling. The four
checked-in manifests and public overlays are generated from the matrix:

```bash
python3 scripts/kustomize-matrix.py unfold recipes/motif-3/vllm/disagg-b200-chat/.kustomize-matrix.yaml
python3 scripts/kustomize-matrix.py render recipes/motif-3/vllm/disagg-b200-chat/.kustomize-matrix.yaml
python3 scripts/kustomize-matrix.py check recipes/motif-3/vllm/disagg-b200-chat/.kustomize-matrix.yaml
```
