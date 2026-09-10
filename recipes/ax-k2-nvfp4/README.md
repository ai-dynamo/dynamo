<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A.X-K2-NVFP4 recipe

This recipe serves [`skt/A.X-K2-NVFP4`](https://huggingface.co/skt/A.X-K2-NVFP4)
with Dynamo and vLLM on B200 GPUs. It provides two independent TP4 aggregate
workers or a disaggregated deployment with two TP4 prefill workers and one
TP4 decode worker behind the Dynamo KV-aware router. The 2P1D variant disables
async scheduling on prefill and enables it on decode; both roles use EAGLE3
with three speculative tokens.

## Configuration

| Setting | Value |
| --- | --- |
| GPU | 8x B200 for aggregate; 12x B200 for 2P1D; 4x B200 per worker |
| Topology | Two aggregate replicas, or two prefill replicas and one decode replica |
| Parallelism | TP4, DP1, expert parallel disabled |
| Weight precision | NVFP4 |
| KV-cache precision | Ordinary per-tensor FP8 |
| Attention | `FLASHINFER_MLA_SPARSE` |
| Speculative decoding | `skt/A.X-K2-EAGLE3`, EAGLE3, k=3 |
| KV transfer (2P1D) | NIXL, CUDA buffers, UCX over InfiniBand |
| Routing | KV-aware, vLLM prefix-cache events, 64-token blocks |
| Context | 262,144 tokens |
| Reasoning parser | `deepseek_v3` |
| Tool-call parser | `hermes` |

Two aggregate workers are intentional: one TP4 worker can prove that KV events are
published, but the router needs at least two candidates to exercise KV-aware
placement. Expert parallelism is deliberately disabled because the selected
four-GPU profile is TP4/DP1 without EP.

The deployment uses the immutable A.X-K2 development image built from Dynamo
1.4.1 and the stock vLLM 0.26.0 base. The Dynamo runtime build overlays the
A.X-K2 model port, DSpark anchor-layout fix, and sparse-MLA/SWA KV-allocation
fix as source-only patches. It deliberately excludes the native FP8 DS-MLA
scale-writer patch and does not rebuild vLLM. This recipe uses ordinary
`--kv-cache-dtype fp8` with `FLASHINFER_MLA_SPARSE`. That is vLLM's standard
per-tensor FP8 layout, not the packed `fp8_ds_mla` layout, so it does not
require the native FP8 DS-MLA scale-writer patch.

The EAGLE3 draft is revision-pinned and loaded through vLLM's real speculative
decoder with `num_speculative_tokens=3`. Both disaggregated roles load the
same pinned EAGLE3 head so prefill builds the draft KV state needed by decode.
The aggregate deployment selects the `speculative-config` ConfigMap key.
For throughput-only benchmarking, the aggregate
ConfigMap also provides `speculative-config-synthetic`, which keeps the draft
cost but forces a synthetic acceptance length of 2.12 using
`rejection_sample_method: synthetic`. Synthetic mode does not measure output
quality and must not be used as the production default. The draft is trained
for A.X-K2's 262,144-token context and must not be paired with a different
target architecture or RoPE configuration.

## Prerequisites

1. Install the Dynamo Kubernetes Platform by following the
   [Kubernetes deployment guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/kubernetes/getting-started/quickstart.mdx).
2. Use at least eight schedulable B200 GPUs for aggregate or twelve for 2P1D.
   Disaggregated workers also require the `rdma/shared_ib` device resource.
3. Create a Hugging Face secret named `hf-token-secret` with an `HF_TOKEN`
   key in the target namespace.
4. Create an image-pull secret named `nvcr-imagepullsecret` that can read the
   branch-specific `nvcr.io/nvstaging/nim` image.

```bash
export NAMESPACE=your-namespace
kubectl create namespace "${NAMESPACE}"
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n "${NAMESPACE}"

kubectl create secret docker-registry nvcr-imagepullsecret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="your-ngc-api-key" \
  -n "${NAMESPACE}"
```

## Deploy

Edit `model-cache/model-cache.yaml` and select a ReadWriteMany storage class,
then create the cache and download the revision-pinned target and EAGLE3 draft
snapshots. The download Job runs the two `hf download` calls sequentially into
the same `HF_HOME=/model-cache` used by the offline worker:

```bash
kubectl apply -f model-cache/model-cache.yaml -n "${NAMESPACE}"
kubectl apply -f model-cache/model-download.yaml -n "${NAMESPACE}"
kubectl wait --for=condition=Complete job/axk2-model-download \
  -n "${NAMESPACE}" --timeout=14400s
```

Deploy the aggregate recipe:

```bash
kubectl apply -f vllm/agg-b200-chat/deploy.yaml -n "${NAMESPACE}"
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=axk2-agg-b200-chat \
  -n "${NAMESPACE}" --timeout=7200s
```

The frontend service is `axk2-agg-b200-chat-frontend:8000`. Run a smoke test
from inside the cluster or port-forward that service before calling
`/v1/chat/completions` with served model name `skt/A.X-K2-NVFP4`.

For the 2P1D deployment, use the cluster-wide `shared-model-cache` PVC with
both pinned snapshots already present. Set `CONTEXT` to the target Kubernetes
context and deploy the generated manifest:

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply \
  -f vllm/disagg-b200-chat-2p1d/deploy-generic.yaml
```

See [the 2P1D recipe](vllm/disagg-b200-chat-2p1d/README.md) for readiness checks,
the frontend service, and Kustomize authoring instructions.

## Benchmark

The benchmark recipe replays the 8K/1K, 70%-KV-reuse Mooncake no-schedule
trace at concurrency 32 with AIPerf 0.12.0. See
[`perf/README.md`](perf/README.md).

## Implementation notes

- Frontend and worker block sizes must match. The worker does not need an
  explicit `--block-size`: vLLM 0.26.0 automatically selects 64 for AX-K2's
  `DEEPSEEK_V32_INDEXER` backend. The frontend is explicitly set to 64 because
  its independent default is 16; leaving it unset would break KV-prefix hash
  alignment.
- The offline worker needs only `HF_HOME=/model-cache` and
  `HF_HUB_OFFLINE=1` for model-cache resolution. `HF_HUB_CACHE` is derived from
  `HF_HOME`; Xet download tuning, Transformers' legacy offline flag, explicit
  Triton/vLLM/module cache paths, and `PYTHONHASHSEED` are unnecessary here.
- `--enable-prefix-caching` and an explicit KV-events configuration are both
  required. Enabling only one does not give the router authoritative worker
  cache state.
- Async scheduling is enabled on aggregate and decode workers. The 2P1D
  prefill workers explicitly use `--no-async-scheduling`. CUDA graphs remain
  enabled on all workers.
- EAGLE3 uses the production acceptance path with three proposed tokens per
  step. The draft revision is pinned to
  `24958e91737d760908f73a8af4b6e06080fc5c1d`.
- For synthetic aggregate throughput experiments only, change the worker's ConfigMap key
  from `speculative-config` to `speculative-config-synthetic`. That variant
  adds `rejection_sample_method: synthetic` and
  `synthetic_acceptance_length: 2.12`; acceptance length includes the target
  bonus token.
- Each TP4 worker requests 400 GiB of host memory.
- The 2P1D workers set `UCX_RCACHE_MAX_UNRELEASED=1024` before NIXL is
  imported and use `UCX_TLS=rc_x,rc,cuda_copy,cuda_ipc` for KV transfer.
- This branch-specific image is experimental and hosted under `nvstaging`.
  Replace it with the corresponding released runtime image when the A.X-K2
  patches land in a Dynamo release.
