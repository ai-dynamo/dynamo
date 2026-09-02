<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A.X-K2-NVFP4 recipe

This recipe serves [`skt/A.X-K2-NVFP4`](https://huggingface.co/skt/A.X-K2-NVFP4)
with Dynamo and vLLM on B200 GPUs. It is an aggregated deployment with two
independent TP4 workers behind the Dynamo KV-aware router.

## Configuration

| Setting | Value |
| --- | --- |
| GPU | 8x B200 total; 4x B200 per worker |
| Topology | Aggregated, two worker replicas |
| Parallelism | TP4, DP1, expert parallel disabled |
| Weight precision | NVFP4 |
| KV-cache precision | `auto` (BF16 for this checkpoint/backend) |
| Attention | `FLASHINFER_MLA_SPARSE` |
| Speculative decoding | `skt/A.X-K2-EAGLE3`, EAGLE3, k=3 |
| Routing | KV-aware, vLLM prefix-cache events, 32-token blocks |
| Context | 262,144 tokens |
| Reasoning parser | `deepseek_v3` |
| Tool-call parser | `hermes` |

Two workers are intentional: one TP4 worker can prove that KV events are
published, but the router needs at least two candidates to exercise KV-aware
placement. Expert parallelism is deliberately disabled because the selected
four-GPU profile is TP4/DP1 without EP.

The deployment uses the immutable A.X-K2 development image built from Dynamo
1.4.1 and the stock vLLM 0.26.0 base. The Dynamo runtime build overlays the
A.X-K2 model port, DSpark anchor-layout fix, and sparse-MLA/SWA KV-allocation
fix as source-only patches. It deliberately excludes the native FP8 DS-MLA
scale-writer patch and does not rebuild vLLM. This recipe keeps
`--kv-cache-dtype auto`, so the deployment remains on the validated BF16-KV
path and does not require that native FP8 KV-cache change.

The EAGLE3 draft is revision-pinned and loaded through vLLM's real speculative
decoder with `num_speculative_tokens=3`. This is not the synthetic-acceptance
benchmark mode. The draft is trained for A.X-K2's 262,144-token context and
must not be paired with a different target architecture or RoPE configuration.

## Prerequisites

1. Install the Dynamo Kubernetes Platform by following the
   [Kubernetes deployment guide](../../docs/fern/kubernetes/quickstart.mdx).
2. Use a cluster with at least eight schedulable B200 GPUs.
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

## Benchmark

The benchmark recipe replays the 8K/1K, 70%-KV-reuse Mooncake no-schedule
trace at concurrency 32 with AIPerf 0.12.0. See
[`perf/README.md`](perf/README.md).

## Implementation notes

- Frontend and worker block sizes must match. On B200,
  `FLASHINFER_MLA_SPARSE` advertises 32 and 64 as supported block sizes, so
  vLLM 0.26.0 automatically selects 32 when the worker does not pass
  `--block-size`. The frontend is explicitly set to 32 because its independent
  default is 16; leaving it unset would break KV-prefix hash alignment.
- The offline worker needs only `HF_HOME=/model-cache` and
  `HF_HUB_OFFLINE=1` for model-cache resolution. `HF_HUB_CACHE` is derived from
  `HF_HOME`; Xet download tuning, Transformers' legacy offline flag, explicit
  Triton/vLLM/module cache paths, and `PYTHONHASHSEED` are unnecessary here.
- `--enable-prefix-caching` and an explicit KV-events configuration are both
  required. Enabling only one does not give the router authoritative worker
  cache state.
- Async scheduling and CUDA graphs remain enabled. `--enforce-eager` and
  `--no-async-scheduling` are intentionally absent.
- EAGLE3 uses the production acceptance path with three proposed tokens per
  step. The draft revision is pinned to
  `24958e91737d760908f73a8af4b6e06080fc5c1d`.
- Each TP4 worker requests 400 GiB of host memory, matching the proven A.X-K2
  TP4 cluster deployment while allowing both replicas to schedule.
- This branch-specific image is experimental and hosted under `nvstaging`.
  Replace it with the corresponding released runtime image when the A.X-K2
  patches land in a Dynamo release.
