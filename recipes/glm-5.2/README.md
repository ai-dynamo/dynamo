<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.2 NVFP4 Recipe

Serves [`nvidia/GLM-5.2-NVFP4`](https://huggingface.co/nvidia/GLM-5.2-NVFP4)
with SGLang, KV-aware Dynamo routing, NIXL prefill/decode transfer, and SGLang
HiCache host-memory offload on B200.

## Configuration

| Setting | Value |
| --- | --- |
| Workload | Agentic coding and tool use |
| Hardware | 12x B200 across two workers |
| Mode | 1 prefill + 1 decode |
| Framework | SGLang on Dynamo |
| Precision | NVFP4 weights + FP8 KV cache |
| Prefill | TP4 / DP4 / EP4, DSA TRTLLM, `mem-fraction-static=0.93` |
| Decode | TP8 / DP8, FlashInfer Cutlass, `mem-fraction-static=0.87` |
| Context length | 500,000 tokens |
| Routing | KV-aware |
| KV transfer | NIXL over GPU-local UCX/InfiniBand devices |
| KV offload | SGLang HiCache, 200 GB per prefill rank |
| Speculative decoding | Built-in EAGLE, draft length 3 / 4 draft tokens |
| Tool calling | Dynamo `glm47` parser |
| Reasoning | Dynamo `glm45` parser |

The deployment is in
[`sglang/disagg-b200-agentic/deploy.yaml`](sglang/disagg-b200-agentic/deploy.yaml).

## Supported features

- Text generation
- Reasoning content
- Tool calling
- KV-aware routing
- NIXL prefill/decode disaggregation
- Host-memory KV offload

## Prerequisites

1. The [Dynamo Platform](../../docs/kubernetes/README.md) is installed.
2. Two B200 nodes are available: four GPUs for prefill and eight GPUs for
   decode.
3. The RDMA device plugin exposes `rdma/ib`. If your cluster uses a different
   extended resource name, update both worker resource blocks.
4. The prefill node has enough host memory for four 200 GB HiCache pools. The
   manifest sets a 1 TiB worker memory limit because `--hicache-size` is per
   tensor-parallel rank.
5. A Hugging Face token secret exists in the target namespace:

   ```bash
   export NAMESPACE=your-namespace
   kubectl create secret generic hf-token-secret \
     --from-literal=HF_TOKEN="your-token" \
     -n ${NAMESPACE}
   ```

The manifest pins the public preview image
`nvcr.io/nvidia/ai-dynamo/sglang-runtime-nightly:20260701-5245c0f` because
GLM-5.2 requires newer SGLang support than the latest stable Dynamo runtime.

## Quick start

### 1. Create the model cache

Edit `model-cache/model-cache.yaml` and set `storageClassName` to a RWX storage
class, then create and populate the cache:

```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/glm52-model-download \
  -n ${NAMESPACE} --timeout=7200s
```

If the target namespace already provides a shared model-cache PVC, skip PVC
creation and change `claimName: model-cache` in the deployment and benchmark
manifests to that PVC.

### 2. Deploy

```bash
kubectl apply -f sglang/disagg-b200-agentic/deploy.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=glm52-b200-tp8-kv-disagg \
  -n ${NAMESPACE} --timeout=7200s
```

The worker startup script discovers GPU-local InfiniBand devices from
`nvidia-smi topo -m` and exports them through
`SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS`. It fails startup instead of
silently falling back when no suitable device is visible.

### 3. Smoke test

```bash
kubectl port-forward svc/glm52-b200-tp8-kv-disagg-frontend \
  8000:8000 -n ${NAMESPACE}

curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/GLM-5.2-NVFP4",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 32
  }'
```

### 4. Benchmark

See [`perf/README.md`](perf/README.md). The default Job replays the 15% agentic
Mooncake trace at concurrency 64 and writes AIPerf artifacts to the model-cache
PVC.

## Important tuning notes

- Both workers use a 500,000-token context. A 524,288-token context exceeded
  the measured 520,000-token prefill KV capacity for this configuration.
- HiCache size is per prefill rank. `--hicache-size 200` therefore allocates
  roughly 800 GB across TP4 before indexer and draft-pool overhead.
- JIT and compilation caches are intentionally pod-local under
  `/tmp/compilation-cache`; only model files use the shared PVC.
- The decode worker sets `SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600` so slow
  transfers are observable rather than failing at the shorter default timeout.
- The `SGLANG_SIMULATE_ACC_*` variables force an acceptance length of 2.69 for
  performance comparison. Remove them for quality or acceptance-length
  evaluation.
- The recipe does not use fake prefill, dummy weights, or simulated uniform
  experts.

## Known limitations

- A longer disaggregation timeout does not recover an orphaned NIXL transfer;
  monitor `State index length mismatch` and `KVPoll.WaitingForInput` during
  long trace replays.
- The included benchmark configuration is reproducible, but no completed
  performance result is published here yet.
