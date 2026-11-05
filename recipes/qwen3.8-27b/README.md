<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.8-27B Recipes

Serve computer-use agents on Qwen3.8-27B with the experimental Dynamo vLLM sidecar. Most requests
go to the worker that caches their prompt and screenshot, so cache reuse equals per-session
pinning without a client session ID.

## What Runs

- Each worker runs vLLM's Rust server (`vllm-rs`) on one GPU beside the
  [Dynamo vLLM sidecar](../../docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/vllm/sidecar.md).
- The frontend parses tool calls and reasoning. Its multimodal-aware KV router (two-tier
  policy) picks the most-cached worker unless load is very uneven.

## Results

AIPerf replays [`Hcompany/h_cua_perf`](https://huggingface.co/datasets/Hcompany/h_cua_perf),
image `uuid` removed, 128 concurrent agent sessions, 8 GB200 GPUs.

| Serving path | Throughput (req/s) | Prompt tokens from cache |
|--------------|--------------------|--------------------------|
| **This recipe (no session ID)** | 7.29 | 88.7% |
| Dynamo, client session-ID routing | 7.26 | 88.6% |
| `vllm-rs` alone, client-pinned sessions | 7.20 | 88.6% |

The three are equal within run-to-run variation (about 1%). Mean time to first token is 363 ms,
the same as session-ID routing. Measured manually on October 2026 main builds with these settings.

## Deploy

You need 8 GB200 GPUs, Kubernetes 1.29 or later, Dynamo operator 1.6.0 or later, and the
[`hf-token-secret`](../README.md#prerequisites) Secret. Set `storageClassName` in
`model-cache/model-cache.yaml`, then run from this directory:

```bash
export NAMESPACE=<your-namespace>
kubectl apply -n $NAMESPACE -f model-cache/model-cache.yaml
kubectl apply -n $NAMESPACE -f model-cache/model-download.yaml
kubectl wait -n $NAMESPACE --for=condition=Complete job/qwen38-27b-model-download --timeout=3600s
kubectl apply -n $NAMESPACE -f vllm/agg-gb200-sidecar/deploy.yaml
```

## Verify

Sidecars log `Resolved image placeholder token for multimodal KV routing`; the frontend logs
`MM-aware KV routing enabled`.

```bash
kubectl port-forward -n $NAMESPACE svc/qwen38-27b-vllm-sidecar-agg-frontend 8000:8000
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3.8-27B",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}},
    {"type": "text", "text": "Describe the image in one sentence."}]}],
  "max_tokens": 512
}'
```

## Keep Multimodal-Aware Routing On

- Mount the model cache in the sidecar to give it the image token.
- Send image parts without a client `uuid` or `mm_processor_kwargs`.
- Keep the default frontend tokenizer: `--tokenizer fastokens` turns this routing off.
