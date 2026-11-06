<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.8-27B Recipes

Serve computer-use agents on Qwen3.8-27B with the experimental
[Dynamo vLLM sidecar](../../docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/vllm/sidecar.md)
beside each GPU's `vllm-rs` (vLLM's Rust server). The multimodal-aware KV router (two-tier policy)
sends most requests to the worker that caches their prompt and screenshot, so cache reuse
matches per-session pinning without a client session ID.

## Results

AIPerf replays [`Hcompany/h_cua_perf`](https://huggingface.co/datasets/Hcompany/h_cua_perf),
image `uuid` removed, 128 concurrent agent sessions, 8 GB200 GPUs, one run block.

| Serving path | Throughput (req/s) | Prompt tokens from cache |
|--------------|--------------------|--------------------------|
| **This recipe (no session ID)** | 7.17 | 88.6% |
| Python `vllm serve`, client-pinned | 7.09 | 88.7% |
| `vllm-rs` alone, client-pinned | 7.10 | 88.7% |

Throughput is within about 1%. Paired per session turn, mean end-to-end latency is 0.23 s and
0.24 s lower than `vllm serve` and `vllm-rs` alone, and mean time to first token (373 ms) is
41 ms lower than `vllm serve` (95% intervals exclude zero). October 2026 main
builds, launched manually, 300 s warm-up dropped.

## Deploy

You need 8 GB200 GPUs, Kubernetes 1.29+, Dynamo operator 1.6.0+, and the
[`hf-token-secret`](../README.md#prerequisites) Secret. Set `storageClassName` in
`model-cache/model-cache.yaml`, then run from here:

```bash
export NAMESPACE=<your-namespace>
kubectl apply -n $NAMESPACE -f model-cache/model-cache.yaml
kubectl apply -n $NAMESPACE -f model-cache/model-download.yaml
kubectl wait -n $NAMESPACE --for=condition=Complete job/qwen38-27b-model-download --timeout=3600s
kubectl apply -n $NAMESPACE -f vllm/agg-gb200-sidecar/deploy.yaml
```

## Verify

Check for `Resolved image placeholder token for multimodal KV routing` (sidecars) and
`MM-aware KV routing enabled` (frontend).

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

## Benchmark

Pin the unreleased AIPerf commit behind these numbers and strip screenshot `uuid`s:

```bash
pip install "aiperf @ git+https://github.com/ai-dynamo/aiperf@59087bab509fc88336a932851b3ed33bf13405c2"
hf download Hcompany/h_cua_perf --repo-type dataset \
  --revision c13201944c7b9769e86357adcfd9bc8d991d46de --local-dir h_cua_perf
zstd -dc h_cua_perf/h_cua.jsonl.zst | python3 -c '
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    for m in r["messages"]:
        for part in m["content"] if isinstance(m.get("content"), list) else []:
            part.pop("uuid", None)
    print(json.dumps(r))
' > h_cua.nouuid.jsonl
AIPERF_DATASET_CONFIGURATION_TIMEOUT=3600 AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=3600 \
aiperf profile -m Qwen/Qwen3.8-27B --url localhost:8000 --endpoint-type chat --streaming \
  --use-server-token-count --input-file h_cua.nouuid.jsonl --custom-dataset-type mooncake_trace \
  --dataset-sampling-strategy sequential --synthesis-max-osl 16384 --concurrency 128 \
  --benchmark-duration 900 --benchmark-grace-period 120 \
  --extra-inputs '{"ignore_eos":true,"temperature":0}' --random-seed 0
```

## Keep Multimodal-Aware Routing On

- Mount the model cache in the sidecar (it reads the image token).
- Send images without a client `uuid` or `mm_processor_kwargs`.
- Keep the default tokenizer; `--tokenizer fastokens` turns it off.
