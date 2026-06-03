<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Gemma 4 12B (text + image + audio)

Serves [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) using
vLLM with an aggregated Dynamo deployment.

`google/gemma-4-12B-it` is a unified, **encoder-free** decoder
(`architectures: ["Gemma4UnifiedForConditionalGeneration"]`, `model_type:
gemma4_unified`): text, image, and audio inputs project directly into a single
decoder-only transformer with no separate vision/audio encoder.

Released vLLM does not yet ship this architecture — it is added by vLLM
**PR [#44429](https://github.com/vllm-project/vllm/pull/44429)** (pure-Python: one
model file + a registry line, no CUDA kernels). This recipe builds a custom
container that layers that PR's vLLM, a `transformers` build carrying
`models/gemma4_unified`, and the `ai-dynamo` wheel
(from <https://pypi.nvidia.com/ai-dynamo/>) onto an upstream vLLM image — no Rust
toolchain.

> **Scope:** text, image, and audio chat inputs are verified locally. The
> benchmark below is image-only.

## Results

Measured on a single **GH200 (96 GB, aarch64)**, aggregated, TP=1,
`--max-model-len 32768`, `--gpu-memory-utilization 0.85`. Benchmark: `aiperf`
0.6.0, **256 requests, concurrency 32**, 1 image/request (512×512, 8-image pool),
~11 input text tokens, 150 output tokens (`min=max=150, ignore_eos`). 256/256
requests succeeded.

| Metric | avg | p50 | p99 |
|--------|----:|----:|----:|
| Output token throughput (tok/s) | **1,616** | — | — |
| Request throughput (req/s) | **11.06** | — | — |
| Time to First Token (ms) | 385 | 379 | 573 |
| Inter-Token Latency (ms) | 17.2 | 16.8 | 21.9 |
| Request Latency (ms) | 2,874 | 2,866 | 3,102 |
| Per-user output (tok/s/user) | 58.4 | 59.6 | 62.5 |

> Local single-container verification run (docker, file-KV discovery), using
> inlined base64 images to avoid network-fetch flakiness. The canonical k8s
> benchmark (`vllm/agg/perf.yaml` + `data-gen/`) uses 1000 requests, concurrency
> 64, and the COCO http image pool — expect higher absolute throughput at that
> concurrency. Numbers here establish a correctness-backed baseline, not a tuned
> peak.

## Topology

| Role | Replicas | GPUs/replica | Notes |
|------|----------|--------------|-------|
| Frontend | 1 | 0 | Dynamo frontend with prefix-hash KV routing |
| vLLM worker | 1 | 1 | Text, image, and audio inputs (encoder-free) |

## Prerequisites

- A Kubernetes cluster with the [Dynamo Operator](../../docs/kubernetes/README.md) installed
- One NVIDIA GPU per worker replica
- Shared PVC storage for the Hugging Face model cache
- Hugging Face access to the gated `google/gemma-4-12B-it`

## Step 1: Build the Container

```bash
docker build \
  --platform=linux/arm64 \
  -t <your-registry>/gemma4-12b-vllm:latest \
  -f recipes/gemma-4-12b/Dockerfile \
  .
docker push <your-registry>/gemma4-12b-vllm:latest
```

> **Pin `--platform` to your target arch.** On an aarch64 host (e.g. GH200) where
> QEMU binfmt is registered, Docker may otherwise resolve the base image to
> `linux/amd64` and build the whole image under emulation (including pulling the
> *amd64* precompiled vLLM wheel), producing an image whose binaries can't run
> natively. `--platform=linux/arm64` forces the arm64 base + the arm64 wheel.
> The PR commit is pinned in the Dockerfile (`VLLM_PR_SHA`); override with
> `--build-arg VLLM_PR_SHA=<sha>` if needed.

Useful build args:

- `VLLM_PR_REF=<ref>` — PR #44429 ref to fetch (default: `refs/pull/44429/head`).
- `VLLM_PR_SHA=<sha>` — exact PR commit to check out after fetching.
- `TRANSFORMERS_REF=<sha>` — pin the `transformers` `main` commit that ships `models/gemma4_unified`.
- `BASE_IMAGE=<image>` — upstream vLLM base (default `vllm/vllm-openai:v0.22.0`); align its vLLM minor with PR #44429's base.
- `DYNAMO_VERSION=<version>` — pin an `ai-dynamo` release/nightly from <https://pypi.nvidia.com/ai-dynamo/>.

If no aarch64 precompiled vLLM wheel exists for the pinned commit, edit the
Dockerfile to drop `VLLM_USE_PRECOMPILED=1` (full source build — slow but
correct on a GH200), or build a wheel in `~/vllm` and `COPY` it in (see the
Dockerfile header).

## Step 2: Download the Model

```bash
export NAMESPACE=<your-namespace>
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# First edit storageClassName in model-cache.yaml for your cluster.
kubectl apply -f recipes/gemma-4-12b/model-cache/model-cache.yaml -n ${NAMESPACE}

kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=<your-hf-token> -n ${NAMESPACE}

kubectl apply -f recipes/gemma-4-12b/model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=complete job/model-download -n ${NAMESPACE} --timeout=3600s
```

## Step 3: Deploy

Edit `vllm/agg/deploy.yaml` and replace `<your-registry>/gemma4-12b-vllm:latest`
with your built image, then:

```bash
kubectl apply -f recipes/gemma-4-12b/vllm/agg/deploy.yaml -n ${NAMESPACE}
kubectl get pods -n ${NAMESPACE} -l nvidia.com/dynamo-graph-deployment-name=gemma4-12b-agg -w
```

## Step 4: Test

```bash
kubectl port-forward svc/gemma4-12b-agg-frontend 8000:8000 -n ${NAMESPACE}
```

Text:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-12B-it",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128
  }'
```

Image:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-12B-it",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"}},
        {"type": "text", "text": "Describe what is in this image."}
      ]
    }],
    "max_tokens": 256
  }'
```

Audio:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-12B-it",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this audio?"},
        {"type": "audio_url", "audio_url": {"url": "<HTTPS-or-data-URI WAV/MP3>"}}
      ]
    }],
    "max_tokens": 128
  }'
```

## Local (no Kubernetes) bring-up

The same image runs directly under Docker — useful on a single GPU box:

```bash
IMG=<your-registry>/gemma4-12b-vllm:latest
docker run --gpus all --rm -it -p 8000:8000 \
  --entrypoint /bin/bash \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e DYN_DISCOVERY_BACKEND=file \
  -e DYN_FILE_KV=/tmp/dynamo-gemma4-kv \
  -e DYN_REQUEST_PLANE=tcp \
  -e DYN_EVENT_PLANE=zmq \
  "$IMG" -lc '
    set -euo pipefail
    rm -rf "${DYN_FILE_KV}"
    python3 -m dynamo.frontend --router-mode kv --no-kv-events --http-port 8000 &
    frontend_pid=$!
    trap "kill ${frontend_pid} 2>/dev/null || true" EXIT
    python3 -m dynamo.vllm \
      --model google/gemma-4-12B-it \
      --served-model-name google/gemma-4-12B-it \
      --enable-multimodal \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.85 \
      --max-model-len 32768 \
      --enable-prefix-caching \
      --no-enable-log-requests \
      --dyn-tool-call-parser gemma4 \
      --dyn-reasoning-parser gemma4'
```

Then use the same `curl` calls against `http://localhost:8000`.

## Benchmark

```bash
# 1. Generate the text+image dataset (writes to the perf-cache PVC).
kubectl apply -f recipes/gemma-4-12b/data-gen/generate-datasets-job.yaml -n ${NAMESPACE}
kubectl wait --for=condition=complete job/gemma4-12b-generate-datasets -n ${NAMESPACE} --timeout=3600s
kubectl logs job/gemma4-12b-generate-datasets -n ${NAMESPACE}

# 2. Deploy + run the benchmark.
NAMESPACE=${NAMESPACE} recipes/gemma-4-12b/vllm/agg/run-benchmark.sh
kubectl logs -f gemma4-12b-agg-benchmark -n ${NAMESPACE}
```

To benchmark locally instead, generate the dataset with
`benchmarks/multimodal/jsonl/main.py` and point `aiperf profile --url
http://localhost:8000` at the Docker deployment above (see `perf.yaml` for the
exact flags).

## Key Configuration Notes

- `--enable-multimodal` enables multimodal input.
- `--dyn-tool-call-parser gemma4` / `--dyn-reasoning-parser gemma4` enable Gemma 4
  tool-call and reasoning parsing (already shipped in Dynamo, architecture-agnostic).
  For tool-calling prompts, also pass `--custom-jinja-template
  examples/chat_templates/gemma4_tool.jinja` (mount it into the image, since this
  recipe image does not vendor the Dynamo `examples/` tree).
- No embedding cache: Gemma 4 is encoder-free, so there is no separate encoder
  output to cache or transfer (unlike `qwen3-vl-30b`).
- The frontend uses `--router-mode kv --no-kv-events` (prefix-hash KV routing
  without backend KV events).

## Notes

- **Aggregated only — disaggregated E/PD is NOT supported for this model.**
  Gemma 4 is encoder-free: image patches project directly inside the decoder, so
  there is no separable vision-encoder stage to run on a standalone encode worker
  and NIXL-transfer to prefill. Do **not** set `--route-to-encoder` /
  `--disaggregation-mode=encode` for this recipe — Dynamo's encode worker calls
  `load_vision_model().visual`, which an encoder-free model does not have. In
  aggregated mode Dynamo passes the image straight to vLLM, which generates the
  embeddings internally (`handlers.py`: *"Without encode worker, the embedding
  will be generated internally by vLLM."*). Prefill/decode (KV-transfer) disagg
  would still be valid, but is out of scope for this recipe.
- **MM-aware KV routing:** Gemma 4 Unified generation works without a separate
  encode worker. Dynamo needs Gemma 4-specific routing-side image-token counting
  to make the Rust/lightseek MM-aware KV router distinguish image content; with
  older images it falls back to normal text-prefix KV routing.
- **Upstream tracking.** vLLM PR #44429 is unmerged. Once it merges and a
  `transformers` release ships `models/gemma4_unified`, repin the Dockerfile to a
  released vLLM/transformers and drop the from-source install.

## File Layout

```text
recipes/gemma-4-12b/
  README.md
  Dockerfile
  model-cache/
    model-cache.yaml
    model-download.yaml
  data-gen/
    generate-datasets-job.yaml
  vllm/
    agg/
      deploy.yaml
      perf.yaml
      run-benchmark.sh
```
