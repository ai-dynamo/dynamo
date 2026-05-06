# Kimi-K2.5 nvidia/Kimi-K2.5-NVFP4 — TokenSpeed Aggregated Deployment on Kubernetes

> **Text only:** the NVFP4 export of Kimi-K2.5 ships without vision-tower weights. The
> TokenSpeed loader logs warnings for the missing `vision_tower.*` parameters and
> continues with text-only forward paths. Image inputs will fail.

This recipe runs `nvidia/Kimi-K2.5-NVFP4` on the **TokenSpeed** engine under Dynamo's
KV-aware aggregated frontend. It mirrors the parameter set documented in the upstream
[TokenSpeed model recipes](https://lightseek.org/tokenspeed/recipes/models) for
Kimi K2.5 / K2.6.

| Deployment | Manifest | Description | Hardware |
|-----------|----------|-------------|----------|
| **Aggregated (TokenSpeed)** | [`deploy.yaml`](deploy.yaml) | Aggregated serving with KV-aware routing using the TokenSpeed engine | 1× 4×B200 (TP=4, EP=4) |

## Image — local build required

There is **no public `nvcr.io/nvidia/ai-dynamo/tokenspeed-runtime` image** at the time
this recipe was written; TokenSpeed integration in Dynamo is still in flight. This
directory ships a [`Dockerfile`](Dockerfile) that builds a Dynamo+TokenSpeed image
on top of a public TokenSpeed runtime image. Run the build, push the result to a
registry your cluster can pull from, then update the `image:` fields in
[`deploy.yaml`](deploy.yaml).

### 1. Build the Dynamo+TokenSpeed image

The build context must be the **Dynamo repo root** (the Dockerfile `COPY`s the source
tree in to build the Dynamo Python wheel via `maturin`).

```bash
# From the repo root.
docker build \
  -f recipes/kimi-k2.5/tokenspeed/agg/nvidia/Dockerfile \
  --build-arg BASE_IMAGE=<your-tokenspeed-image>:<tag> \
  --target dev \
  -t <your-registry>/dynamo-tokenspeed:dev \
  .
```

The Dockerfile has two reachable targets:

- `--target runtime`: slim image, no compilers. Use for production-leaning deploys.
- `--target dev`: keeps `cargo`, `rustc`, `maturin`, build-essential, git, etc. Useful
  when iterating on the Rust bindings inside the pod (`maturin develop` inline).

For arm64 base images, add `--build-arg ARCH=arm64`.

### 2. Push to a cluster-reachable registry

```bash
docker push <your-registry>/dynamo-tokenspeed:dev
```

### 3. Update both `image:` fields in `deploy.yaml`

The `Frontend` and `TokenSpeedWorker` services in [`deploy.yaml`](deploy.yaml) both
reference the placeholder `<your-registry>/dynamo-tokenspeed:dev`. Replace both with
the tag you just pushed.

## Prerequisites

- A Kubernetes cluster with the [Dynamo Operator](https://docs.nvidia.com/dynamo/) installed
- 4× B200 GPUs (or 8× B200 if you raise `--tensor-parallel-size` to 8)
- A `hf-token-secret` Secret containing your Hugging Face token
- A pre-existing `model-cache` PVC with `nvidia/Kimi-K2.5-NVFP4` downloaded (see
  the [`model-cache/nvidia/`](../../../model-cache/nvidia/) sibling job)
- A registry your cluster's nodes can pull from with the `dynamo-tokenspeed` image
  pushed (see the build steps above)

## Deploy

```bash
export NAMESPACE=dynamo-demo

# Download model weights into the model-cache PVC
kubectl apply -f ../../../model-cache/model-cache.yaml -n ${NAMESPACE}
kubectl apply -f ../../../model-cache/nvidia/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=6000s

# After updating both image: fields in deploy.yaml, apply.
kubectl apply -f deploy.yaml -n ${NAMESPACE}
```

This creates a **DynamoGraphDeployment** (`kimi-k25-tokenspeed-agg`) with:

- A **Frontend** running `dynamo.frontend` in KV-router mode on port 8000.
- A **TokenSpeedWorker** running `dynamo.tokenspeed` against `nvidia/Kimi-K2.5-NVFP4`,
  TP=4 + expert parallel, NVFP4 weights, FP8 KV cache, MLA attention via
  `trtllm_mla`, MoE via `flashinfer_trtllm`, and the `kimi_k25` / `kimi_k2`
  Dynamo reasoning + tool-call parsers.

## Test the deployment

```bash
kubectl port-forward svc/kimi-k25-tokenspeed-agg-frontend 8000:8000 -n ${NAMESPACE}
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2.5",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 300
  }'
```

Kimi K2.5 emits a `<think>...</think>` reasoning prefix; the `kimi_k25` reasoning
parser splits it out into the response's `reasoning_content` field, leaving the
final answer in `content`. Send `max_tokens >= 200` to leave room for both phases.

## Notes on parameter choice

- `--tensor-parallel-size 4 --enable-expert-parallel`: matches the upstream TokenSpeed
  recipe; attention is sharded TP=4, MoE experts are sharded EP=4 across the same 4
  ranks. Bump to 8 if you have an 8×B200 node and want lower latency at the cost of
  higher per-GPU memory pressure.
- `--quantization nvfp4 --attention-backend trtllm_mla --moe-backend flashinfer_trtllm`:
  Blackwell-native fast paths. These three together are why the recipe targets B200.
- `--kv-cache-dtype fp8`: paired with `--quantization nvfp4`, halves the KV cache
  footprint vs BF16 KV at no measurable accuracy loss for chat workloads.
- `--gpu-memory-utilization 0.80`: lower than TokenSpeed's 0.85 default. The Dynamo
  worker layer (etcd discovery, NATS event plane, request-plane TCP) holds additional
  memory beyond the engine, and the default headroom is too tight for K2.5's MoE
  weight init. 0.75 is too low — the engine sizes the KV cache pool negative.
- `--dyn-reasoning-parser kimi_k25 --dyn-tool-call-parser kimi_k2`: the upstream
  TokenSpeed recipe uses `--reasoning-parser kimi_k2 --tool-call-parser kimi_k2`,
  which apply at the engine level. Dynamo intercepts these at the worker layer
  with the `--dyn-*` variants; `kimi_k25` is the K2.5/K2.6-specific parser.
- `--max-model-len 262144`: K2.5 supports 256K context. The KV cache budget at this
  length is large; reduce if your prompts are shorter to free GPU memory for batches.

## What's different from the TRT-LLM sibling

The [`../../trtllm/agg/nvidia/`](../../trtllm/agg/nvidia/) deployment uses TP=8 and
a ConfigMap-based engine YAML. TokenSpeed accepts engine knobs directly as worker
CLI flags, so this recipe inlines them in `deploy.yaml` (no ConfigMap), matching the
[`recipes/llama-3-70b/vllm/agg/`](../../../../llama-3-70b/vllm/agg/) layout. The
TRT-LLM recipe ships against the public
`nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime` image; this recipe requires you to
build and push your own Dynamo+TokenSpeed image (see the build steps above) until
NVIDIA publishes one.
