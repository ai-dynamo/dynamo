# [Experimental] Kimi-K2.5 nvidia/Kimi-K2.5-NVFP4 — TokenSpeed Aggregated Deployment on Kubernetes

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

> **Note: raw Kubernetes primitives, not `DynamoGraphDeployment`.** The Dynamo
> Operator's CRD currently only validates `backendFramework` values `vllm`, `sglang`,
> `trtllm` (see `deploy/operator/api/v1beta1/common.go`). Until `tokenspeed` is added
> to that enum, this recipe wires the four processes the operator would otherwise
> generate (etcd discovery + NATS event-plane + frontend HTTP router + worker engine)
> as plain `Deployment`s and `Service`s. The frontend Service is named
> `kimi-k25-tokenspeed-agg-frontend` to match the operator-generated naming so
> port-forward instructions stay stable across the migration to
> `DynamoGraphDeployment` once supported. The `deploy.yaml` carries an inline TODO
> marking the swap point.
>
> **Why etcd and not `--discovery-backend kubernetes`**: the K8s-native discovery
> client requires operator-stamped scaffolding (downward-API env, the
> `dynamoworkermetadatas.nvidia.com` CRD, RBAC for `DynamoWorkerMetadata` +
> `EndpointSlices`, labeled worker `Service`/`EndpointSlice`). Replicating that by
> hand defeats this recipe's "no operator required" stance, so we keep etcd as a
> self-contained sidecar; the eventual DGD migration delegates all of it to the
> operator.

## Image — local build required

There is **no public `nvcr.io/nvidia/ai-dynamo/tokenspeed-runtime` image** at the time
this recipe was written; TokenSpeed integration in Dynamo is still in flight. This
directory ships a [`Dockerfile`](Dockerfile) that builds a Dynamo+TokenSpeed image
on top of `docker.io/lightseekorg/tokenspeed-runner:cu130-torch-2.11.0` (the upstream
TokenSpeed runtime base, pinned by tag for reproducibility).

The lightseek runner image is the **runtime base only** — it ships CUDA 13, PyTorch
2.11, and mooncake, but not the TokenSpeed engine itself (the engine is not yet
published on PyPI; only a name-reservation placeholder exists). This Dockerfile
installs TokenSpeed from upstream source per the official guide at
[lightseek.org/tokenspeed/guides/getting-started](https://lightseek.org/tokenspeed/guides/getting-started),
then layers Dynamo on top.

Run the build, push the result to a registry your cluster can pull from, then
update the `image:` fields in [`deploy.yaml`](deploy.yaml).

### 1. Build the Dynamo+TokenSpeed image

The build context must be the **Dynamo repo root** (the Dockerfile `COPY`s the source
tree in to build the Dynamo Python wheel via `maturin`).

```bash
# From the repo root.
docker build \
  -f recipes/kimi-k2.5/tokenspeed/agg/nvidia/Dockerfile \
  --target dev \
  -t <your-registry>/dynamo-tokenspeed:dev \
  .
```

The Dockerfile defaults `BASE_IMAGE` to a pinned public TokenSpeed runtime tag:

```
docker.io/lightseekorg/tokenspeed-runner:cu130-torch-2.11.0
```

This tag is CUDA-13-based (matches B200's CUDA 13.x driver) and ships PyTorch 2.11.0.
Sibling tags on the same repo: `cu130`, `cu129-torch-2.11.0`, `cu129`, `latest`. To
target a different runtime, override the build arg:

```bash
--build-arg BASE_IMAGE=docker.io/lightseekorg/tokenspeed-runner:cu129-torch-2.11.0
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

- A Kubernetes cluster (the Dynamo Operator is **not** required for this recipe — see
  the note above)
- 4× B200 GPUs (or 8× B200 if you raise `--tensor-parallel-size` to 8)
- A `hf-token-secret` Secret containing your Hugging Face token
- An `nvcrimagepullsecret` Secret in the namespace, or update `imagePullSecrets` in
  [`deploy.yaml`](deploy.yaml) to match your cluster's pull-secret naming
- A pre-existing `model-cache` PVC with `nvidia/Kimi-K2.5-NVFP4` downloaded (see
  the [`model-cache/`](../../../model-cache/) sibling job)
- A registry your cluster's nodes can pull from with the `dynamo-tokenspeed` image
  pushed (see the build steps above)

## Deploy

```bash
export NAMESPACE=dynamo-demo

# Download model weights into the model-cache PVC
kubectl apply -f ../../../model-cache/model-cache.yaml -n ${NAMESPACE}
kubectl apply -f ../../../model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=6000s

# After updating both image: fields in deploy.yaml, apply.
kubectl apply -f deploy.yaml -n ${NAMESPACE}
```

This creates four Deployments + three Services:

| Resource | Purpose | Image |
|---|---|---|
| `kimi-k25-tokenspeed-etcd` (Deployment + Service) | Discovery backend | `gcr.io/etcd-development/etcd:v3.6.7` |
| `kimi-k25-tokenspeed-nats` (Deployment + Service) | Event plane (JetStream) | `nats:2.12.4` |
| `kimi-k25-tokenspeed-frontend` (Deployment) | `dynamo.frontend` in KV-router mode on port 8000 | your locally-built `dynamo-tokenspeed` |
| `kimi-k25-tokenspeed-agg-frontend` (Service) | Stable name for port-forward; selects the frontend Deployment | — |
| `kimi-k25-tokenspeed-worker` (Deployment) | `dynamo.tokenspeed` against `nvidia/Kimi-K2.5-NVFP4`, TP=4 + EP=4, NVFP4 weights, FP8 KV cache, MLA attention via `trtllm_mla`, MoE via `flashinfer_trtllm`, with `kimi_k25` reasoning + `kimi_k2` tool-call parsers | your locally-built `dynamo-tokenspeed` |

Discovery uses the etcd Service (`--discovery-backend etcd`, `ETCD_ENDPOINTS=http://kimi-k25-tokenspeed-etcd:2379`)
on both the frontend and the worker; the event plane uses the nats Service
(`NATS_SERVER=nats://kimi-k25-tokenspeed-nats:4222`).

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
    "max_tokens": 300,
    "skip_special_tokens": false,
    "chat_template_kwargs": {"thinking": true}
  }'
```

Kimi K2.5 emits a `<think>...</think>` reasoning prefix; the `kimi_k25` reasoning
parser splits it out into the response's `reasoning_content` field, leaving the
final answer in `content`. Send `max_tokens >= 200` to leave room for both phases.

Both extra request fields are required:

- `chat_template_kwargs: {thinking: true}` tells the chat template to emit the
  `<think>` opener that primes the model to produce a reasoning block before the
  answer. Without it the template skips the think prefix and the model's first
  decoded token changes. (`thinking` is not a top-level OpenAI field, so it must
  go inside `chat_template_kwargs` — Dynamo's frontend rejects unknown top-level
  fields with `400 Unsupported parameter(s): thinking`.)
- `skip_special_tokens: false` keeps structural tokens (`<|im_end|>`, `</think>`)
  in the streamed output so the `kimi_k25` reasoning parser can find the
  reasoning/content boundary. With the default `true`, the parser sees stripped
  output, can't split the segments, and returns `content: null` with a small
  number of trailing-stop tokens.

The two knobs together restore the engine-side contract that the upstream
`tokenspeed serve` monolith gets implicitly because it owns both the chat
template and the parser in the same process; Dynamo's split (frontend renders
template, worker streams output, parser runs after) creates a seam that needs
explicit user-side wiring.

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
  worker layer (etcd watcher, NATS event-plane subscriber, and TCP request-plane
  listener) holds additional memory beyond the engine, and the default headroom is
  too tight for K2.5's MoE weight init. 0.75 is too low — the engine sizes the KV
  cache pool negative.
- `--dyn-reasoning-parser kimi_k25 --dyn-tool-call-parser kimi_k2`: the upstream
  TokenSpeed recipe uses `--reasoning-parser kimi_k2 --tool-call-parser kimi_k2`,
  which apply at the engine level. Dynamo intercepts these at the worker layer
  with the `--dyn-*` variants; `kimi_k25` is the K2.5/K2.6-specific parser.
- `--max-model-len 262144`: K2.5 supports 256K context. The KV cache budget at this
  length is large; reduce if your prompts are shorter to free GPU memory for batches.

## What's different from the TRT-LLM sibling

The [`../../trtllm/agg/nvidia/`](../../trtllm/agg/nvidia/) deployment uses TP=8 and a
ConfigMap-based engine YAML, and ships as a single `DynamoGraphDeployment` letting
the Dynamo Operator generate the underlying `Deployment`s/`Service`s. This recipe
differs on three axes:

1. **Engine knobs are inline CLI flags, not a ConfigMap.** TokenSpeed accepts engine
   knobs directly on the worker command line, similar to vLLM. Layout therefore
   mirrors [`recipes/llama-3-70b/vllm/agg/`](../../../../llama-3-70b/vllm/agg/) rather
   than the TRT-LLM ConfigMap pattern.
2. **Raw `Deployment` + `Service` resources, not a `DynamoGraphDeployment`.** The
   operator's `backendFramework` enum currently only validates `vllm`, `sglang`,
   `trtllm` — `tokenspeed` is rejected by admission validation. Until the operator
   adds `tokenspeed` support, this recipe lays out the four processes the operator
   would otherwise generate as plain `Deployment`s. See the inline TODO in
   `deploy.yaml` — once the operator backend lands, the deploy.yaml collapses to a
   single `DynamoGraphDeployment` matching the TRT-LLM sibling's shape.
3. **Local image build.** The TRT-LLM recipe ships against the public
   `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime` image; this recipe requires you
   to build and push your own Dynamo+TokenSpeed image (see the build steps above)
   until NVIDIA publishes one.
