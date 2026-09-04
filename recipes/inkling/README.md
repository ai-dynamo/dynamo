<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TML Inkling Recipes

Recipes for **thinkingmachines/Inkling-NVFP4**.

## Configurations

Dynamo + SGLang deployment profile:

|                          | B200                                  |
|--------------------------|---------------------------------------|
| **GPU** (per worker)     | 8x B200                               |
| **Mode**                 | aggregated                            |
| **Framework**            | SGLang (sglang-inkling custom build)  |
| **Precision**            | NVFP4 (`modelopt_fp4`)                |
| **Parallelism**          | TP8                                   |
| **Attention backend**    | FA4                                   |
| **FP4 GEMM backend**     | FLASHINFER_TRTLLM                     |
| **MoE runner backend**   | FLASHINFER_TRTLLM_ROUTED              |
| **AllReduce backend**    | Torch symmetric memory                |
| **Mamba radix cache**    | extra_buffer strategy                 |
| **Speculative decoding** | EAGLE multi-layer (8 steps, topk 1, 9 draft tokens, rejection sampling) |

Dynamo + vLLM deployment profiles, tuned for an agentic workload (64K median
ISL, 400 median OSL, 90% KV cache hit):

|                          | GB300 aggregated                | GB300 disaggregated             |
|--------------------------|---------------------------------|---------------------------------|
| **GPU**                  | 8x GB300 (2 replicas x TP4)     | 8x GB300 (1 prefill + 1 decode, TP4 each) |
| **Mode**                 | aggregated                      | disaggregated prefill/decode    |
| **Framework**            | vLLM                            | vLLM                            |
| **Precision**            | NVFP4 (ModelOpt), BF16 KV       | NVFP4 (ModelOpt), BF16 KV       |
| **Parallelism**          | TP4                             | TP4 per worker                  |
| **MoE runner backend**   | auto: FLASHINFER_TRTLLM (routed) + FlashInfer CUTLASS (shared) | auto: FLASHINFER_TRTLLM (routed) + FlashInfer CUTLASS (shared) |
| **AllReduce backend**    | vLLM default                    | FlashInfer (MNNVL)              |
| **Context length**       | 1,048,576                       | 1,048,576                       |
| **Speculative decoding** | MTP, 8 draft tokens             | MTP, 8 draft tokens             |
| **Routing**              | KV-aware                        | KV-aware                        |
| **KV transfer**          | n/a                             | NIXL over MNNVL, or RDMA where available |

Inkling excludes the shared experts from NVFP4 quantization, so the two expert
groups resolve their backend on their own. `--moe-backend` is global. Pinning it
would also override the choice for the shared experts, which no published number
covers.

## Supported features

- Reasoning (`inkling` reasoning parser) — all profiles
- Tool calling (`inkling` tool-call parser) — all profiles
- Modalities: text, image, and audio input — **SGLang B200 only**. The Dynamo
  vLLM runtime does not support multimodal input yet, so the vLLM GB300
  profiles are text-only.

## Prerequisites

1. **Dynamo Platform installed** — see [Kubernetes Deployment Guide](../../docs/fern/pages/kubernetes/getting-started/quickstart.mdx).
2. **Image pull secret** — SGLang B200 profile only, for access to
   `nvcr.io/nvstaging/nim` (staging registry):
   ```bash
   export NAMESPACE=your-namespace
   kubectl create secret docker-registry nvcr-imagepullsecret \
     --docker-server=nvcr.io \
     --docker-username='$oauthtoken' \
     --docker-password="your-ngc-api-key" \
     -n ${NAMESPACE}
   ```
3. **HuggingFace token** (the model is public, but a token avoids rate limits):
   ```bash
   kubectl create secret generic hf-token-secret \
     --from-literal=HF_TOKEN="your-token" \
     -n ${NAMESPACE}
   ```

## Quick Start

### 1. Create Namespace

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
```

### 2. Create Storage

> [!NOTE]
> Edit `model-cache/model-cache.yaml` first and update `storageClassName` to match your cluster (`kubectl get storageclass`). On clusters that already provide a shared RWX PVC (e.g. `shared-model-cache` on the Dynamo dev clusters), skip this step and replace the `model-cache` claim name in `model-cache/model-download.yaml` (`.spec.template.spec.volumes[0].persistentVolumeClaim.claimName`) and in every `persistentVolumeClaim.claimName` of the profile you deploy. For the Kustomize-based disaggregated profile, change it in `vllm/disagg-gb300-agentic/kustomize/base/deploy.yaml` and re-render, rather than editing `deploy-*.yaml`.

```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
```

### 3. Download the Model

The checkpoint is ~592 GB.

```bash
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/inkling-model-download -n ${NAMESPACE} --timeout=7200s
```

### 4. Deploy the DynamoGraphDeployment

Pick one profile, then apply it:

| Profile | `DEPLOY` |
| --- | --- |
| SGLang, aggregated, B200 | `sglang/agg-b200/deploy.yaml` |
| vLLM, aggregated, GB300 | `vllm/agg-gb300-agentic/deploy.yaml` |
| vLLM, disaggregated, GB300 | `vllm/disagg-gb300-agentic/deploy-generic.yaml` |
| vLLM, disaggregated, GB300, RoCE via DRA | `vllm/disagg-gb300-agentic/deploy-aws-roce.yaml` |

```bash
export DEPLOY=vllm/agg-gb300-agentic/deploy.yaml
kubectl apply -f ${DEPLOY} -n ${NAMESPACE}
```

Both disaggregated profiles use Dynamic Resource Allocation, so they need the
NVIDIA DRA driver (for `ComputeDomain`) installed cluster-wide and a cluster that
serves the DRA APIs. `deploy-aws-roce.yaml` additionally declares a
`resource.k8s.io/v1` `ResourceClaimTemplate`, which requires **Kubernetes 1.34 or
later**, and a cluster exposing the `roce.networking.k8s.aws` device class;
`deploy-generic.yaml` moves KV over MNNVL `cuda_ipc` instead. To compose a
different fabric, apply the overlay instead:
`kubectl apply -k vllm/disagg-gb300-agentic/kustomize/overlays/generic`.

To benchmark a deployment, see [`perf/README.md`](perf/README.md).

### 5. Smoke Test

Forward the frontend of the profile you deployed — the Service is the DGD name
with a `-frontend` suffix:

```bash
export FRONTEND=inkling-vllm-gb300-agg-agentic-frontend
kubectl port-forward svc/${FRONTEND} 8000:8000 -n ${NAMESPACE} &
```

| Profile | Service |
| --- | --- |
| SGLang, aggregated, B200 | `tml-inkling-sglang-agg-frontend` |
| vLLM, aggregated, GB300 | `inkling-vllm-gb300-agg-agentic-frontend` |
| vLLM, disaggregated, GB300 | `inkling-vllm-gb300-disagg-agentic-frontend` |

Every profile serves the model as `thinkingmachines/Inkling-NVFP4`, so the
requests below are identical across them. Image and audio need the SGLang B200
profile.

#### Text

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "thinkingmachines/Inkling-NVFP4",
    "messages": [{"role": "user", "content": "Hello, who are you?"}],
    "max_tokens": 1500
  }'
```

Inkling reasons before answering, so a small `max_tokens` is spent entirely on reasoning and
returns empty `content` with `finish_reason: "length"`. The answer may also land in
`reasoning_content` rather than `content`.

> [!NOTE]
> The image and audio tests below need the **SGLang B200** profile. On vLLM GB300 they are rejected
> with `Received multimodal data but multimodal processing is not enabled` — that is expected; see
> [Known Issues](#known-issues).

#### Image

Images are sent as `image_url` content parts. Use a public HTTP(S) URL that the
worker pod can fetch, or use a base64 `data:` URI for local files.

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "thinkingmachines/Inkling-NVFP4",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"}},
        {"type": "text", "text": "Describe what is in this image."}
      ]
    }],
    "max_tokens": 512
  }'
```

#### Multiple Images

Send multiple `image_url` content parts in the same user message to test
multi-image input:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "thinkingmachines/Inkling-NVFP4",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg"}},
        {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/corgi.jpg"}},
        {"type": "text", "text": "Describe each image separately. Label them Image 1 and Image 2."}
      ]
    }],
    "max_tokens": 512
  }'
```

#### Audio

Audio is sent as an `audio_url` content part. Use a public HTTP(S) URL that the
worker pod can fetch, or use a base64 `data:` URI for local files. Inkling expects
16 kHz audio. Use WAV when possible; MP3, FLAC, and OGG also decode in this
container image. This sample is 16 kHz mono WAV, matching the model card spec.

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "thinkingmachines/Inkling-NVFP4",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "audio_url", "audio_url": {"url": "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/mlk.wav"}},
        {"type": "text", "text": "Transcribe the following speech to text."}
      ]
    }],
    "max_tokens": 256
  }'
```

#### Multiple Audio Clips

Send multiple `audio_url` content parts in the same user message to test
multi-audio input:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "thinkingmachines/Inkling-NVFP4",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "audio_url", "audio_url": {"url": "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/mlk.wav"}},
        {"type": "audio_url", "audio_url": {"url": "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/librispeech_asr_demo_validation_0.wav"}},
        {"type": "text", "text": "Transcribe each audio clip separately. Label them Audio 1 and Audio 2."}
      ]
    }],
    "max_tokens": 512
  }'
```

For air-gapped clusters, send local media files as `data:` URIs instead:

```json
{"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,<BASE64_WAV>"}}
```

Mix multiple media parts in one message as the context budget allows.

#### Reasoning Effort

Inkling's controllable thinking is exposed per request: pass `reasoning_effort` as a
named level (`none` / `minimal` / `low` / `medium` / `high` / `max`) or a float in
`[0.0, 0.99]`; omitted requests default to `0.9` (high). These are the values in this
checkpoint's chat template — `xhigh` from the launch blog is not in its map and is
rejected:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "thinkingmachines/Inkling-NVFP4",
    "messages": [{"role": "user", "content": "What is 17 times 24?"}],
    "chat_template_kwargs": {"reasoning_effort": "low"},
    "max_tokens": 128
  }'
```

## Cleanup

To tear down the deployment and free cluster resources:

```bash
# Stop the port-forward if it is still running
pkill -f "kubectl port-forward svc/tml-inkling-sglang-agg-frontend" 2>/dev/null || true
pkill -f "kubectl port-forward svc/inkling-vllm-gb300-agg-agentic-frontend" 2>/dev/null || true
pkill -f "kubectl port-forward svc/inkling-vllm-gb300-disagg-agentic-frontend" 2>/dev/null || true

# Delete the deployment (stops all pods) -- whichever profile you deployed
kubectl delete dynamographdeployment tml-inkling-sglang-agg -n ${NAMESPACE} 2>/dev/null || true
kubectl delete dynamographdeployment inkling-vllm-gb300-agg-agentic -n ${NAMESPACE} 2>/dev/null || true
kubectl delete dynamographdeployment inkling-vllm-gb300-disagg-agentic -n ${NAMESPACE} 2>/dev/null || true

# Disaggregated only: the ComputeDomain and the RoCE claim template outlive the DGD
kubectl delete computedomain inkling-vllm-gb300-disagg-agentic-compute-domain -n ${NAMESPACE} 2>/dev/null || true
kubectl delete resourceclaimtemplate inkling-vllm-gb300-disagg-agentic-roce -n ${NAMESPACE} 2>/dev/null || true

# Delete the model-download job (idempotent — already finished or not yet run)
kubectl delete job inkling-model-download -n ${NAMESPACE} 2>/dev/null || true

# Delete secrets (optional — omit if you reuse them across deployments)
kubectl delete secret nvcr-imagepullsecret hf-token-secret -n ${NAMESPACE} 2>/dev/null || true

# Delete the PVC — WARNING: this destroys the downloaded 592 GB model.
# Skip this step if you plan to redeploy or share the PVC with other workloads.
# kubectl delete pvc model-cache -n ${NAMESPACE}

# Delete the namespace — WARNING: this destroys everything in it.
# kubectl delete namespace ${NAMESPACE}
```

For a one-shot cleanup run see [`cleanup.sh`](cleanup.sh).

## Known Issues

### vLLM GB300 profiles

**Disable reasoning with `reasoning_effort: "none"`, not with a zero thinking budget.**
`nvext.max_thinking_tokens: 0` returns HTTP 200 but no usable answer: the model emits empty
thinking blocks until it hits `max_tokens` (`finish_reason: "length"`). Inkling's end-of-thinking
token also ends the message, so a zero budget closes the message on the first token and the cycle
repeats. `chat_template_kwargs: {"thinking": false}` is not a substitute either — the model still
reasons, and the reasoning text lands in `content`.

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "thinkingmachines/Inkling-NVFP4",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "reasoning_effort": "none",
    "max_tokens": 128
  }'
```

**Keep both structured-output flags on the worker.** `tool_choice: "required"` and named
`tool_choice` are enforced through a structural-tag grammar that needs both flags the manifests
already set:

```
--dyn-enable-structural-tag
--structured-outputs-config '{"enable_in_reasoning": true}'
```

Drop the second and the grammar is silently inert on a reasoning model — HTTP 200, no error, and
the requested tool is not enforced. Both belong on the worker; the frontend does not accept
`--dyn-enable-structural-tag`.

**Image and audio input are not supported.** The Dynamo vLLM runtime has no multimodal support for
this model yet, so the vLLM GB300 profiles are text-only; use the SGLang B200 profile for image and
audio. Do not set `--enable-multimodal` to work around it — the flag only lifts the request
rejection, so media is still never encoded and requests return HTTP 200 with an answer invented from
the text prompt alone.

**Some OpenAI API fields are rejected rather than silently ignored**, each with a message naming
the reason:

| Request field | Status | Reason |
| --- | --- | --- |
| `min_p`, `logit_bias` | 400 | Unsupported with speculative decoding, and these profiles run MTP 8. Remove `--speculative-config` to use them, at a throughput cost. |
| `logprobs`, `top_logprobs` | 400 | Not implemented on the Dynamo vLLM chat processor. |
| `previous_response_id` (501), `GET /v1/responses/{id}` (404) | — | The Responses API is stateless here. `store: true` is echoed back but nothing is persisted, so reuse the response from the original `POST`. |

### SGLang B200 profile

`deploy.yaml` overrides `DYN_FORWARDPASS_METRIC_PORT` to an empty value. This override is
required for MTP/EAGLE speculative decoding with SGLang because the forward-pass metrics reporter
(auto-enabled by the operator-injected port) crashes the scheduler on speculative batches
(`batch.seq_lens_cpu` is `None`). Only per-forward-pass telemetry is lost. Remove the override
once the container image carries a fix.
