<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Flint with vLLM

Run `RecursiveSI/flint` through NVIDIA Dynamo with vLLM-native chat processing,
reasoning, and tool calling.

> [!WARNING]
> Experimental. This recipe requires a Dynamo runtime image containing the
> implementation from [vLLM PR #62938](https://github.com/vllm-project/vllm/pull/62938).
> The repository's default vLLM 0.31.0 image does not contain that implementation.
> This recipe has not been validated on a GPU or Kubernetes cluster.

## Configuration

| Setting | Value |
| --- | --- |
| Checkpoint | `RecursiveSI/flint`, BF16 |
| Hardware target | One B200, B300, or GB200 GPU per worker |
| Topology | Aggregated, tensor parallel size 1 |
| KV cache | BF16, selected by `--dtype bfloat16 --kv-cache-dtype auto` |
| Context limit | 262,144 tokens, shared by prompt and output |
| Concurrency limit | 4 sequences per worker; qualify for the target workload |
| Routing | Round-robin |
| Chat processing | `--dyn-chat-processor vllm` |
| Tool and reasoning parsers | `glm47`, on the frontend |

The published vLLM configuration uses `glm47` for both parsers and
`--exclude-tools-when-tool-choice-none`. The frontend owns those options; the worker
loads the model and generates tokens. See the
[vLLM day-zero announcement](https://vllm.ai/blog/2026-10-21-flint).

This recipe covers text chat through `/v1/chat/completions`. NVFP4 GGUF,
disaggregated prefill/decode, KV-aware routing, KV cache offloading, speculative
decoding, and multi-GPU parallelism require separate qualification.

## Build the runtime image

Run from the Dynamo repository root. Use the
[container build prerequisites](https://github.com/ai-dynamo/dynamo/blob/main/container/README.md).
Select an upstream CUDA 13 vLLM image tag containing the merged Flint implementation.
The `flint` day-zero tag is available for development before a release includes it.

```bash
export VLLM_IMAGE_TAG=flint
export DYNAMO_FLINT_IMAGE=registry.example.com/dynamo-flint-vllm:dev
python3 container/render.py --framework vllm --target runtime --output-short-filename
docker build -f container/rendered.Dockerfile \
  --build-arg RUNTIME_IMAGE_TAG="$VLLM_IMAGE_TAG" \
  -t "$DYNAMO_FLINT_IMAGE" .
```

The normal Dynamo runtime build preserves the upstream vLLM stack. Do not install
`ai-dynamo[vllm]` afterward: its version pin would replace the Flint-enabled engine.
Use the same built image for the frontend and worker. Record the upstream image
digest, vLLM commit, and Dynamo commit when qualifying the recipe.

Check model registration and Dynamo's frontend and worker imports before pushing:

```bash
docker run --rm --entrypoint python3 "$DYNAMO_FLINT_IMAGE" -c '
from vllm.model_executor.models import ModelRegistry
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager
assert "FlintForCausalLM" in ModelRegistry.get_supported_archs()
ReasoningParserManager.get_reasoning_parser("glm47")
ToolParserManager.get_tool_parser("glm47")
import dynamo.frontend.vllm_processor
import dynamo.vllm.main
print("Flint registration, parsers, and Dynamo imports passed")
'
docker push "$DYNAMO_FLINT_IMAGE"
```

A passing import check does not establish runtime compatibility with a newer vLLM.
Run the smoke checks below before treating an image as supported.

## Prepare the model cache

Use an existing namespace-local `shared-model-cache` PVC and populate its Hugging
Face cache before deployment. The worker mounts it read-only at
`/shared-model-cache`; both components run offline. Allow at least 80 GB for the
BF16 checkpoint, plus cache overhead.

In a download environment that mounts the PVC at `/shared-model-cache`, run:

```bash
export HF_HOME=/shared-model-cache
export HF_HUB_CACHE=/shared-model-cache/hub
hf download RecursiveSI/flint
```

Record the downloaded model revision. Keep Hugging Face's cache layout, including
the snapshot and blob directories, so offline loading can resolve the model ID.
Cluster owners supply PVC provisioning, image credentials, and GPU scheduling.

## Deploy

Install the Dynamo Kubernetes operator with `nvidia.com/v1beta1` support. Copy the
[beta cluster Kustomization starter](https://github.com/ai-dynamo/dynamo/tree/main/recipes/templates/kustomize)
and reference
[`vllm/agg/deploy.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/recipes/flint/vllm/agg/deploy.yaml).
Set both component images to the image built above in your cluster Kustomization.
Keep cluster-specific values outside the portable recipe.

After rendering and validating your cluster composition, apply it to your namespace.
Wait for the frontend's `/health` endpoint to report readiness, then port-forward:

```bash
export NAMESPACE=your-namespace
kubectl port-forward -n "$NAMESPACE" svc/flint-vllm-agg-frontend 8000:8000
```

The manifest's `dynamo-flint-vllm:dev` image is a local development name, not a
published Dynamo image. Replace it on both components before deployment.

## Smoke checks

In another terminal, run from the repository root:

```bash
python3 -m pip install openai
python3 recipes/flint/smoke_test.py --base-url http://localhost:8000/v1
```

The script fails on HTTP errors, truncated turns, missing final content, invalid
tool arguments, or missing tool-call IDs. It checks ordinary and streaming chat,
automatic and required tool calls, a tool result, and a streaming named tool call.
It preserves returned `reasoning_content` and tool arguments in conversation history.
The weather tool returns a local fixture; the script does not execute model-selected
commands or contact a weather service.

Sampling follows the published settings: temperature 1.0, top-p 0.95, and top-k 20.
The default output budget is 32,768 tokens, including reasoning. Increase
`--max-tokens` for requests that end with `finish_reason: length`.

## Validation before release

Qualify parallel calls, structured output, reasoning and
non-reasoning turns, disconnect cancellation, and routing across two worker
replicas in addition to the smoke script. Measure memory use and latency at the
intended context length and concurrency. No Dynamo throughput or latency claims
are available for this recipe.
