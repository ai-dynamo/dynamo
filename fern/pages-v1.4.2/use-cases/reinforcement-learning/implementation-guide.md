---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: RL Implementation Guide
subtitle: Connect RL orchestrators to Dynamo inference and engine administration interfaces
---

**Experimental.** This guide is for engineers using Dynamo to serve RL rollouts during a training loop. It covers how to return token IDs and log probabilities, expose routing or engine metadata, discover live workers, and push weight updates without restarting the serving stack. For SGLang rollouts, use the SGLang-compatible `/generate` API as the primary inference path. Discover vLLM workers through the read-only RL discovery API. Send lifecycle or weight updates directly to the selected worker system URL.

Dynamo can add value to RL rollout serving when the rollout plane needs more than a static inference endpoint. Advanced routing can steer rollout traffic across heterogeneous workers, weight synchronization with Model Express can help move updated checkpoints into serving quickly, fault tolerance can keep rollout generation available across worker failures, and autoscaling can match serving capacity to changing rollout demand during training.

## What This Page Covers

| User need | Dynamo surface | Status |
|---|---|---|
| Serve SGLang token-in/token-out rollouts with native request and response shapes | Frontend `POST /generate` or `PUT /generate` | Experimental |
| Serve rollout generation through an OpenAI-compatible endpoint | Frontend `/v1/completions` and `/v1/chat/completions` | Available |
| Return token IDs, prompt log probabilities, and selected backend metadata | `nvext.token_data` and `nvext.extra_fields` | Available |
| Discover live rollout workers and their direct administration URLs | `/v1/rl/workers` on the frontend RL listener | vLLM only |
| Pause generation, update weights, and resume a selected worker | Direct `/engine/` routes on the worker system server | vLLM supported; SGLang uses backend-specific routes |
| Upload large SGLang rollout metadata out-of-band | `nvext.metadata_upload.url` | SGLang only |

## Backend Support

| Capability | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| Engine-native token-in/token-out API | `/inference/v1/generate` | `/generate` | Not supported |
| Token input through `prompt` token arrays | Supported | Supported | Supported |
| `nvext.token_data` tokenizer bypass | Supported | Supported | Supported |
| `completion_token_ids` response field | Supported | Supported | Supported |
| `prompt_logprobs` response field | Supported | Supported | Not supported |
| Routed expert response data | Compatible builds only | Compatible builds only | Not supported |
| `/v1/rl/workers` discovery | Supported with `--enable-rl` | Not supported | Not supported |
| Direct RL admin routes from discovery | Supported with `--enable-rl` | Not supported | Not supported |
| SGLang `meta_info` upload | Not applicable | Supported with `--enable-rl` | Not applicable |

Experimental means the RL integration surfaces are still converging around real framework usage. For SGLang, build new token-in/token-out integrations on `/generate` first. This API preserves the request fields and streaming response objects from the installed SGLang version. Use the OpenAI-compatible routes when an integration needs a cross-backend schema or `nvext.metadata_upload`. Treat `engine_data`, uploaded `meta_info`, direct `/engine/` route bodies, and custom route names as backend-specific interfaces. Their contracts can change independently.

## Choose an Interface

| Task | Interface | Default Port | Behavior |
|---|---|---:|---|
| Run SGLang token-in/token-out rollouts | `POST /generate` or `PUT /generate` | `8000` | Routes native SGLang requests through Dynamo and returns native streaming response objects. |
| Run rollouts through an OpenAI-compatible API | `POST /v1/completions` or `POST /v1/chat/completions` | `8000` | Provides a cross-backend request and response schema. |
| Discover RL workers | `GET /v1/rl/workers` | `8001` | Returns live vLLM workers, their advertised routes, and direct system URLs. |
| Administer one engine | `POST <system_url>/engine/<route>` | Worker-specific | Calls one worker without frontend routing or fan-out. |

The discovery API runs on a dedicated frontend listener. It is not mounted on the main frontend port, and it does not proxy `/engine/` calls. The request-plane URL in its response is used internally to query route descriptors; an RL orchestrator should use the returned `system_url` for administration.

<Warning>
The worker system server does not add an authentication layer to `/engine/` routes. Restrict the system port and RL discovery listener to the orchestrator network, and do not expose them through a public inference gateway.
</Warning>

## SGLang Happy Path

Use `/generate` for SGLang reinforcement learning clients that send token IDs and consume SGLang streaming responses. Aggregated and prefill/decode deployments accept the same request.

<Steps>
<Step title="Enable the SGLang-compatible API">

Set `DYN_SGLANG_ENABLE_GENERATE=1` on the Dynamo frontend:

```bash
DYN_SGLANG_ENABLE_GENERATE=1 python -m dynamo.frontend
```

The API uses `/generate` by default. To use a different path, set `DYN_HTTP_SVC_SGLANG_GENERATE_PATH` on the frontend.

</Step>
<Step title="Start a SGLang worker">

```bash
python -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B
```

When the worker accepts token input and serves a supported SGLang role, it advertises this capability. Aggregated and decode workers support chat or completions output. Prefill workers support prefill output.

Do not use `--use-sglang-tokenizer`. This flag selects text input. As a result, the worker does not advertise `/generate`. The `/generate` API does not require `--enable-rl`.

</Step>
<Step title="Send a token-input rollout">

```bash
curl -N http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "input_ids": [151644, 8948, 198],
    "sampling_params": {
      "max_new_tokens": 32,
      "temperature": 0,
      "n": 1
    },
    "stream": true,
    "return_logprob": true,
    "top_logprobs_num": 0
  }'
```

Dynamo forwards each SGLang streaming response object as a server-sent event. The stream ends with `[DONE]`.

`top_logprobs_num: 0` returns the selected-token log probability. To request top-k alternatives, set `DYN_SGL_ALLOW_TOP_LOGPROBS=1` on the worker. Then use a positive `top_logprobs_num` value.

</Step>
</Steps>

The current API has these limits:

- Provide one non-empty `input_ids` sequence.
- Set `stream` to `true`.
- Set `sampling_params.n` to `1`.
- Use token input. The API does not support text, batched, multimodal, or non-streaming requests.
- Use an aggregated deployment for prompt log probabilities. They are not parity-complete in prefill/decode deployments.

Dynamo preserves other public SGLang request fields. The worker validates them against the installed SGLang version. The frontend rejects Dynamo-owned bootstrap and routing fields. Dynamo injects those fields after it selects a worker.

## vLLM Happy Path

This is the shortest path for wiring an RL orchestrator to a vLLM rollout worker: start the frontend discovery listener, start an RL-enabled worker, discover its system URL, run a token-in rollout, then pause, update weights, and resume generation.

<Steps>
<Step title="Start the frontend discovery listener">

```bash
DYN_ENABLE_RL=true DYN_RL_PORT=8001 python -m dynamo.frontend
```

</Step>
<Step title="Start an RL-enabled vLLM worker">

```bash
DYN_SYSTEM_PORT=8081 python -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --enable-rl
```

</Step>
<Step title="Discover the worker system URL">

```bash
curl http://localhost:8001/v1/rl/workers
```

Use the returned `system_url` and `routes` for direct worker administration. Do not call mutating engine operations through the frontend.

</Step>
<Step title="Run a token-in rollout through the frontend">

```bash
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "",
    "max_tokens": 32,
    "temperature": 0,
    "logprobs": 5,
    "prompt_logprobs": 5,
    "nvext": {
      "token_data": [151644, 8948, 198],
      "extra_fields": ["completion_token_ids", "prompt_logprobs"]
    }
  }'
```

</Step>
<Step title="Pause, update weights, and resume">

```bash
worker_url=http://10.0.0.12:8081

curl --fail-with-body "$worker_url/engine/pause_generation" \
  -H 'Content-Type: application/json' \
  -d '{"mode": "keep", "clear_cache": false}'

curl --fail-with-body "$worker_url/engine/update_weights_from_disk" \
  -H 'Content-Type: application/json' \
  -d '{"model_path": "/models/checkpoint-42", "weight_version": "42"}'

curl --fail-with-body "$worker_url/engine/resume_generation" \
  -H 'Content-Type: application/json' \
  -d '{}'
```

The full vLLM example below adds result checks and an exit trap so a failed update still attempts to resume the worker.

</Step>
</Steps>

## Frontend Feature Support

The OpenAI-compatible completion routes provide a cross-backend token-in/token-out interface. For SGLang clients that use native request and response shapes, prefer `/generate`.

| Feature | Request | Response | Notes |
|---|---|---|---|
| Token input | Set `prompt` to an integer array on `/v1/completions`, or set `nvext.token_data` on a chat or completion request. | Standard completion response | `nvext.token_data` bypasses frontend tokenization. |
| Completion token IDs | Add `"completion_token_ids"` to `nvext.extra_fields`. | `nvext.completion_token_ids` | Requires one prompt and one generated choice. Streaming responses contain token deltas; non-streaming responses contain the concatenated IDs. |
| Completion log probabilities | Set `logprobs` on `/v1/completions`, or set `logprobs: true` and `top_logprobs` on `/v1/chat/completions`. | Standard `choices[].logprobs` | The selected engine must support the requested log probability mode. |
| Prompt log probabilities | Set top-level `prompt_logprobs` and add `"prompt_logprobs"` to `nvext.extra_fields`. | `nvext.prompt_logprobs` on the final response | The first prompt position is `null` because it has no preceding-token probability. |
| Prefix-cache salt | Set `nvext.cache_salt`. | No response field | vLLM includes the opaque salt in prompt cache keys, separating reuse between requests with different salts. |
| Routed expert data | Add `"routed_experts"` to `nvext.extra_fields`. | `nvext.routed_experts` on the final response | Requires routed-expert capture in the engine build and configuration. |
| Raw engine metadata | Add `"engine_data"` to `nvext.extra_fields`. | `nvext.engine_data` | Backend-specific and not a stable cross-backend schema. Prefer named fields when available. |
| SGLang `meta_info` upload | Set `nvext.metadata_upload.url`. | Out-of-band object per choice | Requires an RL-enabled SGLang worker and fsspec support. |

See [NVIDIA Request Extensions](../../developer-guide/additional-resources/nvidia-request-extensions-nvext.md) for the complete `nvext` reference.

Backend RL flags also select engine-specific behavior:

- On vLLM, `--enable-rl` registers the discovery and administration routes, selects processed log probabilities, and prevents model `generation_config.json` sampling overrides from changing RL token-in requests marked with `nvext.token_data`. Explicit request sampling values still apply.
- On SGLang, `--enable-rl` enables out-of-band `meta_info` upload and the `/engine/call_tokenizer_manager` passthrough route. SGLang also exposes control routes such as `/engine/control/update_weights_from_disk`, but SGLang workers do not currently register with `/v1/rl/workers`.

### Token-In/Token-Out Example

Send pre-tokenized input and request token IDs plus prompt and completion log probabilities:

```bash
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "",
    "max_tokens": 32,
    "temperature": 0,
    "logprobs": 5,
    "prompt_logprobs": 5,
    "nvext": {
      "token_data": [151644, 8948, 198],
      "extra_fields": ["completion_token_ids", "prompt_logprobs"]
    }
  }'
```

The relevant response fields have this shape:

```json
{
  "choices": [
    {
      "index": 0,
      "text": "...",
      "logprobs": {}
    }
  ],
  "nvext": {
    "completion_token_ids": [9707, 11],
    "prompt_logprobs": [
      null,
      {"8948": {"logprob": -0.42}},
      {"198": {"logprob": -0.31}}
    ]
  }
}
```

To use the chat route while retaining pre-tokenized input, provide the normal `messages` field and set `nvext.token_data` to the complete token sequence that the engine should receive. The frontend uses `token_data` instead of rendering and tokenizing `messages`.

### Routed Expert Data

Opt into routed-expert data per request:

```json
{
  "model": "my-moe-model",
  "prompt": [1, 2, 3],
  "max_tokens": 16,
  "nvext": {
    "extra_fields": ["completion_token_ids", "routed_experts"]
  }
}
```

The payload format is backend-specific:

- vLLM-compatible builds return an object containing base64 `data`, `shape`, `dtype`, and `start`. The `start` offset identifies the first returned routing row in the full prompt-plus-completion sequence.
- SGLang builds whose `async_generate` API supports `return_routed_experts` return the engine's base64 string. Start SGLang with `--enable-return-routed-experts` to request capture. Builds that omit this engine argument do not return the field.
- TensorRT-LLM does not currently return routed-expert data through this frontend extension.

Use `nvext.engine_data` only when the orchestrator must consume other backend-specific data. The named `completion_token_ids`, `prompt_logprobs`, and `routed_experts` fields provide more stable contracts.

## Upload SGLang Metadata

The SGLang metadata upload feature uses the OpenAI-compatible route and `nvext`. When the rollout pipeline must store large metadata objects outside the HTTP response, use this feature. SGLang can upload the final cumulative `meta_info` for each choice to any filesystem supported by the installed fsspec backend.

<Warning>
Treat `metadata_upload.url` as trusted RL control-plane input. The worker trims the value, checks that it is a non-empty string, and passes it to fsspec without restricting the storage scheme or destination. Do not allow untrusted inference callers to set this URL; fsspec can access local or remote storage with the worker's permissions and credentials.
</Warning>

Start the worker with RL support. Add the routed-expert flag only when the SGLang engine build supports it:

```bash
DYN_SYSTEM_PORT=8081 python -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B \
  --enable-rl \
  --enable-return-routed-experts
```

Set a unique upload directory for each request:

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "prompt": [151644, 8948, 198],
  "max_tokens": 32,
  "logprobs": 5,
  "nvext": {
    "metadata_upload": {
      "url": "s3://rl-rollouts/run-42/request-7"
    }
  }
}
```

The worker writes `choice_0.msgpack.zst`, `choice_1.msgpack.zst`, and so on beneath the URL. Each file contains a Zstandard-compressed MessagePack object:

```json
{
  "schema_version": 1,
  "metadata": {
    "id": "...",
    "finish_reason": {"type": "stop"},
    "output_token_logprobs": [],
    "output_top_logprobs": [],
    "routed_experts": "..."
  }
}
```

Install the matching fsspec extra for remote storage, such as `fsspec[s3]` for S3. The worker also requires `msgspec` and `zstandard`. Local URLs such as `file:///tmp/rollouts/request-7` are useful for development.

When `metadata_upload` is present, SGLang uploads only the final cumulative `meta_info` for each choice and omits inline log probabilities and routed-expert metadata. The final response waits for the upload. An upload failure fails the request, so use a durable destination and a unique URL to avoid overwriting another request's `choice_<index>.msgpack.zst` objects.

## Discover RL Workers

RL discovery currently covers RL-enabled vLLM workers. Start the frontend discovery listener and a vLLM worker with its system server enabled:

```bash
DYN_ENABLE_RL=true DYN_RL_PORT=8001 python -m dynamo.frontend
```

```bash
DYN_SYSTEM_PORT=8081 python -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --enable-rl
```

Query the dedicated discovery port:

```bash
curl http://localhost:8001/v1/rl/workers
```

The response describes each live worker and probes it for the routes available in that process:

```json
{
  "namespace": "dynamo",
  "workers": [
    {
      "namespace": "dynamo",
      "component": "backend",
      "endpoint": "rl",
      "instance_id": 12345,
      "transport": {"tcp": "10.0.0.12:1234/..."},
      "request_plane_url": "dyn://dynamo.backend.rl",
      "system_url": "http://10.0.0.12:8081",
      "model": "Qwen/Qwen3-0.6B",
      "routes": [
        "get_weight_version",
        "liveness_probe",
        "pause_generation",
        "resume_generation",
        "update_weights_from_disk"
      ]
    }
  ]
}
```

Treat `routes` as the capability list for that worker instead of assuming every engine exposes the same methods. Optional features such as Low-Rank Adaptation (LoRA) add routes only when enabled. Discovery can also return a worker with an `error` and an empty route list when its descriptor probe times out or fails.

The discovery listener uses these environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `DYN_ENABLE_RL` | `false` | Enables the frontend RL discovery listener. |
| `DYN_RL_PORT` | `8001` | Sets the dedicated discovery port. |
| `DYN_NAMESPACE` | `dynamo` | Limits discovery to one Dynamo namespace. |
| `DYN_RL_ENDPOINT` | `rl` | Selects the worker endpoint name. |
| `DYN_RL_COMPONENTS` | all components | Limits discovery to a comma-separated component list. |
| `DYN_RL_REQUEST_TIMEOUT_SECS` | `30` | Bounds each worker descriptor probe. |
| `DYN_RL_MAX_CONCURRENT_PROBES` | `32` | Limits concurrent worker probes across discovery requests. |

## Call vLLM Engine Routes Directly

Read `system_url` and `routes` from the discovery response, then call the selected vLLM worker. Send a JSON object even when the route does not require arguments:

```bash
curl http://10.0.0.12:8081/engine/liveness_probe \
  -H 'Content-Type: application/json' \
  -d '{}'
```

For a vLLM weight update cycle, pause the selected worker, call one of its advertised weight-update routes, validate the result, and resume generation. Install `jq` before running this example. The exit trap attempts to resume the worker if the update or another command fails:

```bash
set -euo pipefail

worker_url=http://10.0.0.12:8081

resume_generation() {
  response=$(curl --fail-with-body "$worker_url/engine/resume_generation" \
    -H 'Content-Type: application/json' \
    -d '{}')
  jq -e '.status == "ok"' <<<"$response" >/dev/null
}

pause_response=$(curl --fail-with-body "$worker_url/engine/pause_generation" \
  -H 'Content-Type: application/json' \
  -d '{"mode": "keep", "clear_cache": false}')
jq -e '.status == "ok"' <<<"$pause_response" >/dev/null
trap resume_generation EXIT

update_response=$(curl --fail-with-body "$worker_url/engine/update_weights_from_disk" \
  -H 'Content-Type: application/json' \
  -d '{"model_path": "/models/checkpoint-42", "weight_version": "42"}')
jq -e '.status == "ok"' <<<"$update_response" >/dev/null

resume_generation
trap - EXIT
```

Route request bodies and response fields are engine-specific. A callback can return `{"status":"error"}` with HTTP 200, while an exception in the callback produces HTTP 500. Check both the HTTP status and the JSON result before advancing the rollout state.

SGLang direct administration uses SGLang-specific routes and response shapes. For example, call `/engine/control/update_weights_from_disk` directly when you already know the worker URL, or call `/engine/call_tokenizer_manager` with `{"method":"update_weights_from_disk", ...}` for tokenizer-manager passthrough behavior.

The `/v1/rl/workers` endpoint is read-only. It intentionally does not expose `/v1/rl/engine` or `/v1/rl/engines` proxy routes, which prevents an accidental frontend fan-out of a mutating engine operation.

## Register a Custom Engine Route

Register an asynchronous Python callback on the worker's `DistributedRuntime`. The route name is appended to `/engine/` on the system server:

```python
from typing import Any


async def set_rollout_state(body: dict[str, Any]) -> dict[str, Any]:
    rollout_id = body.get("rollout_id")
    if not isinstance(rollout_id, str) or not rollout_id:
        return {"status": "error", "message": "rollout_id is required"}

    await engine.set_rollout_state(rollout_id)
    return {"status": "ok", "rollout_id": rollout_id}


runtime.register_engine_route("rl/set_rollout_state", set_rollout_state)
```

Enable the system server with `DYN_SYSTEM_PORT`, then call the route directly:

```bash
curl http://worker-host:8081/engine/rl/set_rollout_state \
  -H 'Content-Type: application/json' \
  -d '{"rollout_id": "run-42-step-7"}'
```

`register_engine_route` makes the HTTP route callable but does not automatically advertise it in the RL discovery response. When extending an RL-enabled vLLM worker, register and advertise the route together with the shared helper:

```python
from dynamo.common.rl import register_rl_routes


register_rl_routes(
    runtime,
    handler.rl_route_registry,
    {"rl/set_rollout_state": set_rollout_state},
    enable_dispatch=handler.config.enable_rl,
)
```

The next `GET /v1/rl/workers` probe includes `rl/set_rollout_state` in that worker's `routes` list.
