---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: RL Integration Reference
subtitle: Generation, worker discovery, administration, and compatibility contracts for RL frameworks
---

Use this reference when implementing or reviewing a Dynamo rollout adapter. It defines the shared serving contract once; framework guides describe only what differs for that integration.

> [!NOTE]
> This reference tracks the current `dev` documentation and Dynamo `main`. For a released version, use the matching versioned documentation and confirm that the selected backend advertises the required routes.

## Separate the Three Planes

| Plane | Endpoint | Use it for |
|---|---|---|
| Request | Dynamo frontend, normally port `8000` | Generation, streaming, cancellation, and routing |
| Discovery | RL listener, normally port `8001` | Read live worker identity, routes, and optional topology metadata |
| Administration | Returned `system_url` when present, or another trusted worker URL supplied by the deployment | Pause, resume, weight operations, health checks, and backend-specific controls |

Send rollout inference through the shared frontend. Send mutating operations only to selected workers. The discovery endpoint is read-only and does not create a fleet-wide transaction.

> [!WARNING]
> Discovery and worker administration do not add a separate authentication layer. Keep them on a trusted orchestrator network and expose only the backend methods the integration requires.

## Choose a Request Interface

| Requirement | Interface | Notes |
|---|---|---|
| Native SGLang token streaming | `POST /generate` | Preserves supported SGLang token-input and streaming response fields. |
| Cross-backend completions | `POST /v1/completions` | Accepts token arrays and supports selected NVIDIA request-extension fields. |
| Cross-backend chat | `POST /v1/chat/completions` | Uses messages or bypasses frontend tokenization with `nvext.token_data`. |
| RL worker discovery | `GET /v1/rl/workers` | Returns protocol version `1` and live worker descriptors. |
| Direct worker operation | `POST` to the advertised route under `system_url/engine/` | Calls one selected worker with a backend-specific request body. |

Use the native SGLang route when the framework already speaks that schema. Use an OpenAI-compatible route when the adapter needs one request envelope across backends or named `nvext` response fields.

## Preserve Token Authority

The engine token sequence used for training is authoritative. Do not reconstruct it by tokenizing generated text; chat templates, normalization, special tokens, and tokenizer versions can change the result.

For OpenAI-compatible completions, send token IDs and request generated token IDs explicitly:

```bash
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": [151644, 8948, 198],
    "max_tokens": 32,
    "temperature": 0,
    "logprobs": 0,
    "nvext": {"extra_fields": ["completion_token_ids"]}
  }'
```

Before admitting a sample to training, verify:

1. Generated token IDs are present and contain the expected number of choices.
2. Selected log probabilities match those token IDs in length and order.
3. Prompt log probabilities preserve undefined positions instead of shifting alignment.
4. The response has a terminal state recognized by the framework.
5. The tokenizer and model identity match the selected rollout worker.

`POST /v1/responses/input_tokens` estimates input size for preflight decisions. It does not return authoritative token IDs, verify the model or tokenizer, or prove that a worker is ready.

## Know What Returns to the Trainer

| Data | Dynamo surface | Contract boundary |
|---|---|---|
| Prompt token IDs | Named `nvext.prompt_token_ids` | Returns the effective single-prompt token sequence used after preprocessing, including token arrays or `nvext.token_data` supplied through the request. |
| Generated token IDs | Named `nvext.completion_token_ids`, or the native SGLang stream | Use the engine-returned sequence; exact placement depends on the selected interface. |
| Selected and prompt log probabilities | Standard completion log probabilities, named `nvext.prompt_logprobs`, or native SGLang metadata | Check support and alignment on the exact backend and response path. |
| Routed experts and raw engine data | Opt-in `nvext.routed_experts` or `nvext.engine_data` | Backend-specific. Prefer named fields over the raw engine payload. |
| Large SGLang `meta_info` | `nvext.metadata_upload.url` on the OpenAI-compatible path | Uploaded out of band by an RL-enabled SGLang worker; the destination is trusted control input. |
| Masks, rewards, advantages, tool or environment state, and trajectory objects | Not a generic Dynamo response contract | The framework derives, stores, validates, and accepts these values. |

See [NVIDIA Request Extensions](../../developer-guide/additional-resources/nvidia-request-extensions-nvext.md) for the complete `nvext` request and response shapes.

## Handle Streaming and Retries

Treat a streaming request as a state machine: accepted, streaming, terminal success, canceled, or failed. Admit only a verified terminal result. Preserve attempt identity when retrying, because a timed-out request may have completed after the client stopped waiting and generation is not idempotent.

The framework owns retry policy, duplicate suppression, partial-rollout handling, and sample acceptance. Dynamo propagates supported cancellations and serving failures but does not decide whether a partial attempt is scoreable.

## Discover Workers Safely

Start the RL listener and an RL-enabled vLLM worker on the trusted control network:

```bash
DYN_ENABLE_RL=true DYN_RL_PORT=8001 python -m dynamo.frontend
```

```bash
DYN_SYSTEM_PORT=8081 python -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --enable-rl
```

```bash
curl http://localhost:8001/v1/rl/workers
```

A protocol version `1` response has this shape. Optional fields such as `model`, `system_url`, `admin_base_url`, and `world_size` appear only when the worker provides them:

```json
{
  "protocol_version": 1,
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

Check `protocol_version` before reading the worker list. In protocol version `1`:

| Field | Contract |
|---|---|
| `namespace`, `component`, `endpoint`, `instance_id`, `transport`, `request_plane_url` | Stable Dynamo endpoint identity |
| `routes` | Worker-advertised Dynamo administration routes |
| `system_url` | Optional direct Dynamo worker URL; never derive it from the request-plane URL |
| `model` | Optional model identity; it can be absent when metadata is unavailable or ambiguous |
| `world_size`, `admin_base_url` | Optional producer-supplied transfer metadata; not a rank map or transfer-backend declaration |
| `error` | Probe failure; fail closed when a required worker or route is unavailable |

Refresh discovery before each control phase and require the complete capability set for the selected update path. Do not cache membership indefinitely or interpret list position as worker identity.

## Coordinate Policy Refresh

The framework owns the fleet-level lifecycle:

1. Gate new rollout work for the target workers.
2. Resolve the current worker set and required capabilities.
3. Pause or drain workers when the backend requires it.
4. Transfer and apply one target policy.
5. Invalidate stale KV state.
6. Verify every worker and run post-update generation before reopening the fleet.

Per-worker success is not fleet-wide atomicity. Keep generation gated when membership changes, an update fails, cache reset fails, or post-update generation does not pass. See [Distribute and Update Rollout Weights](weight-updates.md) for the supported paths and recovery rules.

## Compare vLLM and SGLang Administration

The backends do not share one administration schema. Use the exact route returned by discovery or configured by the deployment, and validate request bodies against the installed backend version.

| Operation | vLLM | SGLang |
|---|---|---|
| Generation | OpenAI-compatible frontend routes; a native unary path is also experimental | Native streaming `/generate` or OpenAI-compatible frontend routes |
| Worker discovery | RL-enabled Python workers and the native sidecar register with `/v1/rl/workers` | Not currently registered; obtain trusted worker URLs from the framework or deployment |
| Administration route family | Python workers advertise names such as `/engine/pause_generation`; the native sidecar advertises `/engine/control/*` and `/engine/update/*` | Built-in controls use `/engine/control/*` |
| Stop and resume work | Python `pause_generation` / `resume_generation`; native-sidecar control routes reflect the installed vLLM RL API | `release_memory_occupation` unregisters and drains the worker; `resume_memory_occupation` restores it |
| Clear KV state | Python `flush_cache`; native-sidecar behavior follows the advertised lifecycle route and vLLM configuration | `clear_kv_blocks` is a Dynamo worker request-plane endpoint, not a built-in `/engine/control/*` route, and rejects active requests |
| Apply weights | Python disk or distributed update routes and group lifecycle; native sidecar init/start/update/finish routes when weight transfer is enabled | Built-in disk, tensor, distributed, or IPC update routes using the installed SGLang request schemas |
| Weight version | Python `get_weight_version` reads caller-supplied update metadata; native sidecar exposes get/update controls | `update_weight_version` changes metadata and can abort requests; it does not replace tensors |
| Custom controls | Python integrations can register and advertise trusted engine routes | Allowlist methods with `--engine-route` or `DYN_SGLANG_ENGINE_ROUTES` using <code>path[=method][:engine&#124;tm]</code> |
| Success body | Python RL routes commonly return a `status` field; native-sidecar routes follow vLLM's RL schemas | Built-in update routes commonly return `success` and `message`; check both HTTP status and body |

Even two vLLM deployments can expose different route families because the Python worker and native sidecar adapt different backend control APIs. Never prepend, remove, or rename route segments returned in `routes`.

## Framework Compatibility

| Framework | Status | Current Dynamo path | Key boundary |
|---|---|---|---|
| [verl](verl.md) | Experimental | Shared Dynamo frontend with colocated vLLM rollout workers | The public recipe owns CUDA IPC updates through Ray/ZMQ control. Choose native Dynamo routing or ThunderAgent as distinct variants. |
| [NeMo RL](nemo-rl.md) | Experimental | NeMo RL-managed Dynamo/vLLM fleet on Slurm and Ray | Fixed non-colocated fleet with framework-owned NCCL refit; not an external Dynamo deployment. |
| [SLIME](https://github.com/THUDM/slime) | Integration in progress | Proposed shared SGLang `/generate` path with direct engine control | No merged, maintained Dynamo recipe yet; discovery and update ownership remain unsettled. |
| [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl) | Router available; full integration in progress | Prime-RL documents the Dynamo router as a drop-in option; a proposed Dynamo/vLLM sidecar adds worker discovery and external updates | Routing can be evaluated today, but the full adapter remains in upstream development and is not a released compatibility surface. |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [Miles](https://github.com/fleet-ai/miles-fleet), [SkyRL](https://github.com/NovaSky-AI/SkyRL), and [Polar](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server) | No Dynamo guide | No maintained Dynamo adapter was found in the reviewed public sources | Add a guide only after a maintained integration completes the same generation, update, failure, and ownership checks. |

A status reflects the documented integration, not whether the framework can call a generic HTTP endpoint.

## Backend Compatibility

| Capability | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| Token input through OpenAI-compatible routes | Supported | Supported | Supported |
| Completion token IDs through named `nvext` fields | Supported | Supported | Supported through the shared response path |
| Native RL generation route | Experimental unary vLLM-compatible route | Streaming `/generate` | None |
| Prompt log probabilities | Supported | Topology-dependent; validate the selected path | No dedicated RL end-to-end coverage recorded here |
| `/v1/rl/workers` registration | RL-enabled workers | Not currently registered | Not currently registered |
| Built-in RL administration routes | Pause/resume, version, disk/distributed update, group lifecycle | Fixed weight controls plus explicit method allowlisting | No shared RL administration contract documented here |

Backend support does not imply every topology or framework combination is validated. Record aggregated or prefill/decode serving, placement, model parallelism, model class, cancellation, discovery, weight layout, and cache behavior for the path you test.

## Integration Checklist

Before publishing a runnable integration, verify:

- [ ] Generation reaches the intended Dynamo frontend and backend.
- [ ] Token IDs, log probabilities, masks, and terminal states match the framework contract.
- [ ] Retries and cancellations cannot admit the same sample twice.
- [ ] Discovery is refreshed and every administration route is negotiated.
- [ ] Mutating calls use direct worker URLs on a trusted network.
- [ ] One complete training iteration includes policy refresh and post-update generation.
- [ ] Request, worker, and update failures have a tested recovery path.
- [ ] Framework identity can be correlated with Dynamo telemetry without high-cardinality metric labels.
- [ ] The tested versions, environment, and topology are recorded with the integration.
