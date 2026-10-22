---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Reference Guide
subtitle: Configuration, arguments, and operational details for the vLLM backend
---

## Overview

The vLLM backend in Dynamo integrates [vLLM](https://github.com/vllm-project/vllm) engines into Dynamo's distributed runtime, enabling disaggregated serving, KV-aware routing, and request cancellation. Dynamo leverages vLLM's native KV cache events, NIXL-based transfer mechanisms, and metric reporting.

Dynamo vLLM uses vLLM's native argument parser — all vLLM engine arguments are passed through directly. Dynamo adds its own arguments for disaggregation mode, KV transfer, and prompt embeddings.

## Argument Reference

The vLLM backend accepts all upstream vLLM engine arguments plus Dynamo-specific arguments. The authoritative source is always the CLI:

```bash
python -m dynamo.vllm --help
```

The `--help` output is organized into the following groups:

- **Dynamo Runtime Options** — Namespace, discovery backend, request/event plane, endpoint types, tool/reasoning parsers, and custom chat templates. These are common across all Dynamo backends and use `DYN_*` env vars. See [Runtime Configuration](../../../../../reference/components/runtime-configuration.mdx) for the full field reference.
- **Dynamo vLLM Options** — Disaggregation mode, tokenizer selection, sleep mode, multimodal flags, vLLM-Omni pipeline configuration, headless mode, and ModelExpress. These use `DYN_VLLM_*` env vars. See [vLLM Configuration](../../../../../reference/backends/vllm-configuration.mdx) for the full field reference.
- **vLLM Engine Options** — All native vLLM arguments (`--model`, `--tensor-parallel-size`, `--kv-transfer-config`, `--kv-events-config`, `--enable-prefix-caching`, etc.). See the [vLLM serve args documentation](https://docs.vllm.ai/en/stable/configuration/serve_args.html).

### Tool and Reasoning Parsers

Use `--dyn-tool-call-parser` and `--dyn-reasoning-parser` to match the model's output format when the model emits tool calls and/or reasoning content. The current supported values are documented in [Tool Call Parsing (Dynamo)](../../../../../use-cases/tool-calling-and-reasoning/tool-call-parsing.mdx#supported-tool-call-parsers) and [Reasoning Parsing (Dynamo)](../../../../../use-cases/tool-calling-and-reasoning/reasoning-parsing.md#supported-reasoning-parsers).

To set the thinking mode used when a request omits an explicit control, pass
`--dyn-default-thinking-mode enabled|disabled` or set
`DYN_DEFAULT_THINKING_MODE`. Request-level thinking controls, including
adaptive thinking, take precedence. See
[Deployment-Level Thinking Default](../../../../../use-cases/tool-calling-and-reasoning/reasoning-parsing.md#deployment-level-thinking-default).

For reasoning models with structured output (`response_format`, JSON schema,
or required/named tool choice), configure both reasoning parsers on the worker:

```bash
python -m dynamo.vllm --model <model> \
  --reasoning-parser <vllm-parser> \
  --dyn-reasoning-parser <dynamo-parser>
```

The vLLM parser delays grammar enforcement until reasoning ends; the Dynamo
parser populates `reasoning_content`. Parser names can differ between registries.

### Priority Scheduling

vLLM engine-level request priority is controlled by the upstream vLLM
`--scheduling-policy priority` argument.

```bash
python -m dynamo.vllm \
    --model <model> \
    --scheduling-policy priority
```

Clients still send the Dynamo API value directly:
`nvext.agent_hints.priority`. Higher values mean higher priority at the Dynamo
API layer. Dynamo converts that value before passing it to vLLM, which uses a
different native priority polarity internally.

Do not negate `nvext.agent_hints.priority` in the client for vLLM. If you are
also using the router queue, configure the frontend-side
`--router-queue-threshold` separately; vLLM engine scheduling only applies
after a request reaches the worker.

For the cross-layer behavior, see
[Priority Scheduling](../../../../../use-cases/agents/priority-scheduling.md). For the upstream
flag definition, see the
[vLLM serve args documentation](https://docs.vllm.ai/en/stable/configuration/serve_args.html).

### Prompt Embeddings

Dynamo supports [vLLM prompt embeddings](https://docs.vllm.ai/en/stable/features/prompt_embeds.html) — pre-computed embeddings bypass tokenization in the Rust frontend and are decoded to tensors in the worker.

- Enable with `--enable-prompt-embeds` (disabled by default)
- Embeddings are sent as base64-encoded PyTorch tensors via the `prompt_embeds` field in the Completions API
- NATS must be configured with a 15MB max payload for large embeddings (already set in default deployments)

### Custom JSON Content

**Experimental, unreleased.** Aggregated Python vLLM token workers started with
`--enable-multimodal` accept explicit custom parts in user and tool messages:

```json
{
  "model": "your-model",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Predict this: <chemistry>"},
      {"type": "custom", "kind": "chemistry", "fields": {
        "chemistry": {"data_base64": "..."}, "units": "angstrom"
      }}
    ]
  }],
  "max_tokens": 64
}
```

`kind` is a non-empty string. The built-in wire names `text`, `image_url`,
`video_url`, `audio_url`, and `input_audio` are reserved and rejected as custom
kinds, including when mixed with typed media. `fields` is a JSON object whose values may be any
JSON, including arrays and null. Dynamo passes the **complete fields object** as
`multi_modal_data[kind]`: the example plugin receives
`{"chemistry": {"data_base64": "..."}, "units": "angstrom"}` under the chemistry
modality. It does **not** select `fields[kind]`. This deliberately revises the
previous draft's matching-field forwarding contract; update draft clients to the
explicit envelope. Repeated parts of one kind become an ordered list of fields
objects. Built-in image, video and audio parts retain their existing typed schema
and validation. They may share requests with custom parts when their engine
modality names do not collide.

The model plugin owns domain validation and interpretation. Install or mount it
in the worker environment. Dynamo excludes custom parts before serializing chat
templates and auxiliary messages, keeping the payload in `multi_modal_data` only.
Put processor placeholders such as `<chemistry>` in text parts. For a pre-rendered
prompt, set `nvext.use_raw_prompt: true` and provide exactly one user message;
Dynamo concatenates its text parts without another chat template. For token IDs,
use `nvext.token_data`. Engine token IDs/logprobs remain available through
`nvext.engine_data` independently of parser-rewritten choice text.

#### Raw Tensor Adapter

Arbitrary JSON does not require tensors. To use Dynamo's raw tensor adapter,
place a reserved `{"$dynamo_tensor": {...}}` object anywhere inside `fields`.
The adapter replaces only these objects with CPU PyTorch tensors before the
model's vLLM processor runs. Other JSON and item boundaries remain intact.
The reserved object cannot have sibling keys. Tensor descriptors use version 1,
little-endian bytes, C order, at most 32 non-negative dimensions, and at most
64 MiB per tensor, 128 MiB of resolved tensor bytes per request, and 32 tensor
occurrences per request. Repeated references count each time; empty tensors
also count toward the item limit. Admission validates the complete custom input
across modalities before decoding, reading, or allocating any tensor storage.
Supported dtypes are `float16`, `bfloat16`, `float32`,
`float64`, `int8`, `uint8`, `int16`, `int32`, `int64` and `bool`.

Worker preparation runs off the async event loop in a process-wide pool with
at most two concurrent requests and no pending work queue. Excess requests
receive `WorkerOverloaded`. Each admitted request resolves tensors serially.
Cancelling a waiter retains its capacity slot until the running resolution
finishes and releases its unconsumed private storage. Cancellation cannot stop
a Python thread in the middle of a decode or SHM read.

Inline representation (two float32 values):

```json
{"$dynamo_tensor": {
  "version": 1, "transport": "inline", "dtype": "float32", "shape": [2],
  "order": "C", "data_base64": "AACAPwAAAEA="
}}
```

For shared-memory references, use the producer API and retain the lease across
the HTTP request and any retries:

```python
from dynamo.common.multimodal.custom_tensor import ShmTensorPublisher, inline_tensor

# tensor is a PyTorch tensor; worker_id is the discovered aggregated worker ID.
with ShmTensorPublisher(tensor, worker_id=worker_id, ttl=60) as lease:
    part = {"type": "custom", "kind": "chemistry", "fields": {"input": lease.envelope()}}
    # Send part in messages, with nvext.decode_worker_id = worker_id.
    # Wait for the response (or handle cancellation/failure) inside this context.

# The equivalent inline input uses {"input": inline_tensor(tensor)}.
```

The emitted reference contains `version`, `transport: "shm"`, `dtype`, `shape`,
`order`, `name`, `size`, `placement`, `worker_id`, `expires_at` (Unix seconds),
and `sha256`. It contains no bulk bytes or inline fallback. Unknown descriptor
fields, unsupported transports, invalid dimensions, mismatched byte counts,
placement, checksum, and expired/released references produce explicit errors.

Shared-memory mode requires Linux, the **same host boot, IPC namespace and
shared-memory mount** for producer and worker, and permission to read the segment.
Containers require an explicitly shared IPC namespace/mount. The frontend need
not share that placement: it forwards the descriptor only. Set
`nvext.decode_worker_id` to the descriptor's worker ID; the frontend rejects a
missing/mismatched pin and workers without `custom_tensor_shm_v1`. Routing validates
the worker is discovered; a missing/unavailable target must fail, not select a
remote replacement. The worker rechecks placement and expiry before reading.
Header routing overrides that also set `backend_instance_id` are unsupported for
custom content; use the body field. Request migration is unsupported for these
leases. Retry against the pinned worker before expiry or create a fresh lease.

The producer owns close/unlink on response, failure and cancellation through the
context manager. A daemon timer unlinks abandoned leases at TTL (default 60 s,
maximum 3600 s). Multiple readers/retries are valid while the lease is alive.
Readers copy to private writable memory and close their mappings before returning;
producer release cannot invalidate a consumed model input. Process crashes rely
on Python's producer resource tracker; no worker unlinks producer allocations.

This reuses the shared-memory byte transfer used by processed `mm_kwargs` without
using that vLLM-specific serializer. GPU producer tensors stage to CPU; noncontiguous
inputs become contiguous; tensor bytes are copied into SHM, copied into private
worker storage, then moved by the plugin/engine if GPU input is needed. Inline
mode adds base64 encoding, JSON serialization and decoding. Neither path claims
zero-copy GPU transport.

#### Capability and Transport Limits

Selected workers must advertise `custom_json_v1`; missing capability yields an
explicit request error. Enabled capabilities partition discovery cohorts, including
LoRA registrations. Custom P/D, encoder disaggregation, speculative prefill,
custom caching/KVBM, SGLang, TensorRT-LLM and native vLLM sidecars are unsupported.
Existing typed media paths and their transports remain unchanged.

#### NIXL References

NIXL references use the same maintained sender, receiver, registration and
completion machinery as embedding transfers. Multimodal aggregated vLLM workers
with NIXL installed advertise `custom_tensor_nixl_v1`; no additional opt-in is
required. Discover a compatible worker and pin the request to its ID:

```python
from dynamo.common.multimodal.custom_tensor_nixl import NixlTensorPublisher

async with NixlTensorPublisher(tensor, worker_id=worker_id, ttl=60) as publisher:
    part = {"type": "custom", "kind": "chemistry", "fields": {"input": publisher.envelope()}}
    # Send part with nvext.decode_worker_id = worker_id and check the HTTP response.
```

The envelope contains `transport: "nixl"`, the tensor spec, byte count, checksum,
worker ID, expiry, and the maintained transfer descriptor in `serialized_request`.
Bulk bytes travel through NIXL. Producer tensors stage to contiguous CPU bytes;
the receiver copies into private model storage before calling `release_tensor`.
The original shape and dtype are restored, including scalars. Plugins retain all
other JSON fields and own domain interpretation and any subsequent GPU transfer.

Each reference permits one READ; use a fresh publisher for retries. Expiry is an
admission check. Source lifetime follows the existing embedding completion
waiter, including cancellation and failed-reader limitations; no custom abort or
reclamation protocol is added. Native connectivity and transport availability
follow the deployment's existing NIXL configuration. Cross-host CPU transfer over
UCX/TCP is tested; GPU-direct transport is not claimed.

## Hashing Consistency for KV Events

When using KV-aware routing, ensure deterministic hashing across processes to avoid radix tree mismatches. Choose one of the following:

- Set `PYTHONHASHSEED=0` for all vLLM processes when relying on Python's built-in hashing for prefix caching.
- If your vLLM version supports it, configure a deterministic prefix caching algorithm:

```bash
vllm serve ... --enable-prefix-caching --prefix-caching-algo sha256
```

See the high-level notes in [Router Design](../../router/router-design.md#deterministic-event-ids) on deterministic event IDs.

## Graceful Shutdown

vLLM workers use Dynamo's graceful shutdown mechanism. When a `SIGTERM` or `SIGINT` is received:

1. **Discovery unregister**: The worker is removed from service discovery so no new requests are routed to it
2. **Grace period**: In-flight requests are allowed to complete (configurable via `DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS`, default 5s)
3. **Resource cleanup**: Engine resources and temporary files (Prometheus dirs, LoRA adapters) are released

All vLLM endpoints use `graceful_shutdown=True`, meaning they wait for in-flight requests to finish before exiting. An internal `VllmEngineMonitor` also checks engine health every 2 seconds and initiates shutdown if the engine becomes unresponsive.

For more details, see [Graceful Shutdown](../../../../../kubernetes/fault-tolerance/graceful-shutdown.md).

## Health Checks

Each worker type has a specialized health check payload that validates the full inference pipeline:

| Worker Type | Health Check Strategy |
|------------|----------------------|
| Decode / Aggregated | Short generation request (`max_tokens=1`) using the model's BOS token |
| Prefill | Same payload structure as decode, adapted for prefill request format |
| vLLM-Omni | Short generation request via AsyncOmni with the model's BOS token |

Health checks are registered with the Dynamo runtime and called by the frontend or Kubernetes liveness probes. The payload can be overridden via `DYN_HEALTH_CHECK_PAYLOAD` environment variable. See [Observability Architecture](../../../concepts/observability-architecture.md#active-worker-health-checks) for the active health-check design.

## Request Cancellation

When a user cancels a request (e.g., by disconnecting from the frontend), the request is automatically cancelled across all workers, freeing compute resources.

| | Prefill | Decode |
|-|---------|--------|
| **Aggregated** | ✅ | ✅ |
| **Disaggregated** | ✅ | ✅ |

For more details, see the [Request Cancellation Architecture](../../../concepts/fault-tolerance/request-cancellation-architecture.md) documentation.

## Request Migration

Dynamo supports [request migration](../../../../../kubernetes/fault-tolerance/request-migration.md) to handle worker failures gracefully. When enabled, requests can be automatically migrated to healthy workers if a worker fails mid-generation. See the [Request Migration Architecture](../../../../../kubernetes/fault-tolerance/request-migration.md) documentation for configuration details.

## See Also

- **[Examples](../../../../../recipes/cli-templates/vllm.mdx)**: Local deployment launch scripts
- **[vLLM README](overview.md)**: Quick start and feature overview
- **[Observability](observability.md)**: Metrics and monitoring setup
- **[Configuration and Tuning](../../router/configuration-and-tuning.md)**: KV-aware routing configuration
- **[Fault Tolerance](../../../../../kubernetes/fault-tolerance/introduction.md)**: Request migration, cancellation, and graceful shutdown
