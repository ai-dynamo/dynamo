---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Realtime Bidirectional Python Workers
sidebar-title: Realtime Python Workers
subtitle: Write streaming-input, streaming-output workers for the OpenAI Realtime API
---

Realtime workers serve the OpenAI Realtime WebSocket path through a Dynamo bidirectional engine: the client streams JSON events into `/v1/realtime`, the frontend forwards those events to the selected worker, and the worker streams JSON server events back.

Use this path when one request is a long-lived session with more than one client frame, such as audio input, tool/control events, or stateful realtime interaction. For single request to streaming response workloads, use the regular `endpoint.serve_endpoint()` Python worker path instead.

## Worker Shape

A realtime Python worker has the same outer lifecycle as other lower-level Python workers:

1. Create or receive a `DistributedRuntime`.
2. Get an endpoint with `runtime.endpoint("namespace.component.endpoint")`.
3. Register the model with `ModelInput.Text` and `ModelType.Realtime`.
4. Attach an async generator with `endpoint.serve_bidirectional_endpoint()`.

For a standalone realtime worker, use `worker_type=WorkerType.Aggregated`. The `model_name` must match the model name the client sends in the first usable `session.update` event.

```python
from __future__ import annotations

import asyncio
import uuid

import uvloop

from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import DistributedRuntime, dynamo_worker

MODEL_NAME = "realtime-echo"
ENDPOINT_PATH = "realtime_echo.backend.generate"


def event_id() -> str:
    return f"event_{uuid.uuid4().hex}"


def response_payload(response_id: str, status: str) -> dict:
    return {
        "id": response_id,
        "max_output_tokens": "inf",
        "object": "realtime.response",
        "output": [],
        "output_modalities": ["audio"],
        "status": status,
    }


async def generate(request_stream, context):
    async for client_event in request_stream:
        if context.is_stopped():
            return

        event_type = client_event.get("type") if isinstance(client_event, dict) else None

        if event_type == "session.update":
            yield {
                "type": "session.updated",
                "event_id": event_id(),
                "session": client_event.get("session"),
            }

        elif event_type == "input_audio_buffer.append":
            audio = client_event.get("audio", "")
            response_id = f"resp_{uuid.uuid4().hex}"
            item_id = f"item_{uuid.uuid4().hex}"

            yield {
                "type": "response.created",
                "event_id": event_id(),
                "response": response_payload(response_id, "in_progress"),
            }
            yield {
                "type": "response.output_audio.delta",
                "event_id": event_id(),
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": audio,
            }
            yield {
                "type": "response.output_audio.done",
                "event_id": event_id(),
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
            }
            yield {
                "type": "response.done",
                "event_id": event_id(),
                "response": response_payload(response_id, "completed"),
            }

        else:
            yield {
                "type": "error",
                "event_id": event_id(),
                "error": {
                    "type": "invalid_request_error",
                    "code": "unsupported_client_event",
                    "message": f"unsupported realtime event: {event_type}",
                },
            }


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    endpoint = runtime.endpoint(ENDPOINT_PATH)
    await register_model(
        ModelInput.Text,
        ModelType.Realtime,
        endpoint,
        MODEL_NAME,
        model_name=MODEL_NAME,
        worker_type=WorkerType.Aggregated,
    )
    await endpoint.serve_bidirectional_endpoint(generate)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
```

The `request_stream` argument is a `PyAsyncRequestStream`. It is an async iterator that yields inbound JSON-like Python objects, usually dictionaries. The handler itself is an async generator: it yields response frames as JSON-like Python objects.

The optional `context` argument is the same request context shape used by regular Python workers. Keep it in the signature when the engine needs cancellation checks, tracing context, or request metadata.

## Realtime Event Contract

The frontend owns the WebSocket transport and the first part of the realtime handshake:

- On connect, the frontend sends a `session.created` server event before any engine events.
- Before selecting a worker, the frontend waits for a `session.update` client event with `session.model` set to a registered `ModelType.Realtime` model name.
- The same `session.update` event is then forwarded to the selected engine as the first item in `request_stream`.
- Text WebSocket frames must contain JSON-encoded OpenAI Realtime client events. Audio is base64 text inside `input_audio_buffer.append`; binary WebSocket frames are rejected.

The worker is responsible for emitting valid OpenAI Realtime server events. A common minimum contract is:

| Client event | Typical server events |
| --- | --- |
| `session.update` | `session.updated` |
| `input_audio_buffer.append` | `response.created`, one or more `response.output_audio.delta`, `response.output_audio.done`, `response.done` |
| Unsupported event | `error` |

The frontend passes response frames through to the client after validating that they serialize as realtime server events.

## Cancellation And Stream End

Request-stream end is not cancellation. If `async for client_event in request_stream` exits, the client has stopped sending input, but the engine may still have response frames to emit. Continue yielding until the engine has completed its response or until `context.is_stopped()` becomes true.

`context.is_stopped()` becomes true when Dynamo needs the request to stop, for example after client disconnect or frontend shutdown. Check it inside long loops and before expensive work.

## Current Limitations

- `serve_bidirectional_endpoint()` does not take a `health_check_payload`. Realtime engines are usually stateful, and the health-check canary path is not wired for bidirectional streams yet.
- Local in-process engine registration is not enabled for bidirectional endpoints; use the normal discovery and request-plane path.
- The realtime frontend currently exposes JSON text frames for `/v1/realtime`; send audio as base64 in the event payload.

## Related Tests

The Python bridge has an echo worker and WebSocket tests that demonstrate the same event contract:

- `tests/frontend/realtime_echo_worker.py`
- `tests/frontend/test_realtime_python_bridge.py`
