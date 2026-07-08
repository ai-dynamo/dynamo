# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Realtime bidirectional echo worker for the Python bridge e2e test.

Registers a ``ModelType.Realtime`` + ``ModelInput.Text`` model and serves a
Python ``async def generate(request_stream, context)`` bidirectional engine
via ``serve_bidirectional_endpoint``. A launched ``dynamo.frontend`` discovers
this worker via the configured discovery backend (the e2e uses the file
backend, ``DYN_FILE_KV``) and installs a typed realtime PushRouter to it.
"""

from __future__ import annotations

import asyncio
import uuid

import uvloop

from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import DistributedRuntime
from tests.frontend.test_realtime_python_bridge import ENDPOINT_PATH, MODEL_NAME


def _event_id() -> str:
    return f"event_{uuid.uuid4().hex}"


def _response_payload(response_id: str, status: str) -> dict:
    """Minimal `RealtimeResponse` payload accepted by the frontend's typed reader.

    The typed reader requires `id`, `max_output_tokens`, `object`, `output`,
    `output_modalities`, and `status` to be present, so they are minted verbatim.
    """
    return {
        "id": response_id,
        "max_output_tokens": "inf",
        "object": "realtime.response",
        "output": [],
        "output_modalities": ["audio"],
        "status": status,
    }


async def _python_realtime_echo(request_stream, context):
    """
    Realtime echo semantics expressed as a Python bidirectional engine:

      - `session.update` -> `session.updated` echoing the session block.
      - `input_audio_buffer.append` -> `response.created` ->
        `response.output_audio.delta` -> `response.output_audio.done` ->
        `response.done`. The single delta carries the full audio payload.
      - Anything else -> `error` event with a stable code.
    """
    async for client_event in request_stream:
        if context.is_stopped():
            return
        etype = client_event.get("type") if isinstance(client_event, dict) else None

        if etype == "session.update":
            yield {
                "type": "session.updated",
                "event_id": _event_id(),
                "session": client_event.get("session"),
            }
        elif etype == "input_audio_buffer.append":
            audio = client_event.get("audio", "")
            response_id = f"resp_{uuid.uuid4().hex}"
            item_id = f"item_{uuid.uuid4().hex}"
            # Intentionally reuse this mutable dict across yields. The Rust
            # network engine must serialize each frame before polling us again,
            # otherwise queued PyObject handles would all observe later updates.
            response_event = {
                "type": "response.created",
                "event_id": _event_id(),
                "response": _response_payload(response_id, "in_progress"),
            }

            yield response_event
            response_event.clear()
            response_event.update(
                {
                    "type": "response.output_audio.delta",
                    "event_id": _event_id(),
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": audio,
                }
            )
            yield response_event
            response_event.clear()
            response_event.update(
                {
                    "type": "response.output_audio.done",
                    "event_id": _event_id(),
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                }
            )
            yield response_event
            response_event.clear()
            response_event.update(
                {
                    "type": "response.done",
                    "event_id": _event_id(),
                    "response": _response_payload(response_id, "completed"),
                }
            )
            yield response_event
        else:
            yield {
                "type": "error",
                "event_id": _event_id(),
                "error": {
                    "type": "invalid_request_error",
                    "code": "unsupported_client_event",
                    "message": f"python realtime echo does not support {etype}",
                },
            }


async def main() -> None:
    """Register the realtime echo model and serve the bidirectional engine.

    Uses file-based discovery (``DYN_FILE_KV``) + the tcp request plane, so the
    worker and the launched frontend coordinate without etcd or nats.
    """
    # event_plane="zmq" must be passed explicitly (not just via DYN_EVENT_PLANE):
    # the runtime's NATS gating only treats the event plane as "no nats" when the
    # event_plane argument is set, so an ambient NATS_SERVER (as in CI) would
    # otherwise force a NATS connection.
    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "file", "tcp", event_plane="zmq"
    )
    endpoint = runtime.endpoint(ENDPOINT_PATH)
    await register_model(
        ModelInput.Text,
        ModelType.Realtime,
        endpoint,
        MODEL_NAME,
        model_name=MODEL_NAME,
        worker_type=WorkerType.Aggregated,
    )
    await endpoint.serve_bidirectional_endpoint(_python_realtime_echo)


if __name__ == "__main__":
    uvloop.run(main())
