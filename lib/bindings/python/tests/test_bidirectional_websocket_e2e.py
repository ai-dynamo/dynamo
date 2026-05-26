# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end WebSocket test through the realtime HTTP frontend that
exercises the actual Python bidirectional bridge.

A Python `async def generate(request_stream, context)` is registered via
`Endpoint.serve_bidirectional_endpoint`. A second `DistributedRuntime` in
the same process hosts the realtime HTTP service with a discovery
watcher, so the Python worker's MDC (`ModelType::Realtime` +
`ModelInput::Text`) is picked up and a typed
`PushRouter<RealtimeClientEvent, Annotated<RealtimeServerEvent>>` is
installed as the realtime engine. A WebSocket client connects to
`/v1/realtime`, sends client events, and asserts the spec-shaped server
events come back through the full bridge.

The frontend's typed reader requires fully-populated `RealtimeServerEvent`
payloads — in particular `RealtimeResponse` needs `id`, `max_output_tokens`,
`object`, `output`, `output_modalities`, and `status` — so the helper
below mints those fields verbatim.
"""

import asyncio
import json
import socket
import uuid

import pytest
import websockets

from dynamo._core import HttpService
from dynamo.llm import ModelInput, ModelType, register_model
from dynamo.runtime import DistributedRuntime

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
]

MODEL_NAME = "py-realtime-echo"
ENDPOINT_PATH = "test_py_ws_e2e.realtime.generate"


def _get_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _response_payload(response_id: str, status: str) -> dict:
    """Minimal `RealtimeResponse` payload accepted by the typed reader."""
    return {
        "id": response_id,
        "max_output_tokens": "inf",
        "object": "realtime.response",
        "output": [],
        "output_modalities": ["audio"],
        "status": status,
    }


def _event_id() -> str:
    return f"event_{uuid.uuid4().hex}"


async def _python_realtime_echo(request_stream, context):
    """
    Mirror of the Rust `EchoBidirectionalEngine` realtime semantics
    expressed as a Python `async def` consumed by
    `serve_bidirectional_endpoint`.

      - `session.update` -> `session.updated` echoing the session block.
      - `input_audio_buffer.append` -> `response.created` ->
        `response.output_audio.delta` -> `response.output_audio.done` ->
        `response.done`. Single delta carries the full audio payload.
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

            yield {
                "type": "response.created",
                "event_id": _event_id(),
                "response": _response_payload(response_id, "in_progress"),
            }
            yield {
                "type": "response.output_audio.delta",
                "event_id": _event_id(),
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": audio,
            }
            yield {
                "type": "response.output_audio.done",
                "event_id": _event_id(),
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
            }
            yield {
                "type": "response.done",
                "event_id": _event_id(),
                "response": _response_payload(response_id, "completed"),
            }
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


async def _wait_for_http_health(port: int, timeout_s: float = 10.0) -> None:
    """Poll the HTTP health endpoint until the frontend is up."""
    import aiohttp

    deadline = asyncio.get_event_loop().time() + timeout_s
    last_err: Exception | None = None
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=1.0)
    ) as session:
        while asyncio.get_event_loop().time() < deadline:
            try:
                async with session.get(f"http://127.0.0.1:{port}/health") as resp:
                    if resp.status == 200:
                        return
            except Exception as e:  # noqa: BLE001
                last_err = e
            await asyncio.sleep(0.05)
    raise RuntimeError(
        f"HTTP frontend never became healthy on port {port}; last error: {last_err!r}"
    )


async def _wait_for_realtime_model(port: int, timeout_s: float = 10.0) -> None:
    """
    Poll the frontend's model list until the realtime watcher has wired
    the Python worker's MDC into the typed PushRouter. Without this the
    very first WebSocket frame can race against discovery and the
    `get_realtime_engine` lookup will reject the session.
    """
    import aiohttp

    deadline = asyncio.get_event_loop().time() + timeout_s
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=1.0)
    ) as session:
        while asyncio.get_event_loop().time() < deadline:
            try:
                async with session.get(f"http://127.0.0.1:{port}/v1/models") as resp:
                    if resp.status == 200:
                        payload = await resp.json()
                        models = [m.get("id") for m in payload.get("data", [])]
                        if MODEL_NAME in models:
                            return
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(0.05)
    raise RuntimeError(
        f"Realtime watcher never picked up '{MODEL_NAME}' on port {port}"
    )


async def _drain_until(ws, expected_type: str, timeout_s: float = 3.0) -> dict:
    """Read frames until one with `type == expected_type` arrives."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        remaining = deadline - asyncio.get_event_loop().time()
        msg = await asyncio.wait_for(ws.recv(), timeout=max(remaining, 0.01))
        event = json.loads(msg)
        if event.get("type") == expected_type:
            return event
    raise AssertionError(
        f"timed out waiting for a {expected_type!r} frame on the websocket"
    )


async def _run_realtime_worker(runtime: DistributedRuntime, ready_event: asyncio.Event):
    """
    Register a Realtime+Text MDC and serve the bidirectional engine.
    Blocks on `serve_bidirectional_endpoint` until the runtime cancels.
    """
    endpoint = runtime.endpoint(ENDPOINT_PATH)
    await register_model(
        ModelInput.Text,
        ModelType.Realtime,
        endpoint,
        MODEL_NAME,
        model_name=MODEL_NAME,
    )
    ready_event.set()
    await endpoint.serve_bidirectional_endpoint(_python_realtime_echo)


@pytest.mark.forked
@pytest.mark.asyncio
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
async def test_websocket_session_update_round_trip_through_python_worker(
    temp_file_store, runtime
):
    """
    Drive a `session.update` over a real WebSocket, through the realtime
    frontend, into the Python bidirectional engine, and back. Asserts
    the round-tripped `session.updated` reaches the client unchanged.
    """
    port = _get_free_port()
    service = HttpService(port=port)
    service.enable_endpoint("realtime", True)
    await service.start_discovery_watcher(runtime)

    worker_ready = asyncio.Event()
    worker_task = asyncio.create_task(_run_realtime_worker(runtime, worker_ready))
    service_task = asyncio.ensure_future(service.run(runtime))

    try:
        await asyncio.wait_for(worker_ready.wait(), timeout=10.0)
        await _wait_for_http_health(port)
        await _wait_for_realtime_model(port)

        async with websockets.connect(f"ws://127.0.0.1:{port}/v1/realtime") as ws:
            created = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
            assert created.get("type") == "session.created", created

            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {"type": "realtime", "model": MODEL_NAME},
                    }
                )
            )

            updated = await _drain_until(ws, "session.updated")
            assert updated["session"]["type"] == "realtime", updated
            assert updated["session"]["model"] == MODEL_NAME, updated
    finally:
        service.shutdown()
        worker_task.cancel()
        for task in (worker_task, service_task):
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass


@pytest.mark.forked
@pytest.mark.asyncio
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
async def test_websocket_audio_envelope_round_trip_through_python_worker(
    temp_file_store, runtime
):
    """
    Drive `input_audio_buffer.append` end-to-end and assert the full
    multi-frame response envelope (`response.created` ->
    `response.output_audio.delta` -> `response.output_audio.done` ->
    `response.done`) arrives over the WebSocket with consistent
    `response_id` across the turn.
    """
    port = _get_free_port()
    service = HttpService(port=port)
    service.enable_endpoint("realtime", True)
    await service.start_discovery_watcher(runtime)

    worker_ready = asyncio.Event()
    worker_task = asyncio.create_task(_run_realtime_worker(runtime, worker_ready))
    service_task = asyncio.ensure_future(service.run(runtime))

    try:
        await asyncio.wait_for(worker_ready.wait(), timeout=10.0)
        await _wait_for_http_health(port)
        await _wait_for_realtime_model(port)

        async with websockets.connect(f"ws://127.0.0.1:{port}/v1/realtime") as ws:
            await _drain_until(ws, "session.created")

            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {"type": "realtime", "model": MODEL_NAME},
                    }
                )
            )
            await _drain_until(ws, "session.updated")

            audio = "QUJDREVGRw=="
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": audio,
                    }
                )
            )

            response_id: str | None = None
            deltas: list[str] = []
            saw_audio_done = False
            response_done_status: str | None = None

            deadline = asyncio.get_event_loop().time() + 5.0
            while response_done_status is None:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise AssertionError(
                        "timed out before observing response.done; "
                        f"deltas so far: {deltas!r}, saw_audio_done={saw_audio_done}"
                    )
                msg = await asyncio.wait_for(ws.recv(), timeout=remaining)
                event = json.loads(msg)
                etype = event.get("type")
                if etype == "response.created":
                    response_id = event["response"]["id"]
                elif etype == "response.output_audio.delta":
                    deltas.append(event["delta"])
                    assert event["response_id"] == response_id, event
                elif etype == "response.output_audio.done":
                    saw_audio_done = True
                    assert event["response_id"] == response_id, event
                elif etype == "response.done":
                    response_done_status = event["response"]["status"]
                    assert event["response"]["id"] == response_id, event
                else:
                    raise AssertionError(f"unexpected event type {etype!r}: {event}")

            assert response_id is not None
            assert saw_audio_done, "engine should emit response.output_audio.done"
            assert response_done_status == "completed", response_done_status
            assert "".join(deltas) == audio, deltas
    finally:
        service.shutdown()
        worker_task.cancel()
        for task in (worker_task, service_task):
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
