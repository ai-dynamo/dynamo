# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end realtime WebSocket test through a launched ``dynamo.frontend`` and
a separately launched Python bidirectional worker.

The frontend discovers the worker's ``ModelType.Realtime`` MDC and installs a
typed realtime PushRouter to it; a WebSocket client connects to
``/v1/realtime``, drives client events through the full bridge, and asserts the
spec-shaped server events come back. This exercises the real frontend's own
discovery wiring rather than constructing an in-process service.

Discovery uses the file backend (``DYN_FILE_KV``) and the tcp request plane, so
the two processes coordinate without standing up etcd or nats.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile

import aiohttp
import pytest
import requests

from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.port_utils import ServicePorts

logger = logging.getLogger(__name__)

# Shared with the worker module (tests/frontend/realtime_echo_worker.py imports
# these), mirroring the test-owns-constants pattern used by the vLLM pre/post
# integration test.
MODEL_NAME = "py-realtime-echo"
ENDPOINT_PATH = "test_py_ws_e2e.realtime.generate"

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.gpu_0,
]


class RealtimeEchoWorkerProcess(ManagedProcess):
    """Launch the realtime echo worker; ready once the frontend lists its model."""

    def __init__(self, request, *, frontend_port: int) -> None:
        super().__init__(
            command=["python3", "-m", "tests.frontend.realtime_echo_worker"],
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", self._model_listed)
            ],
            timeout=60,
            display_output=True,
            terminate_all_matching_process_names=False,
            straggler_commands=["-m tests.frontend.realtime_echo_worker"],
            log_dir=f"{request.node.name}_realtime_worker",
        )

    @staticmethod
    def _model_listed(response: requests.Response) -> bool:
        try:
            if response.status_code != 200:
                return False
            data = response.json()
        except (ValueError, KeyError):
            return False
        return any(model.get("id") == MODEL_NAME for model in data.get("data", []))


@pytest.fixture(scope="function")
def realtime_frontend(request, monkeypatch, dynamo_dynamic_ports: ServicePorts):
    """Launch the frontend + realtime worker; yield the frontend port once discovered.

    Coordinates over file-based discovery (``DYN_FILE_KV`` points both processes
    at a shared temp dir) and the tcp request plane, so no etcd/nats is needed.
    The worker's health check polls the frontend's ``/v1/models`` for
    ``MODEL_NAME``, so by the time the fixture yields, discovery has wired the
    realtime endpoint and a WebSocket session can be opened immediately.
    """
    frontend_port = dynamo_dynamic_ports.frontend_port
    with tempfile.TemporaryDirectory(prefix="dyn_realtime_kv_") as file_kv:
        # Shared file discovery store for both subprocesses (which copy
        # os.environ at spawn). monkeypatch restores it on teardown without
        # clobbering external state.
        monkeypatch.setenv("DYN_FILE_KV", file_kv)
        # Drop any ambient NATS_SERVER (CI sets one for the suite). The runtime
        # forces a NATS connection when NATS_SERVER is set and the event plane
        # is not set via an explicit argument — which is the case for the
        # launched frontend — so an ambient NATS_SERVER would make this
        # otherwise-local test require a NATS the fixture never starts.
        monkeypatch.delenv("NATS_SERVER", raising=False)
        # `--event-plane zmq` is passed explicitly (not via DYN_EVENT_PLANE):
        # the runtime only treats the event plane as "no nats" when the event
        # plane is set explicitly, so an ambient NATS_SERVER (as in CI) would
        # otherwise force a NATS connection. With file discovery + tcp request
        # plane + explicit zmq event plane, the test needs no etcd/nats.
        with DynamoFrontendProcess(
            request,
            frontend_port=frontend_port,
            extra_args=[
                "--discovery-backend",
                "file",
                "--request-plane",
                "tcp",
                "--event-plane",
                "zmq",
            ],
            terminate_all_matching_process_names=False,
        ):
            logger.info("Frontend started on port %s", frontend_port)
            with RealtimeEchoWorkerProcess(request, frontend_port=frontend_port):
                logger.info("Realtime echo worker registered model %s", MODEL_NAME)
                yield frontend_port


async def _recv_json(ws: aiohttp.ClientWebSocketResponse, timeout_s: float) -> dict:
    """Receive the next text frame and parse it as JSON."""
    msg = await asyncio.wait_for(ws.receive(), timeout=timeout_s)
    if msg.type is not aiohttp.WSMsgType.TEXT:
        raise AssertionError(f"unexpected websocket frame: {msg.type!r} {msg.data!r}")
    return json.loads(msg.data)


async def _drain_until(
    ws: aiohttp.ClientWebSocketResponse, expected_type: str, timeout_s: float = 5.0
) -> dict:
    """Read frames until one with ``type == expected_type`` arrives."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        remaining = deadline - loop.time()
        event = await _recv_json(ws, max(remaining, 0.01))
        if event.get("type") == expected_type:
            return event
    raise AssertionError(
        f"timed out waiting for a {expected_type!r} frame on the websocket"
    )


async def _session_update_round_trip(port: int) -> None:
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(f"ws://127.0.0.1:{port}/v1/realtime") as ws:
            created = await _recv_json(ws, 5.0)
            assert created.get("type") == "session.created", created

            await ws.send_str(
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


async def _audio_envelope_round_trip(port: int) -> None:
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(f"ws://127.0.0.1:{port}/v1/realtime") as ws:
            await _drain_until(ws, "session.created")

            await ws.send_str(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {"type": "realtime", "model": MODEL_NAME},
                    }
                )
            )
            await _drain_until(ws, "session.updated")

            audio = "QUJDREVGRw=="
            await ws.send_str(
                json.dumps({"type": "input_audio_buffer.append", "audio": audio})
            )

            response_id: str | None = None
            deltas: list[str] = []
            saw_audio_done = False
            response_done_status: str | None = None

            loop = asyncio.get_event_loop()
            deadline = loop.time() + 5.0
            while response_done_status is None:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise AssertionError(
                        "timed out before observing response.done; "
                        f"deltas so far: {deltas!r}, saw_audio_done={saw_audio_done}"
                    )
                event = await _recv_json(ws, remaining)
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


@pytest.mark.timeout(120)
def test_websocket_session_update_round_trip(realtime_frontend) -> None:
    """`session.update` round-trips through the launched frontend + Python worker."""
    asyncio.run(_session_update_round_trip(realtime_frontend))


@pytest.mark.timeout(120)
def test_websocket_audio_envelope_round_trip(realtime_frontend) -> None:
    """`input_audio_buffer.append` returns the full response envelope over the WebSocket."""
    asyncio.run(_audio_envelope_round_trip(realtime_frontend))
