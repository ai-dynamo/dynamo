# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end realtime WebSocket test for the vLLM-Omni realtime bridge.

A launched ``dynamo.frontend`` discovers a mock-Omni realtime worker (the real
``RealtimeOmniHandler`` backed by a fake AsyncOmni that echoes audio) and
installs a typed realtime PushRouter to it. A WebSocket client connects to
``/v1/realtime``, drives OpenAI Realtime client events, and asserts the
spec-shaped server events come back — exercising the full bridge without a GPU
or model download.

Discovery uses the file backend (``DYN_FILE_KV``) and the tcp request plane, so
the two processes coordinate without etcd or nats. Mirrors
``test_realtime_python_bridge.py``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging

import aiohttp
import numpy as np
import pytest
import requests

from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.port_utils import ServicePorts

logger = logging.getLogger(__name__)

# Shared with the worker module (realtime_omni_mock_worker.py imports these).
MODEL_NAME = "omni-realtime-mock"
ENDPOINT_PATH = "test_omni_ws_e2e.realtime.generate"
MOCK_TRANSCRIPT = "mock omni transcript"

# The vllm-runtime image bumped to vLLM 0.24.0 (PR #11076) but still pins
# vLLM-Omni 0.23.0rc1, whose vllm_omni/platforms/__init__.py imports
# `supports_xccl` from vllm.utils.torch_utils — a symbol 0.24.0 removed. So
# `import vllm_omni` fails image-wide and this test's mock worker (which
# hard-imports vLLM-Omni) crashes on startup. Skip the whole module up front —
# no point importing vLLM-Omni when we know it can't load.
# TODO: remove this skip once vLLM-Omni is bumped to a vLLM-0.24-compatible release.
pytest.skip(
    "vLLM-Omni 0.23.0rc1 is incompatible with the image's vLLM 0.24.0 "
    "(missing supports_xccl); re-enable when the vLLM-Omni pin is realigned.",
    allow_module_level=True,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.vllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
]


class RealtimeOmniMockWorkerProcess(ManagedProcess):
    """Launch the mock-Omni realtime worker; ready once the frontend lists it."""

    def __init__(self, request, *, frontend_port: int) -> None:
        super().__init__(
            command=["python3", "-m", "tests.frontend.realtime_omni_mock_worker"],
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", self._model_listed)
            ],
            timeout=60,
            display_output=True,
            terminate_all_matching_process_names=False,
            straggler_commands=["-m tests.frontend.realtime_omni_mock_worker"],
            log_dir=f"{request.node.name}_realtime_omni_worker",
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
def realtime_omni_frontend(
    request, file_storage_backend, dynamo_dynamic_ports: ServicePorts
):
    """Launch the frontend + mock-Omni worker; yield the frontend port once discovered.

    Uses file-based discovery (the ``file_storage_backend`` fixture sets
    ``DYN_FILE_KV``), the tcp request plane, and an explicit zmq event plane, so
    the two processes coordinate without etcd or nats and an ambient
    ``NATS_SERVER`` never forces a connection. Mirrors ``test_prompt_embeds.py``.
    """
    _ = file_storage_backend  # sets DYN_FILE_KV for both subprocesses
    frontend_port = dynamo_dynamic_ports.frontend_port
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
        with RealtimeOmniMockWorkerProcess(request, frontend_port=frontend_port):
            logger.info("Mock-Omni realtime worker registered %s", MODEL_NAME)
            yield frontend_port


async def _recv_json(ws: aiohttp.ClientWebSocketResponse, timeout_s: float) -> dict:
    msg = await asyncio.wait_for(ws.receive(), timeout=timeout_s)
    if msg.type is not aiohttp.WSMsgType.TEXT:
        raise AssertionError(f"unexpected websocket frame: {msg.type!r} {msg.data!r}")
    return json.loads(msg.data)


async def _drain_until(
    ws: aiohttp.ClientWebSocketResponse, expected_type: str, timeout_s: float = 5.0
) -> dict:
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


async def _audio_round_trip(port: int) -> None:
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

            # Send a short PCM16 ramp; the mock engine echoes it back as audio.
            pcm16 = np.linspace(-8000, 8000, 128, dtype=np.int16).tobytes()
            await ws.send_str(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm16).decode("utf-8"),
                    }
                )
            )
            await ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))

            response_id: str | None = None
            audio_b64_parts: list[str] = []
            transcript_parts: list[str] = []
            saw_audio_done = False
            response_done_status: str | None = None

            loop = asyncio.get_event_loop()
            deadline = loop.time() + 10.0
            while response_done_status is None:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise AssertionError(
                        "timed out before response.done; "
                        f"audio_parts={len(audio_b64_parts)}, "
                        f"saw_audio_done={saw_audio_done}"
                    )
                event = await _recv_json(ws, remaining)
                etype = event.get("type")
                if etype == "response.created":
                    response_id = event["response"]["id"]
                elif etype == "response.output_audio_transcript.delta":
                    transcript_parts.append(event["delta"])
                    assert event["response_id"] == response_id, event
                elif etype == "response.output_audio.delta":
                    audio_b64_parts.append(event["delta"])
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
            assert "".join(transcript_parts) == MOCK_TRANSCRIPT, transcript_parts

            # Concatenated audio deltas decode back to the input ramp (echo).
            out_bytes = b"".join(base64.b64decode(p) for p in audio_b64_parts)
            in_f32 = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
            out_f32 = (
                np.frombuffer(out_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            )
            assert out_f32.shape == in_f32.shape, (out_f32.shape, in_f32.shape)
            assert np.allclose(out_f32, in_f32, atol=2e-4)


@pytest.mark.timeout(120)
def test_websocket_audio_round_trip(realtime_omni_frontend) -> None:
    """Appended audio echoes back as the full spec response envelope."""
    asyncio.run(_audio_round_trip(realtime_omni_frontend))
