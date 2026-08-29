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
import logging

import aiohttp
import numpy as np
import pytest
import requests

from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.port_utils import ServicePorts
from tests.utils.realtime_ws import collect_turn, commit_audio, open_session

logger = logging.getLogger(__name__)

# Shared with the worker module (realtime_omni_mock_worker.py imports these).
MODEL_NAME = "omni-realtime-mock"
ENDPOINT_PATH = "test_omni_ws_e2e.realtime.generate"
MOCK_TRANSCRIPT = "mock omni transcript"

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


async def _audio_round_trip(port: int) -> None:
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(f"ws://127.0.0.1:{port}/v1/realtime") as ws:
            await open_session(ws, {"type": "realtime", "model": MODEL_NAME})

            # A short PCM16 ramp; the mock engine echoes it back as audio.
            pcm16 = np.linspace(-8000, 8000, 128, dtype=np.int16).tobytes()
            await commit_audio(ws, pcm16)
            turn = await collect_turn(ws, timeout_s=10.0)

            assert turn.saw_audio_done, "engine should emit response.output_audio.done"
            assert turn.status == "completed", turn.status
            assert turn.transcript == MOCK_TRANSCRIPT, turn.transcript

            # Concatenated audio deltas decode back to the input ramp (echo).
            in_f32 = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
            out_f32 = turn.audio_pcm16.astype(np.float32) / 32767.0
            assert out_f32.shape == in_f32.shape, (out_f32.shape, in_f32.shape)
            assert np.allclose(out_f32, in_f32, atol=2e-4)


@pytest.mark.timeout(120)
def test_websocket_audio_round_trip(realtime_omni_frontend) -> None:
    """Appended audio echoes back as the full spec response envelope."""
    asyncio.run(_audio_round_trip(realtime_omni_frontend))
