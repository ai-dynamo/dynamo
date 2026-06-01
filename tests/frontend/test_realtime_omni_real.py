# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GPU end-to-end test for the realtime vLLM-Omni worker against a real model.

Launches the real ``python -m dynamo.vllm.omni --realtime`` worker (which builds
an actual ``AsyncOmni`` engine) plus a ``dynamo.frontend``, then streams a
synthetic 16 kHz mono PCM16 utterance through ``/v1/realtime`` and asserts the
spec-shaped audio response envelope comes back.

Gated: a realtime-capable Omni model (e.g. Qwen3-Omni) needs a GPU and a large
download, beyond pre-merge CI capacity, so this is opt-in. Set
``DYN_TEST_OMNI_REALTIME_MODEL`` to a local model path / HF id to enable it.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import struct
import tempfile

import aiohttp
import pytest
import requests

from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.port_utils import ServicePorts

logger = logging.getLogger(__name__)

OMNI_REALTIME_MODEL = os.environ.get("DYN_TEST_OMNI_REALTIME_MODEL")
SERVED_MODEL_NAME = "omni-realtime"
ENDPOINT_PATH = "test_omni_real_ws.realtime.generate"

pytestmark = [
    pytest.mark.gpu_1,
    pytest.mark.integration,
    pytest.mark.skipif(
        not OMNI_REALTIME_MODEL,
        reason="set DYN_TEST_OMNI_REALTIME_MODEL to a realtime-capable Omni model",
    ),
]


def _sine_pcm16(seconds: float = 1.0, hz: int = 220, sample_rate: int = 16000) -> bytes:
    """Synthesize a mono PCM16 sine tone at the model's 16 kHz input rate."""
    n = int(seconds * sample_rate)
    samples = (
        int(0.3 * 32767 * math.sin(2 * math.pi * hz * i / sample_rate))
        for i in range(n)
    )
    return b"".join(struct.pack("<h", s) for s in samples)


class RealtimeOmniWorkerProcess(ManagedProcess):
    """Launch the real --realtime Omni worker; ready once the frontend lists it."""

    def __init__(self, request, *, frontend_port: int) -> None:
        super().__init__(
            command=[
                "python3",
                "-m",
                "dynamo.vllm.omni",
                "--realtime",
                "--model",
                OMNI_REALTIME_MODEL,
                "--served-model-name",
                SERVED_MODEL_NAME,
                "--endpoint",
                ENDPOINT_PATH,
                "--discovery-backend",
                "file",
                "--request-plane",
                "tcp",
            ],
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", self._model_listed)
            ],
            timeout=600,
            display_output=True,
            terminate_all_matching_process_names=False,
            stragglers=["VLLM:EngineCore"],
            straggler_commands=["-m dynamo.vllm.omni"],
            log_dir=f"{request.node.name}_realtime_omni_real",
        )

    @staticmethod
    def _model_listed(response: requests.Response) -> bool:
        try:
            if response.status_code != 200:
                return False
            data = response.json()
        except (ValueError, KeyError):
            return False
        return any(
            model.get("id") == SERVED_MODEL_NAME for model in data.get("data", [])
        )


@pytest.fixture(scope="function")
def realtime_omni_real_frontend(request, dynamo_dynamic_ports: ServicePorts):
    frontend_port = dynamo_dynamic_ports.frontend_port
    with tempfile.TemporaryDirectory(prefix="dyn_realtime_omni_real_kv_") as file_kv:
        os.environ["DYN_FILE_KV"] = file_kv
        try:
            with DynamoFrontendProcess(
                request,
                frontend_port=frontend_port,
                extra_args=["--discovery-backend", "file", "--request-plane", "tcp"],
                terminate_all_matching_process_names=False,
            ):
                logger.info("Frontend started on port %s", frontend_port)
                with RealtimeOmniWorkerProcess(request, frontend_port=frontend_port):
                    logger.info("Realtime Omni worker registered %s", SERVED_MODEL_NAME)
                    yield frontend_port
        finally:
            os.environ.pop("DYN_FILE_KV", None)


async def _recv_json(ws: aiohttp.ClientWebSocketResponse, timeout_s: float) -> dict:
    msg = await asyncio.wait_for(ws.receive(), timeout=timeout_s)
    if msg.type is not aiohttp.WSMsgType.TEXT:
        raise AssertionError(f"unexpected websocket frame: {msg.type!r} {msg.data!r}")
    return json.loads(msg.data)


async def _drain_until(
    ws: aiohttp.ClientWebSocketResponse, expected_type: str, timeout_s: float
) -> dict:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        remaining = deadline - loop.time()
        event = await _recv_json(ws, max(remaining, 0.01))
        if event.get("type") == expected_type:
            return event
    raise AssertionError(f"timed out waiting for {expected_type!r}")


async def _real_audio_round_trip(port: int) -> None:
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
            f"ws://127.0.0.1:{port}/v1/realtime", max_msg_size=64 * 1024 * 1024
        ) as ws:
            await _drain_until(ws, "session.created", 10.0)

            await ws.send_str(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {"type": "realtime", "model": SERVED_MODEL_NAME},
                    }
                )
            )
            await _drain_until(ws, "session.updated", 10.0)

            # Stream the utterance in ~100 ms chunks, then commit to generate.
            pcm16 = _sine_pcm16()
            chunk = 16000 * 2 // 10
            for i in range(0, len(pcm16), chunk):
                await ws.send_str(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(pcm16[i : i + chunk]).decode(),
                        }
                    )
                )
            await ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))

            response_id: str | None = None
            audio_bytes = b""
            saw_audio_done = False
            done_status: str | None = None

            loop = asyncio.get_event_loop()
            deadline = loop.time() + 300.0
            while done_status is None:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise AssertionError("timed out before response.done")
                event = await _recv_json(ws, remaining)
                etype = event.get("type")
                if etype == "response.created":
                    response_id = event["response"]["id"]
                elif etype == "response.output_audio.delta":
                    audio_bytes += base64.b64decode(event["delta"])
                elif etype == "response.output_audio.done":
                    saw_audio_done = True
                elif etype == "response.done":
                    done_status = event["response"]["status"]
                elif etype == "error":
                    raise AssertionError(f"server error event: {event}")

            assert response_id is not None
            assert done_status == "completed", done_status
            assert saw_audio_done, "expected response.output_audio.done"
            assert len(audio_bytes) > 0, "expected non-empty synthesized audio"


@pytest.mark.timeout(900)
def test_realtime_omni_real_audio_round_trip(realtime_omni_real_frontend) -> None:
    """A real Omni model returns a spec audio envelope over /v1/realtime."""
    asyncio.run(_real_audio_round_trip(realtime_omni_real_frontend))
