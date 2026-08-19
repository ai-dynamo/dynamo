# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Realtime WebSocket end-to-end test against a real vLLM-Omni model.

``tests/frontend/test_realtime_omni_bridge.py`` covers the same wire protocol
with a mock engine, which keeps it GPU-free but means the engine's own output
shapes are never exercised -- so a change to those shapes upstream lands
unnoticed. This closes that gap: it launches the real
``dynamo.vllm.omni --realtime`` worker (the same command as
``examples/backends/vllm/launch/agg_omni_realtime.sh``) and drives one turn.

Skipped on CI, and it cannot currently be otherwise. The realtime path requires
the model class to implement vLLM's ``SupportsRealtime``, and in vLLM-Omni
exactly one model does: ``Qwen3OmniMoeForConditionalGeneration``. Every
published Qwen3-Omni checkpoint is 30B-A3B, so there is no small stand-in --
``Qwen2.5-Omni-3B`` is supported by vLLM-Omni generally but does not implement
``SupportsRealtime``. CI runs on shared 24 GB L4s; ``test_vllm_omni.py`` already
skips the *7B* omni as needing ~80 GB.

To run it on a suitable host, drop the ``pytest.mark.skip`` entry from
``pytestmark`` below (pytest has no flag that overrides an unconditional skip)::

    pytest tests/serve/test_realtime_omni_real_model.py -s

``TENSOR_PARALLEL_SIZE`` and the ``gpu_4`` marker are a starting point, not a
measured configuration. Profile with ``tests/utils/profile_pytest.py`` and add
the resulting ``profiled_vram_gib`` / ``requested_vllm_kv_cache_bytes`` markers
before wiring this into any automated tier (see
``.ai/test-model-size-guardrails.md``).
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

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
TENSOR_PARALLEL_SIZE = 4

SKIP_REASON = (
    f"{MODEL_ID} is the only vLLM-Omni model implementing SupportsRealtime, and "
    "its weights exceed CI capacity (24 GB L4). Run manually on a multi-GPU host."
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.integration,
    pytest.mark.vllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_4,
    pytest.mark.post_merge,
    pytest.mark.model(MODEL_ID),
    pytest.mark.skip(reason=SKIP_REASON),
]

# Model load dominates; the turn itself is seconds.
WORKER_READY_TIMEOUT_S = 1800
TURN_TIMEOUT_S = 120.0


class RealtimeOmniWorkerProcess(ManagedProcess):
    """Launch the real realtime Omni worker; ready once the frontend lists it.

    Mirrors ``agg_omni_realtime.sh``: ``--output-modalities audio`` drives the
    talker so the response carries synthesized speech rather than text alone.
    """

    def __init__(self, request, *, frontend_port: int) -> None:
        super().__init__(
            command=[
                "python3",
                "-m",
                "dynamo.vllm.omni",
                "--realtime",
                "--model",
                MODEL_ID,
                "--output-modalities",
                "audio",
                "--trust-remote-code",
                "--enforce-eager",
                "--tensor-parallel-size",
                str(TENSOR_PARALLEL_SIZE),
            ],
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", self._model_listed)
            ],
            timeout=WORKER_READY_TIMEOUT_S,
            display_output=True,
            terminate_all_matching_process_names=False,
            straggler_commands=["-m dynamo.vllm.omni"],
            log_dir=f"{request.node.name}_realtime_omni_real_worker",
        )

    @staticmethod
    def _model_listed(response: requests.Response) -> bool:
        try:
            if response.status_code != 200:
                return False
            data = response.json()
        except (ValueError, KeyError):
            return False
        return any(model.get("id") == MODEL_ID for model in data.get("data", []))


@pytest.fixture(scope="function")
def realtime_omni_real_frontend(
    request, file_storage_backend, dynamo_dynamic_ports: ServicePorts
):
    """Launch the frontend + real Omni realtime worker; yield the frontend port."""
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
        with RealtimeOmniWorkerProcess(request, frontend_port=frontend_port):
            logger.info("Realtime Omni worker registered %s", MODEL_ID)
            yield frontend_port


def _tone_pcm16() -> bytes:
    """One second of 16 kHz PCM16 to open and commit a turn.

    The content is not asserted on -- what matters is that a committed turn
    reaches the engine and comes back completed. A tone keeps the test hermetic.
    """
    t = np.linspace(0.0, 1.0, 16000, endpoint=False, dtype=np.float32)
    return (0.3 * np.sin(2 * np.pi * 440.0 * t) * 32767.0).astype(np.int16).tobytes()


async def _real_model_turn(port: int) -> None:
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(f"ws://127.0.0.1:{port}/v1/realtime") as ws:
            await open_session(
                ws,
                {
                    "type": "realtime",
                    "model": MODEL_ID,
                    "output_modalities": ["audio"],
                },
            )
            await commit_audio(ws, _tone_pcm16())
            # A real engine may emit event types the mock never produces; an
            # ``error`` frame still fails the turn.
            turn = await collect_turn(
                ws, timeout_s=TURN_TIMEOUT_S, allow_unknown_events=True
            )

            assert turn.status == "completed", turn.status
            assert turn.saw_audio_done, "engine should emit response.output_audio.done"
            assert turn.audio_b64_parts, "talker produced no audio deltas"

            # Real audio is only checked for shape: it must decode as PCM16 and
            # carry a non-silent signal.
            samples = turn.audio_pcm16
            assert samples.size > 0, "decoded audio was empty"
            assert np.abs(samples).max() > 0, "decoded audio was pure silence"

            logger.info(
                "realtime turn completed: %d audio samples, transcript=%r",
                samples.size,
                turn.transcript,
            )


@pytest.mark.timeout(WORKER_READY_TIMEOUT_S + 300)
def test_real_model_audio_round_trip(realtime_omni_real_frontend) -> None:
    """One committed turn against the real engine yields audio and completes."""
    asyncio.run(_real_model_turn(realtime_omni_real_frontend))
