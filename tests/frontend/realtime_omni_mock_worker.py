# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Realtime Omni worker driven by a mock vLLM-Omni engine, for the bridge e2e.

Serves the real ``RealtimeOmniHandler`` (the production translation layer) but
backs it with a fake AsyncOmni that echoes appended audio back as
``OmniRequestOutput``-shaped frames. This exercises the full bidirectional
bridge — frontend ``/v1/realtime`` -> PushRouter -> Python engine -> handler —
without a GPU or model download, so it runs in the same file-discovery e2e
shape as ``realtime_echo_worker.py``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import uvloop
from vllm_omni.engine.mm_outputs import MultimodalPayload

from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import DistributedRuntime
from dynamo.vllm.omni.realtime_handler import RealtimeOmniHandler
from tests.frontend.test_realtime_omni_bridge import (
    ENDPOINT_PATH,
    MOCK_TRANSCRIPT,
    MODEL_NAME,
)


async def _passthrough_factory(audio_stream, input_stream):
    """Stand-in for ``OpenAIServingRealtime.transcribe_realtime``.

    The real factory buffers audio into model prompts; the mock engine only
    needs the raw float32 waveforms, so we yield each audio chunk straight
    through. ``input_stream`` (the talker token-feedback queue) is unused here.
    """
    async for waveform in audio_stream:
        yield waveform


class _MockAsyncOmni:
    """Fake AsyncOmni: drains the streaming audio input, then echoes it back.

    Yields a stage-0 text frame (transcript) followed by the accumulated audio
    as a single multimodal output, matching the fields RealtimeOmniHandler reads
    off a real ``OmniRequestOutput`` (``stage_id``, ``outputs[].text``, and a
    ``MultimodalPayload`` whose ``tensors['audio']`` holds the waveform).
    """

    default_sampling_params_list: list = []

    async def generate(
        self, *, prompt, request_id, sampling_params_list=None, output_modalities=None
    ):
        chunks = [chunk async for chunk in prompt]
        full = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        yield SimpleNamespace(
            stage_id=0,
            outputs=[SimpleNamespace(text=MOCK_TRANSCRIPT, token_ids=[1])],
            prompt_token_ids=[0],
            multimodal_output=MultimodalPayload(),
        )
        yield SimpleNamespace(
            stage_id=1,
            outputs=[],
            multimodal_output=MultimodalPayload(
                tensors={"audio": full}, metadata={"sr": 16000}
            ),
        )


async def main() -> None:
    runtime = DistributedRuntime(asyncio.get_running_loop(), "file", "tcp")
    endpoint = runtime.endpoint(ENDPOINT_PATH)
    handler = RealtimeOmniHandler(
        engine_client=_MockAsyncOmni(),
        model_name=MODEL_NAME,
        streaming_input_factory=_passthrough_factory,
    )
    await register_model(
        ModelInput.Text,
        ModelType.Realtime,
        endpoint,
        MODEL_NAME,
        model_name=MODEL_NAME,
        worker_type=WorkerType.Aggregated,
    )
    await endpoint.serve_bidirectional_endpoint(handler.generate)


if __name__ == "__main__":
    uvloop.run(main())
