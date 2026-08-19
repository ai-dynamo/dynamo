# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for the ASR worker: audio in, RIVA streaming recognition, transcript out.

Mocks ``riva.client.ASRService`` so it runs without a RIVA server.
"""

import base64
from unittest.mock import MagicMock

import pytest
from riva.client import AudioEncoding
from riva_nim import asr_worker, config

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture
def backend():
    be = asr_worker.AsrBackend(
        config.RivaConnectionConfig(),
        sample_rate_hz=16000,
        language_code="en-US",
        model="parakeet-1.1b-en-US-asr-streaming",
        timeout_s=5.0,
    )
    be.asr = MagicMock()
    return be


def _final_response(transcript):
    alternative = MagicMock()
    alternative.transcript = transcript
    result = MagicMock()
    result.is_final = True
    result.alternatives = [alternative]
    response = MagicMock()
    response.results = [result]
    return response


async def test_generate_streams_audio_and_joins_final_transcript(backend):
    backend.asr.streaming_response_generator.return_value = [
        _final_response("hello world")
    ]

    audio = b"\x00\x01" * 6000  # ~larger than one chunk, forces multiple frames
    resp = await backend.generate(
        asr_worker.AsrRequest(audio_base64=base64.b64encode(audio).decode())
    )

    backend.asr.streaming_response_generator.assert_called_once()
    kwargs = backend.asr.streaming_response_generator.call_args.kwargs
    # The full buffer is streamed as chunks with the worker's recognition config.
    assert b"".join(kwargs["audio_chunks"]) == audio
    rec_config = kwargs["streaming_config"].config
    assert rec_config.sample_rate_hertz == 16000
    assert rec_config.language_code == "en-US"
    assert rec_config.encoding == AudioEncoding.LINEAR_PCM
    assert rec_config.model == "parakeet-1.1b-en-US-asr-streaming"

    assert resp.transcript == "hello world"


async def test_generate_returns_empty_for_empty_audio(backend):
    resp = await backend.generate(asr_worker.AsrRequest(audio_base64=""))
    assert resp.transcript == ""
    backend.asr.streaming_response_generator.assert_not_called()
