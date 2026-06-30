# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for the ASR worker: audio in, RIVA offline_recognize, transcript out.

Mocks ``riva.client.ASRService`` so it runs without a RIVA server.
"""

import base64
from unittest.mock import MagicMock

import pytest
from riva.client import AudioEncoding
from riva_nim import asr_worker, config


@pytest.fixture
def backend():
    be = asr_worker.AsrBackend(
        config.RivaConnectionConfig(),
        sample_rate_hz=16000,
        language_code="en-US",
        model="",
    )
    be.asr = MagicMock()
    return be


async def test_generate_recognizes_audio_to_transcript(backend):
    alternative = MagicMock()
    alternative.transcript = "hello world"
    result = MagicMock()
    result.alternatives = [alternative]
    recognize_response = MagicMock()
    recognize_response.results = [result]
    backend.asr.offline_recognize.return_value = recognize_response

    audio = b"\x00\x01fake-pcm"
    resp = await backend.generate(
        asr_worker.AsrRequest(audio_base64=base64.b64encode(audio).decode())
    )

    # The decoded PCM is forwarded to RIVA with a config carrying the worker's
    # recognition settings.
    backend.asr.offline_recognize.assert_called_once()
    args, _ = backend.asr.offline_recognize.call_args
    assert args[0] == audio
    recognition_config = args[1]
    assert recognition_config.sample_rate_hertz == 16000
    assert recognition_config.language_code == "en-US"
    assert recognition_config.encoding == AudioEncoding.LINEAR_PCM

    assert resp.transcript == "hello world"
