# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for the TTS worker: text in, RIVA synthesize, base64 audio out.

Mocks ``riva.client.SpeechSynthesisService`` so it runs without a RIVA server.
"""

import base64
from unittest.mock import MagicMock

import pytest
from riva_nim import config, tts_worker


@pytest.fixture
def backend():
    be = tts_worker.TtsBackend(
        config.RivaConnectionConfig(),
        voice="English-US.Female-1",
        language_code="en-US",
        sample_rate_hz=22050,
    )
    # Replace the real RIVA service with a mock; the handler must not touch a
    # live server in unit tests.
    be.tts = MagicMock()
    return be


async def test_generate_synthesizes_and_base64_encodes_audio(backend):
    synth_response = MagicMock()
    synth_response.audio = b"RIFF....fake-pcm-bytes"
    backend.tts.synthesize.return_value = synth_response

    resp = await backend.generate(tts_worker.TtsRequest(text="hello world"))

    # Text is forwarded to RIVA with the worker's configured voice settings.
    backend.tts.synthesize.assert_called_once()
    args, kwargs = backend.tts.synthesize.call_args
    assert args[0] == "hello world"
    assert kwargs["voice_name"] == "English-US.Female-1"
    assert kwargs["language_code"] == "en-US"
    assert kwargs["sample_rate_hz"] == 22050

    # The synthesized PCM is returned base64-encoded, with the sample rate echoed.
    assert base64.b64decode(resp.audio_base64) == b"RIFF....fake-pcm-bytes"
    assert resp.sample_rate_hz == 22050
