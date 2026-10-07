# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The JSON fixtures in lib/llm/src/protocols/openai/media_fixtures/ parse with
the generated models and dump back unchanged.

The Rust tests in lib/llm/src/protocols/openai/media_fixtures.rs hold the same
files to the Rust types. One message that both sides accept and reproduce
shows that the generated model reads what the frontend writes, and that the
frontend reads what a worker writes.
"""

import json
from pathlib import Path

import pytest

from dynamo.common.protocols.audio_protocol import (
    NvAudioSpeechResponse,
    NvCreateAudioSpeechRequest,
)
from dynamo.common.protocols.image_protocol import (
    NvCreateImageRequest,
    NvImagesResponse,
)
from dynamo.common.protocols.video_protocol import (
    NvCreateVideoRequest,
    NvVideosResponse,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

# components/src/dynamo/common/tests -> the repository root.
FIXTURES = (
    Path(__file__).resolve().parents[5] / "lib/llm/src/protocols/openai/media_fixtures"
)

CASES = [
    ("audio_request.json", NvCreateAudioSpeechRequest),
    ("audio_response.json", NvAudioSpeechResponse),
    ("image_request.json", NvCreateImageRequest),
    ("image_response.json", NvImagesResponse),
    ("video_request.json", NvCreateVideoRequest),
    ("video_response.json", NvVideosResponse),
]


@pytest.mark.parametrize(("file", "model"), CASES, ids=[file for file, _ in CASES])
def test_media_fixture_round_trips(file, model):
    text = (FIXTURES / file).read_text()

    parsed = model.model_validate_json(text)

    # A fixture omits an absent field, so an absent field must dump as absent.
    assert parsed.model_dump(mode="json", exclude_none=True) == json.loads(text)
