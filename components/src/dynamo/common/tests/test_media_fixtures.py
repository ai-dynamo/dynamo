# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The JSON fixtures in lib/llm/src/protocols/openai/media_fixtures/, shared
with the Rust tests in lib/llm/src/protocols/openai/media_fixtures.rs, which
hold the Rust types to the same files.

- `<module>_request.json` and `<module>_response.json`: one full message each.
  Both sides parse it and dump it back unchanged, so the generated model reads
  what the frontend writes, and the frontend reads what a worker writes.
- `<module>_cases.json`: bodies that both sides must accept, and bodies that
  both sides must reject: a wrong type, a value outside a bound, a missing
  required field, an unknown value of a response enum. A case marked
  `frontend_only` is rejected by the frontend alone and accepted here: an
  unknown value of a request enum the worker does not read, which it must
  accept from a newer frontend, or a rule the schema cannot state. This is
  the test of the claim that a worker rejects what the frontend rejects, for
  everything the generated model mirrors.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

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


CASE_FILES = [
    ("audio_cases.json", NvCreateAudioSpeechRequest, NvAudioSpeechResponse),
    ("image_cases.json", NvCreateImageRequest, NvImagesResponse),
    ("video_cases.json", NvCreateVideoRequest, NvVideosResponse),
]


def _cases():
    for file, request_model, response_model in CASE_FILES:
        module = file.split("_")[0]
        for case in json.loads((FIXTURES / file).read_text())["cases"]:
            model = {"request": request_model, "response": response_model}[case["type"]]
            frontend_only = case.get("frontend_only", False)
            assert not (frontend_only and case["valid"]), f"{file}: {case['name']}"
            yield pytest.param(
                model,
                case["body"],
                case["valid"] or frontend_only,
                id=f"{module}-{case['type']}-{case['name']}",
            )


@pytest.mark.parametrize(("model", "body", "accepted_by_a_worker"), list(_cases()))
def test_media_case_is_accepted_or_rejected_as_the_case_says(
    model, body, accepted_by_a_worker
):
    # A worker parses the request as a dict, as the handlers do.
    try:
        model.model_validate(body)
        accepted = True
    except ValidationError:
        accepted = False

    assert accepted == accepted_by_a_worker
