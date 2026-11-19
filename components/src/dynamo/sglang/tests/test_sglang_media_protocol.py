# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SGLang media models fill in a default for an absent field only, and
keep the constraints of the shared models they derive from."""

import pytest
from pydantic import ValidationError

from dynamo.common.protocols.image_protocol import ImageNvExt, NvCreateImageRequest
from dynamo.common.protocols.video_protocol import NvCreateVideoRequest, VideoNvExt
from dynamo.sglang.media_protocol import (
    SGLangCreateImageRequest,
    SGLangCreateVideoRequest,
    SGLangImageNvExt,
    SGLangVideoNvExt,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
]

REQUIRED = {"prompt": "a cat", "model": "m"}


def test_video_defaults_fill_absent_fields_only():
    req = SGLangCreateVideoRequest(**REQUIRED)
    assert (req.size, req.seconds, req.response_format) == ("832x480", 4, "url")
    nvext = req.nvext
    assert (nvext.fps, nvext.num_inference_steps, nvext.guidance_scale) == (24, 50, 5.0)

    # One step and a guidance scale of zero are explicit values, not "unset".
    req = SGLangCreateVideoRequest(
        **REQUIRED,
        seconds=1,
        nvext={"fps": 1, "num_inference_steps": 1, "guidance_scale": 0.0},
    )
    assert req.seconds == 1
    nvext = req.nvext
    assert (nvext.fps, nvext.num_inference_steps, nvext.guidance_scale) == (1, 1, 0.0)


def test_image_defaults_fill_absent_fields_only():
    req = SGLangCreateImageRequest(**REQUIRED)
    assert (req.n, req.response_format) == (1, "url")
    assert (req.nvext.num_inference_steps, req.nvext.guidance_scale) == (50, 7.5)

    req = SGLangCreateImageRequest(
        **REQUIRED, n=2, nvext={"num_inference_steps": 1, "guidance_scale": 0.0}
    )
    assert req.n == 2
    assert (req.nvext.num_inference_steps, req.nvext.guidance_scale) == (1, 0.0)


# Every field whose default an SGLang model overrides. `nvext` is overridden
# with a default factory and is checked through its own model.
OVERRIDES = [
    (SGLangImageNvExt, ImageNvExt, {}, ["guidance_scale", "num_inference_steps"]),
    (
        SGLangCreateImageRequest,
        NvCreateImageRequest,
        REQUIRED,
        ["n", "response_format"],
    ),
    (
        SGLangVideoNvExt,
        VideoNvExt,
        {},
        ["fps", "guidance_scale", "num_inference_steps"],
    ),
    (
        SGLangCreateVideoRequest,
        NvCreateVideoRequest,
        REQUIRED,
        ["response_format", "seconds", "size"],
    ),
]

# Values on and around the bounds, of the wrong type, and of each literal.
PROBES = [0, 1, 4, 10, 11, -1, 0.0, 4.5, "4", True, "url", "b64_json", "ftp", "1x1"]


def _accepts(model, body) -> bool:
    try:
        model.model_validate(body)
    except ValidationError:
        return False
    return True


@pytest.mark.parametrize(
    ("derived", "base", "required", "fields"),
    OVERRIDES,
    ids=[derived.__name__ for derived, *_ in OVERRIDES],
)
def test_an_override_accepts_and_rejects_what_the_shared_field_does(
    derived, base, required, fields
):
    """An override repeats the base constraint; this keeps the two equal. The
    one intended difference is `None`: the base takes it as absent, the
    override has a value in its place, so `None` is not probed."""
    for name in fields:
        for probe in PROBES:
            body = {**required, name: probe}
            assert _accepts(derived, body) == _accepts(
                base, body
            ), f"{derived.__name__}.{name} = {probe!r} differs from {base.__name__}"


@pytest.mark.parametrize(
    ("derived", "base", "fields"),
    [(derived, base, fields) for derived, base, _, fields in OVERRIDES],
    ids=[derived.__name__ for derived, *_ in OVERRIDES],
)
def test_the_overridden_fields_are_the_listed_ones(derived, base, fields):
    overridden = sorted(
        name
        for name, field in derived.model_fields.items()
        if name != "nvext" and field.default != base.model_fields[name].default
    )
    assert overridden == fields
    assert set(derived.model_fields) == set(base.model_fields), "no new fields"
