# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang's media request models: the shared protocols with SGLang's defaults.

The shared models in ``dynamo.common.protocols`` are generated from the Rust
types and leave the tunable fields optional: the frontend forwards a request
as the client sent it, and each worker owns its defaults. The classes here
derive from the shared models and override only the default of each field. A
default applies to an absent field and never to an explicit value, so a
request that asks for one denoising step or a guidance scale of zero gets
what it asked for.

An override repeats the constraint of the base field. The tests in
``tests/test_sglang_media_protocol.py`` check that the two accept and reject
the same values.
"""

from typing import Annotated, Literal

from pydantic import Field, StrictFloat, StrictStr

from dynamo.common.protocols.image_protocol import ImageNvExt, NvCreateImageRequest
from dynamo.common.protocols.video_protocol import NvCreateVideoRequest, VideoNvExt


class SGLangImageNvExt(ImageNvExt):
    num_inference_steps: Annotated[int, Field(ge=1, strict=True)] = 50
    guidance_scale: StrictFloat = 7.5


class SGLangCreateImageRequest(NvCreateImageRequest):
    """An absent ``n`` means one image, as the OpenAI API documents."""

    n: Annotated[int, Field(ge=1, le=10, strict=True)] = 1
    response_format: Literal["url", "b64_json"] = "url"
    nvext: SGLangImageNvExt = Field(default_factory=SGLangImageNvExt)


class SGLangVideoNvExt(VideoNvExt):
    fps: Annotated[int, Field(ge=1, strict=True)] = 24
    num_inference_steps: Annotated[int, Field(ge=1, strict=True)] = 50
    guidance_scale: StrictFloat = 5.0


class SGLangCreateVideoRequest(NvCreateVideoRequest):
    """Wan's default clip: 832x480 at 24 fps for 4 seconds."""

    seconds: Annotated[int, Field(ge=1, strict=True)] = 4
    size: StrictStr = "832x480"
    response_format: Literal["url", "b64_json"] = "url"
    nvext: SGLangVideoNvExt = Field(default_factory=SGLangVideoNvExt)
