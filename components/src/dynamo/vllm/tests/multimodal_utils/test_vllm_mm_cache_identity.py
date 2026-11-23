# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache identity that the vLLM worker gives the engine for client image uuids.

These tests use only ``VllmMultimodalRequestProcessor`` and
``build_tokens_prompt`` so they run unchanged on release branches.
"""

import base64
from io import BytesIO

import pytest
from PIL import Image

from dynamo.vllm.multimodal_utils.request_processor import (
    VllmMultimodalRequestProcessor,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

SHARED_UUID = "shared-image-id"


def _image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (8, 8), color=color)


def _data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _request(caller: str, image: Image.Image) -> dict:
    return {
        "token_ids": [1, 2, 3],
        "image_cache_scope": caller,
        "multi_modal_data": {"image_url": [{"Url": _data_url(image)}]},
        "multi_modal_uuids": {"image_url": [SHARED_UUID]},
    }


class _FakeEngineCache:
    """Engine media cache keyed by the identity in ``multi_modal_uuids``.

    The first image stored under an identity is the one every later prompt
    with that identity gets back. With no identity, the key is the pixel
    bytes, as when the engine hashes the content itself.
    """

    def __init__(self) -> None:
        self.entries: dict[object, Image.Image] = {}

    @staticmethod
    def identity(prompt) -> object:
        uuids = (prompt.get("multi_modal_uuids") or {}).get("image") or [None]
        if uuids[0] is not None:
            return uuids[0]
        return ("content", prompt["multi_modal_data"]["image"].tobytes())

    def resolve(self, prompt) -> tuple[object, Image.Image]:
        identity = self.identity(prompt)
        image = prompt["multi_modal_data"]["image"]
        return identity, self.entries.setdefault(identity, image)


def _processor() -> VllmMultimodalRequestProcessor:
    return VllmMultimodalRequestProcessor(
        model="Qwen/Qwen3-VL-2B-Instruct",
        enable_multimodal=True,
        use_unified_vision_chunk=False,
    )


def _send(processor, cache, caller: str, image: Image.Image):
    prompt = processor.build_tokens_prompt(
        _request(caller, image), {"image": image}, None
    )
    return cache.resolve(prompt)


def test_same_uuid_with_different_images_gets_distinct_cache_identity():
    processor = _processor()
    cache = _FakeEngineCache()
    red = _image((255, 0, 0))
    blue = _image((0, 0, 255))

    identity_a, served_a = _send(processor, cache, "caller-a", red)
    identity_b, served_b = _send(processor, cache, "caller-b", blue)

    assert identity_a != identity_b
    assert served_a.tobytes() == red.tobytes()
    assert served_b.tobytes() == blue.tobytes()


def test_same_uuid_and_same_image_reuses_cache_identity():
    processor = _processor()
    cache = _FakeEngineCache()
    red = _image((255, 0, 0))

    identity_first, _ = _send(processor, cache, "caller-a", red)
    identity_repeat, served = _send(processor, cache, "caller-a", _image((255, 0, 0)))

    assert identity_first == identity_repeat
    assert len(cache.entries) == 1
    assert served.tobytes() == red.tobytes()
