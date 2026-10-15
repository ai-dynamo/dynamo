# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from dynamo.common.http import HttpStatusError
from dynamo.common.multimodal import ImageLoader
from dynamo.vllm.omni import image_edit

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(30),
]


@pytest.fixture(autouse=True)
def metadata_lookup(monkeypatch):
    monkeypatch.setattr(
        image_edit,
        "get_diffusion_model_metadata",
        Mock(
            return_value=SimpleNamespace(
                supports_multimodal_inputs=True, max_multimodal_image_inputs=4
            )
        ),
    )


def image_reference(size=(2, 2)):
    with io.BytesIO() as buffer:
        Image.new("RGB", size, "red").save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@pytest.mark.asyncio
async def test_aggregate_pixel_budget_applies_before_rgb_expansion(monkeypatch):
    reference = image_reference()
    loader = ImageLoader(max_pixels=4)
    original = Image.Image.convert
    decoded = []

    def convert(image, *args, **kwargs):
        decoded.append(image.size)
        return original(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "convert", convert)
    with pytest.raises(HttpStatusError) as caught:
        await image_edit.load_image_edit_references(
            [reference, reference],
            image_loader=loader,
            model_class_name="Flux2KleinPipeline",
            max_total_pixels=7,
        )
    assert caught.value.status == 413
    assert "input_reference[1]" in caught.value.message
    assert decoded == [(2, 2)]
    loaded = await image_edit.load_image_edit_references(
        [reference, reference],
        image_loader=loader,
        model_class_name="Flux2KleinPipeline",
        max_total_pixels=8,
    )
    assert len(loaded) == 2


@pytest.mark.asyncio
async def test_inline_byte_limit_is_typed_and_redacted():
    reference = image_reference()
    with pytest.raises(HttpStatusError) as caught:
        await image_edit.load_image_edit_references(
            [reference],
            image_loader=ImageLoader(max_bytes=1),
            model_class_name="Flux2KleinPipeline",
        )
    assert caught.value.status == 413
    assert "byte limit" in caught.value.message
    assert reference not in str(caught.value)


@pytest.mark.asyncio
async def test_request_budget_checks_shared_cache_before_decode(monkeypatch):
    reference = image_reference((3, 3))
    loader = ImageLoader(max_pixels=4)
    loader._shared_image_cache = SimpleNamespace(
        get=AsyncMock(return_value=base64.b64decode(reference.split(",")[1]))
    )
    fetch = AsyncMock()
    monkeypatch.setattr("dynamo.common.multimodal.image_loader.fetch_bytes", fetch)
    with pytest.raises(HttpStatusError) as caught:
        await image_edit.load_image_edit_references(
            ["https://8.8.8.8/cached.png"],
            image_loader=loader,
            model_class_name="Flux2KleinPipeline",
        )
    assert caught.value.status == 413
    fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_edit_cancels_fetch_and_does_not_start_next(monkeypatch):
    started, stopped = asyncio.Event(), asyncio.Event()

    async def fetch(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    fetch_mock = AsyncMock(side_effect=fetch)
    monkeypatch.setattr("dynamo.common.multimodal.image_loader.fetch_bytes", fetch_mock)
    loader = ImageLoader()
    task = asyncio.create_task(
        image_edit.load_image_edit_references(
            ["https://8.8.8.8/first", "https://8.8.8.8/second"],
            image_loader=loader,
            model_class_name="Flux2KleinPipeline",
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()
    assert not loader._inflight
    fetch_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_remote_error_preserves_status_but_not_url():
    loader = SimpleNamespace(
        load_image=AsyncMock(
            side_effect=HttpStatusError(
                404, "private signed URL", "https://example.com/?secret=token"
            )
        )
    )
    with pytest.raises(HttpStatusError) as caught:
        await image_edit.load_image_edit_references(
            ["reference"], image_loader=loader, model_class_name="Flux2KleinPipeline"
        )
    assert caught.value.status == 404
    assert caught.value.message == "input_reference could not be loaded"
    assert "secret" not in str(caught.value)
