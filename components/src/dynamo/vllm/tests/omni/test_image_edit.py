# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

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


@pytest.fixture
def metadata_lookup(monkeypatch):
    lookup = Mock(
        return_value=SimpleNamespace(
            supports_multimodal_inputs=True, max_multimodal_image_inputs=4
        )
    )
    monkeypatch.setattr(image_edit, "get_diffusion_model_metadata", lookup)
    return lookup


@pytest.mark.parametrize("operator_limit, expected", [(1, 1), (2, 2), (4, 4), (8, 4)])
def test_model_and_operator_limits(metadata_lookup, operator_limit, expected):
    assert (
        image_edit.image_edit_input_limit(
            "Flux2KleinPipeline", operator_limit=operator_limit
        )
        == expected
    )
    metadata_lookup.assert_called_once_with("Flux2KleinPipeline")


@pytest.mark.parametrize("operator_limit", [0, True, 1.5])
def test_invalid_operator_configuration(metadata_lookup, operator_limit):
    with pytest.raises(ValueError, match="positive integer"):
        image_edit.image_edit_input_limit(
            "Flux2KleinPipeline", operator_limit=operator_limit
        )
    metadata_lookup.assert_not_called()


@pytest.mark.parametrize("model_limit", [None, 0, True, 2.5])
def test_unusable_model_limit_allows_only_single_image(metadata_lookup, model_limit):
    metadata_lookup.return_value.max_multimodal_image_inputs = model_limit
    assert image_edit.image_edit_input_limit("custom") == 1


@pytest.mark.parametrize("support", [False, None, 1, "true"])
def test_multi_input_requires_explicit_support(metadata_lookup, support):
    metadata_lookup.return_value.supports_multimodal_inputs = support
    assert image_edit.image_edit_input_limit("custom") == 1


def test_missing_metadata_fields_preserve_single_image(metadata_lookup):
    metadata_lookup.return_value = SimpleNamespace()
    assert image_edit.image_edit_input_limit("custom") == 1


def test_missing_optional_registry_preserves_single_image(monkeypatch):
    monkeypatch.setattr(image_edit, "get_diffusion_model_metadata", None)
    assert image_edit.image_edit_input_limit("custom") == 1


def test_video_reference_limits_do_not_enable_multiple_image_edits(metadata_lookup):
    metadata_lookup.return_value.final_output_type = "video"
    metadata_lookup.return_value.max_multimodal_image_inputs = 9
    assert image_edit.image_edit_input_limit("video-pipeline") == 1


def test_invalid_output_metadata_preserves_single_image(metadata_lookup):
    metadata_lookup.return_value.final_output_type = True
    assert image_edit.image_edit_input_limit("custom") == 1


@pytest.mark.skipif(
    image_edit.get_diffusion_model_metadata is None,
    reason="Installed Omni does not publish diffusion capability metadata",
)
@pytest.mark.parametrize(
    "pipeline,expected",
    [
        ("Flux2KleinPipeline", 4),
        ("HunyuanImage3ForCausalMM", 3),
        ("BooguImageTurboPipeline", 1),
    ],
)
def test_installed_registry_resolves_pipeline_aliases(pipeline, expected):
    assert image_edit.image_edit_input_limit(pipeline) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [0, 5])
async def test_invalid_count_rejected_before_any_fetch(metadata_lookup, count):
    loader = SimpleNamespace(load_image=AsyncMock())
    with pytest.raises(ValueError):
        await image_edit.load_image_edit_references(
            ["https://example.com/image"] * count,
            image_loader=loader,
            model_class_name="Flux2KleinPipeline",
        )
    loader.load_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_model_rejects_multiple_before_fetch(metadata_lookup):
    metadata_lookup.return_value = SimpleNamespace()
    loader = SimpleNamespace(load_image=AsyncMock())
    with pytest.raises(ValueError, match="at most 1"):
        await image_edit.load_image_edit_references(
            ["first", "second"], image_loader=loader, model_class_name="unknown"
        )
    loader.load_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_second_reference_rejected_before_any_fetch(metadata_lookup):
    loader = SimpleNamespace(load_image=AsyncMock())
    with pytest.raises(ValueError, match=r"^Failed to load input_reference\[1\]$"):
        await image_edit.load_image_edit_references(
            ["first", " "], image_loader=loader, model_class_name="Flux2KleinPipeline"
        )
    loader.load_image.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2, 3, 4])
async def test_every_reference_loaded_in_order(metadata_lookup, count):
    references = [f"image-{index}" for index in range(count)]
    images = [Image.new("RGB", (2, 2), (index, 0, 0)) for index in range(count)]
    loader = SimpleNamespace(load_image=AsyncMock(side_effect=images))
    loaded = await image_edit.load_image_edit_references(
        references, image_loader=loader, model_class_name="Flux2KleinPipeline"
    )
    assert loaded == images
    assert [call.args[0] for call in loader.load_image.await_args_list] == references
    assert [image.getpixel((0, 0)) for image in loaded] == [
        (index, 0, 0) for index in range(count)
    ]
    for image in loaded:
        image.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [ValueError, Image.DecompressionBombError])
async def test_bad_second_reference_preserves_shared_images_and_redacts_error(
    metadata_lookup, failure
):
    first = Image.new("RGB", (2, 2), "red")
    loader = SimpleNamespace(
        load_image=AsyncMock(side_effect=[first, failure("private signed URL")])
    )
    with pytest.raises(ValueError, match=r"^Failed to load input_reference\[1\]$"):
        await image_edit.load_image_edit_references(
            ["first", "second", "third"],
            image_loader=loader,
            model_class_name="Flux2KleinPipeline",
        )
    assert loader.load_image.await_count == 2
    assert first.getpixel((0, 0)) == (255, 0, 0)
    first.close()


@pytest.mark.asyncio
async def test_single_image_error_stays_compatible(metadata_lookup):
    loader = SimpleNamespace(load_image=AsyncMock(side_effect=ValueError("secret")))
    with pytest.raises(ValueError, match="^Failed to load input_reference$"):
        await image_edit.load_image_edit_references(
            ["reference"], image_loader=loader, model_class_name="Flux2KleinPipeline"
        )


@pytest.mark.asyncio
async def test_failed_edit_does_not_poison_the_image_cache(
    metadata_lookup, monkeypatch
):
    with io.BytesIO() as buffer:
        Image.new("RGB", (2, 2), "red").save(buffer, format="PNG")
        encoded = buffer.getvalue()
    fetch = AsyncMock(return_value=encoded)
    monkeypatch.setattr("dynamo.common.multimodal.image_loader.fetch_bytes", fetch)
    loader = ImageLoader(session_scoped_cache=False)
    reference = "https://8.8.8.8/reference.png"

    with pytest.raises(ValueError, match=r"^Failed to load input_reference\[1\]$"):
        await image_edit.load_image_edit_references(
            [reference, "data:image/png;base64,%%%"],
            image_loader=loader,
            model_class_name="Flux2KleinPipeline",
        )

    cached = await loader.load_image(reference)
    assert cached.getpixel((0, 0)) == (255, 0, 0)
    fetch.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [asyncio.CancelledError, RuntimeError])
async def test_cancellation_and_internal_errors_preserve_shared_images(
    metadata_lookup, failure
):
    first = Image.new("RGB", (2, 2), "red")
    error = failure("interrupted")
    loader = SimpleNamespace(load_image=AsyncMock(side_effect=[first, error]))
    with pytest.raises(failure) as caught:
        await image_edit.load_image_edit_references(
            ["first", "second"],
            image_loader=loader,
            model_class_name="Flux2KleinPipeline",
        )
    assert caught.value is error
    assert first.getpixel((0, 0)) == (255, 0, 0)
    first.close()
