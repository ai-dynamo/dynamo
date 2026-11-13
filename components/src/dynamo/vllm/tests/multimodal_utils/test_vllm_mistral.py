# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Mistral-format image placeholder expansion."""

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image, ImageOps

from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm.multimodal_utils.models.mistral import MistralImagePlaceholders

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

_IMG, _IMG_BREAK, _IMG_END = 10, 12, 13


def _grid(ncols: int, nrows: int) -> list[int]:
    grid = ([_IMG] * ncols + [_IMG_BREAK]) * nrows
    grid[-1] = _IMG_END
    return grid


def _tokenizer(encoder: object) -> SimpleNamespace:
    return SimpleNamespace(instruct=SimpleNamespace(mm_encoder=encoder))


def _placeholders() -> MistralImagePlaceholders:
    """Grid sizing as mistral-common does it for patch 14, merge 2."""
    encoder = SimpleNamespace(
        special_ids=SimpleNamespace(img=_IMG, img_break=_IMG_BREAK, img_end=_IMG_END),
        _image_to_num_tokens=lambda image: (
            (image.size[0] - 1) // 28 + 1,
            (image.size[1] - 1) // 28 + 1,
        ),
    )
    placeholders = MistralImagePlaceholders.from_tokenizer(_tokenizer(encoder))
    assert placeholders is not None
    return placeholders


def _loader_image(orientation: int | None) -> Image.Image:
    exif = Image.Exif()
    if orientation is not None:
        exif[274] = orientation
    with BytesIO() as image_bytes:
        Image.new("RGB", (56, 28)).save(image_bytes, format="JPEG", exif=exif)
        image_bytes.seek(0)
        with Image.open(image_bytes) as image:
            return image.convert("RGB")


def test_placeholders_expand_to_each_images_grid():
    result = _placeholders().expand(
        [1, _IMG, 7, _IMG, 2],
        [Image.new("RGB", (256, 256)), Image.new("RGB", (448, 224))],
    )

    assert result == [1, *_grid(10, 10), 7, *_grid(16, 8), 2]


def test_grid_matches_mistral_common():
    image_module = pytest.importorskip("mistral_common.tokens.tokenizers.image")
    encoder = image_module.ImageEncoder(
        image_module.ImageConfig(
            image_patch_size=14, max_image_size=1540, spatial_merge_size=2
        ),
        image_module.SpecialImageIDs(img=_IMG, img_break=_IMG_BREAK, img_end=_IMG_END),
    )
    placeholders = MistralImagePlaceholders.from_tokenizer(_tokenizer(encoder))
    assert placeholders is not None

    result = placeholders.expand([_IMG], [Image.new("RGB", (256, 256))])

    # The 256x256 probe image is 110 tokens upstream (137 prompt tokens).
    assert result == _grid(10, 10)
    assert len(result) == 110


@pytest.mark.parametrize("image", [Image.new("RGB", (56, 56)), None])
def test_prompt_expanded_upstream_is_untouched(image: Image.Image | None) -> None:
    expanded = [1, *_grid(2, 2), 2]

    assert _placeholders().expand(expanded, [image]) is expanded


def test_placeholder_count_must_match_images() -> None:
    with pytest.raises(InvalidArgument, match="refusing to expand"):
        _placeholders().expand([_IMG], [Image.new("RGB", (28, 28))] * 2)


def test_uuid_only_slots_cannot_be_expanded() -> None:
    with pytest.raises(InvalidArgument, match="UUID-only"):
        _placeholders().expand([_IMG], [None])


@pytest.mark.parametrize("orientation", [None, 1, 2, 3, 4, 5, 6, 7, 8])
def test_grid_matches_stock_vllm_exif_normalization(orientation: int | None) -> None:
    image_module = pytest.importorskip("mistral_common.tokens.tokenizers.image")
    chunk_module = pytest.importorskip("mistral_common.protocol.instruct.chunk")
    encoder = image_module.ImageEncoder(
        image_module.ImageConfig(
            image_patch_size=14, max_image_size=1540, spatial_merge_size=2
        ),
        image_module.SpecialImageIDs(img=_IMG, img_break=_IMG_BREAK, img_end=_IMG_END),
    )
    placeholders = MistralImagePlaceholders.from_tokenizer(_tokenizer(encoder))
    assert placeholders is not None
    image = _loader_image(orientation)

    # Stock vLLM normalizes PIL images before the public mistral-common encoder.
    expected = encoder(chunk_module.ImageChunk(image=ImageOps.exif_transpose(image)))
    assert placeholders.expand([_IMG], [image]) == expected.tokens
    assert image.size == (56, 28)
    assert image.getexif().get(274) == orientation


@pytest.mark.parametrize("orientation", [None, 1])
def test_unoriented_images_skip_the_exif_pixel_copy(
    orientation: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    transpose = MagicMock(side_effect=AssertionError("unexpected pixel copy"))
    monkeypatch.setattr(ImageOps, "exif_transpose", transpose)

    assert _placeholders().expand([_IMG], [_loader_image(orientation)]) == _grid(2, 1)
    transpose.assert_not_called()


@pytest.mark.parametrize("sizing_helper", [None, 42], ids=["missing", "not-callable"])
def test_mistral_encoder_requires_a_callable_sizing_helper(
    sizing_helper: object,
) -> None:
    encoder = SimpleNamespace(
        special_ids=SimpleNamespace(img=_IMG, img_break=_IMG_BREAK, img_end=_IMG_END)
    )
    if sizing_helper is not None:
        encoder._image_to_num_tokens = sizing_helper

    with pytest.raises(RuntimeError, match="callable _image_to_num_tokens"):
        MistralImagePlaceholders.from_tokenizer(_tokenizer(encoder))


@pytest.mark.parametrize(
    "tokenizer",
    [SimpleNamespace(encode=MagicMock()), MagicMock(), None],
    ids=["no-mm-encoder", "mock-attributes", "none"],
)
def test_tokenizers_without_an_image_encoder_have_no_placeholders(tokenizer: object):
    assert MistralImagePlaceholders.from_tokenizer(tokenizer) is None
