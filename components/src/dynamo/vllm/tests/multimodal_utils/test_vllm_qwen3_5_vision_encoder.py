# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the Qwen3.5 custom vision-encoder example."""

import base64
import io
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from dynamo.vllm.multimodal_utils.custom_encoder import (
    Preprocessed,
    Qwen3VLImageEncoding,
)
from examples.custom_encoder.qwen3_5_vision_encoder import (
    Qwen35ImageInputs,
    Qwen35VisionEncoder,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class _FakeImageProcessor:
    def __call__(self, *, images, return_tensors):
        assert len(images) == 1
        assert return_tensors == "pt"
        return {
            "pixel_values": torch.ones((4, 6)),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }


class _FakeVisual(torch.nn.Module):
    dtype = torch.bfloat16
    spatial_merge_size = 2

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        *,
        return_dict: bool,
    ) -> SimpleNamespace:
        assert pixel_values.dtype == torch.bfloat16
        assert return_dict is True
        token_count = int((grid_thw.prod(dim=-1) // 4).sum().item())
        return SimpleNamespace(
            pooler_output=torch.arange(
                token_count * 4,
                dtype=torch.bfloat16,
            ).reshape(token_count, 4)
        )


def _item(grid: tuple[int, int, int]) -> Qwen35ImageInputs:
    return Qwen35ImageInputs(
        pixel_values=torch.ones((grid[0] * grid[1] * grid[2], 6)),
        image_grid_thw=torch.tensor([grid]),
    )


def _png_data_url() -> str:
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), color="green").save(stream, format="PNG")
    encoded = base64.b64encode(stream.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def test_preprocess_returns_processor_output():
    encoder = Qwen35VisionEncoder()
    encoder._processor = SimpleNamespace(image_processor=_FakeImageProcessor())

    result = encoder.preprocess(_png_data_url())

    assert isinstance(result, Preprocessed)
    assert result.cost == 1
    assert result.item.pixel_values.shape == (4, 6)
    assert result.item.image_grid_thw.tolist() == [[1, 2, 2]]


@pytest.mark.parametrize(
    "raw",
    [
        "https://example.com/image.png",
        "file:///etc/passwd",
        "data:text/plain;base64,aGVsbG8=",
        "data:image/png,not-base64",
    ],
)
def test_preprocess_rejects_non_base64_image_data_urls(raw):
    encoder = Qwen35VisionEncoder()
    encoder._processor = SimpleNamespace(image_processor=_FakeImageProcessor())

    with pytest.raises(ValueError, match="inline base64 image data URL"):
        encoder.preprocess(raw)


def test_preprocess_rejects_invalid_base64():
    encoder = Qwen35VisionEncoder()
    encoder._processor = SimpleNamespace(image_processor=_FakeImageProcessor())

    with pytest.raises(ValueError, match="invalid base64"):
        encoder.preprocess("data:image/png;base64,not_valid!")


def test_preprocess_rejects_oversized_inline_image(monkeypatch):
    encoder = Qwen35VisionEncoder()
    encoder._processor = SimpleNamespace(image_processor=_FakeImageProcessor())
    monkeypatch.setattr(
        "examples.custom_encoder.qwen3_5_vision_encoder._MAX_BASE64_CHARS", 4
    )

    with pytest.raises(ValueError, match="exceeds"):
        encoder.preprocess("data:image/png;base64,AAAAA")


def test_forward_batch_splits_qwen_artifacts_in_input_order():
    encoder = Qwen35VisionEncoder()
    encoder._device = torch.device("cpu")
    encoder._visual = _FakeVisual()

    outputs = encoder.forward_batch([_item((1, 2, 2)), _item((1, 2, 4))])

    assert all(isinstance(output, Qwen3VLImageEncoding) for output in outputs)
    assert [output.grid_thw for output in outputs] == [(1, 2, 2), (1, 2, 4)]
    assert [tuple(output.embeddings.shape) for output in outputs] == [(1, 4), (2, 4)]
    assert outputs[0].embeddings[:, 0].tolist() == [0]
    assert outputs[1].embeddings[:, 0].tolist() == [4, 8]
    assert all(output.embeddings.device.type == "cpu" for output in outputs)


def test_close_releases_example_resources():
    encoder = Qwen35VisionEncoder()
    encoder._processor = object()
    encoder._visual = _FakeVisual()

    encoder.close()

    assert encoder._processor is None
    assert encoder._visual is None
