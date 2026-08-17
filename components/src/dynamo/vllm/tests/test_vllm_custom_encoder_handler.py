# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm.handlers import DecodeWorkerHandler
from dynamo.vllm.multimodal_utils.custom_encoder import (
    Qwen3VLImageEncoding,
    VisionEncoderBackend,
    create_custom_encoder_adapter,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class _Backend(VisionEncoderBackend):
    image_token_id = 99

    def build(self, model_id: str) -> None:
        pass

    def forward_batch(self, items, target_bucket=None):
        raise NotImplementedError


class _QwenBackend(_Backend):
    image_token_id = None


def _adapter():
    return create_custom_encoder_adapter(
        _Backend(),
        SimpleNamespace(
            dtype=torch.bfloat16,
            get_hidden_size=lambda: 4,
            is_multimodal_model=False,
        ),
        SimpleNamespace(enable_prompt_embeds=True),
    )


def _qwen_adapter():
    return create_custom_encoder_adapter(
        _QwenBackend(),
        SimpleNamespace(
            is_multimodal_model=lambda: True,
            architectures=["Qwen3VLForConditionalGeneration"],
            hf_config=SimpleNamespace(
                vision_config=SimpleNamespace(spatial_merge_size=2),
            ),
        ),
        SimpleNamespace(),
    )


async def test_custom_encoder_handler_returns_adapter_prepared_prompt():
    handler = object.__new__(DecodeWorkerHandler)
    handler._custom_encoder_adapter = _adapter()
    handler._custom_encoder = SimpleNamespace(
        encode=AsyncMock(return_value=[torch.ones((2, 4), dtype=torch.bfloat16)])
    )

    prompt = await handler._assemble_custom_encoder_prompt(
        {
            "token_ids": [1, 99, 2],
            "multi_modal_data": {
                "image_url": [{"Url": "data:image/png;base64,unused"}]
            },
        },
        "request-id",
    )

    assert prompt is not None
    assert tuple(prompt["prompt_embeds"].shape) == (4, 4)
    assert prompt["prompt_token_ids"] == [1, 99, 99, 2]


async def _assemble_with_encoder_error(exc: Exception):
    handler = object.__new__(DecodeWorkerHandler)
    handler._custom_encoder_adapter = _adapter()
    handler._custom_encoder = SimpleNamespace(encode=AsyncMock(side_effect=exc))
    return await handler._assemble_custom_encoder_prompt(
        {
            "token_ids": [99],
            "multi_modal_data": {
                "image_url": [{"Url": "data:image/png;base64,unused"}]
            },
        },
        "request-id",
    )


async def test_custom_encoder_input_fault_propagates_as_value_error():
    """An input fault stays a `ValueError`, which the bindings already map to
    `Backend(InvalidArgument)` — HTTP 400 carrying the message. The adapters
    raise `ValueError`/`TypeError` for every validation failure, so the
    actionable case needs no conversion here."""
    with pytest.raises(ValueError) as excinfo:
        await _assemble_with_encoder_error(ValueError("placeholder tokens (0) != 1"))

    assert "placeholder tokens (0) != 1" in str(excinfo.value)


async def test_custom_encoder_engine_fault_keeps_its_own_type():
    """The complement, and the point of not converting: an engine fault must
    not be relabelled a client error. Doing so would answer 400 and suppress
    retries for timeouts, CUDA faults and batcher shutdown — and since
    `encode()` is co-batched, could blame a caller for another request's
    failure."""
    with pytest.raises(RuntimeError) as excinfo:
        await _assemble_with_encoder_error(RuntimeError("CUDA error: out of memory"))

    assert not isinstance(excinfo.value, InvalidArgument)
    assert "out of memory" in str(excinfo.value)


async def test_custom_encoder_timeout_keeps_its_own_type():
    with pytest.raises(TimeoutError):
        await _assemble_with_encoder_error(TimeoutError("encoder timed out"))


async def test_custom_encoder_handler_rejects_unsupported_modality():
    handler = object.__new__(DecodeWorkerHandler)
    handler._custom_encoder_adapter = _adapter()
    handler._custom_encoder = SimpleNamespace(encode=AsyncMock())

    with pytest.raises(InvalidArgument) as excinfo:
        await handler._assemble_custom_encoder_prompt(
            {"token_ids": [1], "multi_modal_data": {"video_url": [{"Url": "v"}]}},
            "request-id",
        )

    assert "image inputs only" in str(excinfo.value)
    assert "video_url" in str(excinfo.value)


async def test_custom_encoder_handler_rejects_image_item_without_url():
    handler = object.__new__(DecodeWorkerHandler)
    handler._custom_encoder_adapter = _adapter()
    handler._custom_encoder = SimpleNamespace(encode=AsyncMock())

    with pytest.raises(InvalidArgument) as excinfo:
        await handler._assemble_custom_encoder_prompt(
            {"token_ids": [1], "multi_modal_data": {"image_url": [{"Decoded": "x"}]}},
            "request-id",
        )

    assert "'Url'" in str(excinfo.value)


async def test_custom_encoder_handler_returns_none_for_text_only_request():
    handler = object.__new__(DecodeWorkerHandler)
    handler._custom_encoder_adapter = _adapter()
    handler._custom_encoder = SimpleNamespace(encode=AsyncMock())

    prompt = await handler._assemble_custom_encoder_prompt(
        {"token_ids": [1, 2], "multi_modal_data": {}},
        "request-id",
    )

    assert prompt is None


async def test_custom_encoder_handler_returns_native_qwen3_vl_prompt():
    handler = object.__new__(DecodeWorkerHandler)
    handler._custom_encoder_adapter = _qwen_adapter()
    handler._custom_encoder = SimpleNamespace(
        encode=AsyncMock(
            return_value=[
                Qwen3VLImageEncoding(
                    torch.zeros((1, 8), dtype=torch.bfloat16), (1, 2, 2)
                )
            ]
        )
    )

    prompt = await handler._assemble_custom_encoder_prompt(
        {
            "token_ids": [100, 101, 102],
            "multi_modal_data": {
                "image_url": [{"Url": "data:image/png;base64,unused"}]
            },
        },
        "request-id",
    )

    assert prompt is not None
    assert prompt["prompt_token_ids"] == [100, 101, 102]
    image = prompt["multi_modal_data"]["image"]
    assert image["image_embeds"].shape == (1, 8)
    assert image["image_grid_thw"].tolist() == [[1, 2, 2]]
