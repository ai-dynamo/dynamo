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


async def test_custom_encoder_handler_rejects_encoder_failure_with_message():
    """An encoder failure must reach the caller as a typed rejection.

    `InvalidArgument` is what the bindings translate into a backend error the
    frontend answers with HTTP 400, forwarding the message verbatim. Reporting
    the same condition through `finish_reason` strips the message.
    """
    handler = object.__new__(DecodeWorkerHandler)
    handler._custom_encoder_adapter = _adapter()
    handler._custom_encoder = SimpleNamespace(
        encode=AsyncMock(side_effect=RuntimeError("encoder failed"))
    )

    with pytest.raises(InvalidArgument) as excinfo:
        await handler._assemble_custom_encoder_prompt(
            {
                "token_ids": [99],
                "multi_modal_data": {
                    "image_url": [{"Url": "data:image/png;base64,unused"}]
                },
            },
            "request-id",
        )

    assert str(excinfo.value) == "CustomEncoder failed: encoder failed"


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
