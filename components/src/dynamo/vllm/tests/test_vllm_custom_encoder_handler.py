# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from dynamo.vllm.constants import DisaggregationMode
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

    prompt, error = await handler._assemble_custom_encoder_prompt(
        {
            "token_ids": [1, 99, 2],
            "multi_modal_data": {
                "image_url": [{"Url": "data:image/png;base64,unused"}]
            },
        },
        "request-id",
    )

    assert error is None
    assert prompt is not None
    assert tuple(prompt["prompt_embeds"].shape) == (4, 4)
    assert prompt["prompt_token_ids"] == [1, 99, 99, 2]


async def test_custom_encoder_handler_preserves_string_error_contract():
    handler = object.__new__(DecodeWorkerHandler)
    handler._custom_encoder_adapter = _adapter()
    handler._custom_encoder = SimpleNamespace(
        encode=AsyncMock(side_effect=RuntimeError("encoder failed"))
    )

    prompt, error = await handler._assemble_custom_encoder_prompt(
        {
            "token_ids": [99],
            "multi_modal_data": {
                "image_url": [{"Url": "data:image/png;base64,unused"}]
            },
        },
        "request-id",
    )

    assert prompt is None
    assert error is not None
    assert error["finish_reason"] == "error: CustomEncoder failed: encoder failed"


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

    prompt, error = await handler._assemble_custom_encoder_prompt(
        {
            "token_ids": [100, 101, 102],
            "multi_modal_data": {
                "image_url": [{"Url": "data:image/png;base64,unused"}]
            },
        },
        "request-id",
    )

    assert error is None
    assert prompt is not None
    assert prompt["prompt_token_ids"] == [100, 101, 102]
    image = prompt["multi_modal_data"]["image"]
    assert image["image_embeds"].shape == (1, 8)
    assert image["image_grid_thw"].tolist() == [[1, 2, 2]]


async def test_custom_encoder_handler_passes_opaque_image_payloads():
    handler = object.__new__(DecodeWorkerHandler)
    handler._custom_encoder_adapter = _qwen_adapter()
    handler._custom_encoder = SimpleNamespace(
        encode=AsyncMock(
            return_value=[
                Qwen3VLImageEncoding(
                    torch.zeros((1, 8), dtype=torch.bfloat16), (1, 2, 2)
                ),
                Qwen3VLImageEncoding(
                    torch.ones((1, 8), dtype=torch.bfloat16), (1, 2, 2)
                ),
            ]
        )
    )
    payloads = [
        {"kind": "tensor_ref", "uri": "s3://bucket/first"},
        {"kind": "tensor_ref", "uri": "s3://bucket/second"},
    ]

    prompt, error = await handler._assemble_custom_encoder_prompt(
        {
            "token_ids": [100, 101, 102, 103],
            "multi_modal_data": {
                "custom_encoder_data": [
                    {
                        "CustomEncoderData": {
                            "modality": "image",
                            "payload": payloads[0],
                        }
                    },
                    {
                        "CustomEncoderData": {
                            "modality": "image",
                            "payload": payloads[1],
                        }
                    },
                ]
            },
        },
        "request-id",
    )

    assert error is None
    assert prompt is not None
    handler._custom_encoder.encode.assert_awaited_once_with(payloads)
    assert "multi_modal_uuids" not in prompt


async def test_custom_encoder_handler_rejects_unsupported_modality():
    handler = object.__new__(DecodeWorkerHandler)
    handler._custom_encoder_adapter = _qwen_adapter()
    handler._custom_encoder = SimpleNamespace(encode=AsyncMock())

    prompt, error = await handler._assemble_custom_encoder_prompt(
        {
            "token_ids": [100, 101, 102],
            "multi_modal_data": {
                "custom_encoder_data": [
                    {
                        "CustomEncoderData": {
                            "modality": "video",
                            "payload": {"kind": "tensor_ref"},
                        }
                    }
                ]
            },
        },
        "request-id",
    )

    assert prompt is None
    assert error is not None
    assert "only 'image' is currently supported" in error["finish_reason"]
    handler._custom_encoder.encode.assert_not_awaited()


async def test_custom_encoder_handler_rejects_mixed_custom_and_legacy_inputs():
    handler = object.__new__(DecodeWorkerHandler)
    handler._custom_encoder_adapter = _adapter()
    handler._custom_encoder = SimpleNamespace(encode=AsyncMock())

    prompt, error = await handler._assemble_custom_encoder_prompt(
        {
            "token_ids": [99],
            "multi_modal_data": {
                "custom_encoder_data": [
                    {
                        "CustomEncoderData": {
                            "modality": "image",
                            "payload": {"id": 1},
                        }
                    }
                ],
                "image_url": [{"Url": "data:image/png;base64,unused"}],
            },
        },
        "request-id",
    )

    assert prompt is None
    assert error is not None
    assert "cannot combine" in error["finish_reason"]
    handler._custom_encoder.encode.assert_not_awaited()


@pytest.mark.parametrize(
    "multi_modal_data, expected_error",
    [
        ([], "multi_modal_data must be an object"),
        *[
            ({"custom_encoder_data": value}, "custom_encoder_data must be a list")
            for value in (0, 1, {}, "")
        ],
        *[
            ({"image_url": value}, "image_url must be a list")
            for value in (0, 1, {}, "")
        ],
    ],
)
async def test_custom_encoder_handler_rejects_malformed_modality_containers(
    multi_modal_data,
    expected_error,
):
    handler = object.__new__(DecodeWorkerHandler)
    handler._custom_encoder_adapter = _adapter()
    handler._custom_encoder = SimpleNamespace(encode=AsyncMock())

    prompt, error = await handler._assemble_custom_encoder_prompt(
        {"token_ids": [99], "multi_modal_data": multi_modal_data},
        "request-id",
    )

    assert prompt is None
    assert error is not None
    assert expected_error in error["finish_reason"]
    handler._custom_encoder.encode.assert_not_awaited()


async def test_custom_encoder_handler_leaves_text_only_request_unchanged():
    handler = object.__new__(DecodeWorkerHandler)
    handler._custom_encoder_adapter = _adapter()
    handler._custom_encoder = SimpleNamespace(encode=AsyncMock())

    prompt, error = await handler._assemble_custom_encoder_prompt(
        {"token_ids": [1, 2, 3]},
        "request-id",
    )

    assert prompt is None
    assert error is None
    handler._custom_encoder.encode.assert_not_awaited()


async def test_custom_encoder_data_requires_configured_worker():
    handler = object.__new__(DecodeWorkerHandler)
    handler.config = SimpleNamespace(disaggregation_mode=DisaggregationMode.AGGREGATED)
    handler._custom_encoder = None

    chunks = [
        chunk
        async for chunk in handler._generate_token_mode(
            {
                "token_ids": [1, 2, 3],
                "multi_modal_data": {
                    "custom_encoder_data": [
                        {
                            "CustomEncoderData": {
                                "modality": "image",
                                "payload": {"id": 1},
                            }
                        }
                    ]
                },
            },
            None,
            "request-id",
        )
    ]

    assert chunks == [
        {
            "finish_reason": (
                "error: custom_encoder_data requires a worker configured with "
                "--custom-encoder-class"
            ),
            "token_ids": [],
        }
    ]


async def test_token_mode_rejects_non_mapping_multimodal_data():
    handler = object.__new__(DecodeWorkerHandler)
    handler.config = SimpleNamespace(disaggregation_mode=DisaggregationMode.AGGREGATED)
    handler._custom_encoder = None

    chunks = [
        chunk
        async for chunk in handler._generate_token_mode(
            {"token_ids": [1, 2, 3], "multi_modal_data": []},
            None,
            "request-id",
        )
    ]

    assert chunks == [
        {
            "finish_reason": "error: multi_modal_data must be an object",
            "token_ids": [],
        }
    ]


@pytest.mark.parametrize("custom_encoder_data", [0, 1, {}, ""])
async def test_token_mode_rejects_non_list_custom_encoder_data(custom_encoder_data):
    handler = object.__new__(DecodeWorkerHandler)
    handler.config = SimpleNamespace(disaggregation_mode=DisaggregationMode.AGGREGATED)
    handler._custom_encoder = None

    chunks = [
        chunk
        async for chunk in handler._generate_token_mode(
            {
                "token_ids": [1, 2, 3],
                "multi_modal_data": {"custom_encoder_data": custom_encoder_data},
            },
            None,
            "request-id",
        )
    ]

    assert chunks == [
        {
            "finish_reason": "error: custom_encoder_data must be a list",
            "token_ids": [],
        }
    ]
