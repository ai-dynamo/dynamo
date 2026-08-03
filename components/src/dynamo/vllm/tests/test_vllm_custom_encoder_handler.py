# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from dynamo.vllm.handlers import (
    DecodeWorkerHandler,
    _attach_custom_encoder_data,
    _prepare_custom_encoder_results,
)
from dynamo.vllm.multimodal_utils.custom_encoder import (
    EncoderResult,
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


async def test_custom_encoder_handler_returns_adapter_prepared_prompt():
    handler = object.__new__(DecodeWorkerHandler)
    handler._custom_encoder_adapter = _adapter()
    handler._custom_encoder = SimpleNamespace(
        encode=AsyncMock(
            return_value=[
                EncoderResult(
                    artifact=torch.ones((2, 4), dtype=torch.bfloat16),
                    response_data={"score": 0.75},
                )
            ]
        )
    )

    prompt, response_data, error = await handler._assemble_custom_encoder_prompt(
        {
            "token_ids": [1, 99, 2],
            "multi_modal_data": {
                "image_url": [{"Url": "data:image/png;base64,unused"}]
            },
        },
        "request-id",
    )

    assert error is None
    assert response_data == {"items": [{"score": 0.75}]}
    assert prompt is not None
    assert tuple(prompt["prompt_embeds"].shape) == (4, 4)
    assert prompt["prompt_token_ids"] == [1, 99, 99, 2]


async def test_custom_encoder_handler_preserves_string_error_contract():
    handler = object.__new__(DecodeWorkerHandler)
    handler._custom_encoder_adapter = _adapter()
    handler._custom_encoder = SimpleNamespace(
        encode=AsyncMock(side_effect=RuntimeError("encoder failed"))
    )

    prompt, response_data, error = await handler._assemble_custom_encoder_prompt(
        {
            "token_ids": [99],
            "multi_modal_data": {
                "image_url": [{"Url": "data:image/png;base64,unused"}]
            },
        },
        "request-id",
    )

    assert prompt is None
    assert response_data is None
    assert error is not None
    assert error["finish_reason"] == "error: CustomEncoder failed: encoder failed"


def test_prepare_custom_encoder_results_preserves_null_positions():
    artifact = torch.ones((1, 4))
    response_item = {"score": 0.5}

    artifacts, response_data = _prepare_custom_encoder_results(
        [
            EncoderResult(artifact=artifact, response_data=response_item),
            EncoderResult(artifact=artifact),
        ]
    )
    response_item["score"] = 1.0

    assert artifacts == [artifact, artifact]
    assert response_data == {"items": [{"score": 0.5}, None]}


def test_prepare_custom_encoder_results_omits_empty_response_payload():
    artifact = torch.ones((1, 4))

    artifacts, response_data = _prepare_custom_encoder_results(
        [EncoderResult(artifact=artifact)]
    )

    assert artifacts == [artifact]
    assert response_data is None


@pytest.mark.parametrize(
    "result, message",
    [
        (torch.ones((1, 4)), "must return EncoderResult"),
        (
            EncoderResult(artifact=torch.ones((1, 4)), response_data={"bad": {1}}),
            "JSON-serializable",
        ),
        (
            EncoderResult(
                artifact=torch.ones((1, 4)),
                response_data={"score": float("nan")},
            ),
            "JSON-serializable",
        ),
    ],
)
def test_prepare_custom_encoder_results_rejects_invalid_results(result, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _prepare_custom_encoder_results([result])


def test_prepare_custom_encoder_results_enforces_response_limit():
    result = EncoderResult(
        artifact=torch.ones((1, 4)),
        response_data={"value": "x" * (64 * 1024)},
    )

    with pytest.raises(ValueError, match="64 KiB"):
        _prepare_custom_encoder_results([result])


def test_custom_encoder_data_attaches_only_to_first_success_chunk():
    response_data = {"items": [{"score": 0.75}]}
    first = {"token_ids": [1], "index": 0}
    second = {"token_ids": [2], "index": 0}

    sent = _attach_custom_encoder_data(
        first,
        response_data,
        requested=True,
        already_sent=False,
    )
    sent = _attach_custom_encoder_data(
        second,
        response_data,
        requested=True,
        already_sent=sent,
    )

    assert sent is True
    assert first["custom_encoder_data"] == response_data
    assert "custom_encoder_data" not in second


def test_custom_encoder_data_is_not_attached_to_error_chunk():
    error = {"finish_reason": "error: generation failed", "token_ids": []}

    sent = _attach_custom_encoder_data(
        error,
        {"items": [{"score": 0.75}]},
        requested=True,
        already_sent=False,
    )

    assert sent is False
    assert "custom_encoder_data" not in error


def test_custom_encoder_data_is_not_attached_without_opt_in() -> None:
    chunk = {"token_ids": [1], "index": 0}

    sent = _attach_custom_encoder_data(
        chunk,
        {"items": [{"score": 0.75}]},
        requested=False,
        already_sent=False,
    )

    assert sent is False
    assert "custom_encoder_data" not in chunk
