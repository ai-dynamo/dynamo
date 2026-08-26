# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for external encoder results consumed by stock vLLM."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from vllm.inputs import EmbedsPrompt

from dynamo.common.external_encoder import (
    ExternalEncoderResult,
    encode_request_plane_tensor,
)
from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm.constants import DisaggregationMode
from dynamo.vllm.handlers import DecodeWorkerHandler
from dynamo.vllm.multimodal_utils.external_encoder import ExternalEncoderPromptLoader

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]

_HIDDEN = 4
_IMAGE_TOKEN_ID = 99


def _model_config(
    *,
    dtype: torch.dtype = torch.bfloat16,
    multimodal: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        dtype=dtype,
        get_hidden_size=lambda: _HIDDEN,
        is_multimodal_model=multimodal,
    )


def _engine_args(*, enable_prompt_embeds: bool = True) -> SimpleNamespace:
    return SimpleNamespace(enable_prompt_embeds=enable_prompt_embeds)


def _encoder_result(
    packed: torch.Tensor | None = None,
    *,
    row_splits: tuple[int, ...] = (0, 2, 3),
) -> dict:
    if packed is None:
        packed = torch.arange(12, dtype=torch.bfloat16).reshape(3, _HIDDEN)
    return ExternalEncoderResult(
        features=encode_request_plane_tensor(packed),
        row_splits=row_splits,
        image_token_id=_IMAGE_TOKEN_ID,
    ).to_dict()


def _handler(
    mode: DisaggregationMode = DisaggregationMode.AGGREGATED,
) -> DecodeWorkerHandler:
    handler = object.__new__(DecodeWorkerHandler)
    handler.config = SimpleNamespace(
        disaggregation_mode=mode,
        engine_args=_engine_args(),
    )
    handler.model_config = _model_config()
    handler._external_encoder_prompt_loader = None
    handler._custom_encoder = None
    return handler


async def test_loader_builds_mixed_prompt_from_request_plane_features() -> None:
    packed = torch.arange(12, dtype=torch.bfloat16).reshape(3, _HIDDEN)
    loader = ExternalEncoderPromptLoader(_model_config(), _engine_args())

    prompt = await loader.load(
        _encoder_result(packed),
        [1, _IMAGE_TOKEN_ID, 2, _IMAGE_TOKEN_ID, 3],
    )

    assert prompt["prompt_token_ids"] == [1, 99, 99, 2, 99, 3]
    assert prompt["prompt_is_token_ids"] == [True, False, False, True, False, True]
    torch.testing.assert_close(prompt["prompt_embeds"][1:3], packed[:2])
    torch.testing.assert_close(prompt["prompt_embeds"][4], packed[2])


@pytest.mark.parametrize(
    "model_config,engine_args,match",
    [
        (_model_config(multimodal=True), _engine_args(), "text-only decoder"),
        (_model_config(), _engine_args(enable_prompt_embeds=False), "prompt-embeds"),
    ],
)
def test_loader_rejects_incompatible_decoder_configuration(
    model_config: SimpleNamespace,
    engine_args: SimpleNamespace,
    match: str,
) -> None:
    with pytest.raises(RuntimeError, match=match):
        ExternalEncoderPromptLoader(model_config, engine_args)


@pytest.mark.parametrize(
    "packed,row_splits,match",
    [
        (
            torch.ones((3, _HIDDEN + 1), dtype=torch.bfloat16),
            (0, 3),
            "hidden size",
        ),
        (
            torch.ones((3, _HIDDEN), dtype=torch.float32),
            (0, 3),
            "expected decoder dtype",
        ),
        (
            torch.ones((2, _HIDDEN), dtype=torch.bfloat16),
            (0, 0, 2),
            "empty image",
        ),
    ],
)
async def test_loader_rejects_invalid_tensor(
    packed: torch.Tensor,
    row_splits: tuple[int, ...],
    match: str,
) -> None:
    loader = ExternalEncoderPromptLoader(_model_config(), _engine_args())
    token_ids = [_IMAGE_TOKEN_ID] * (len(row_splits) - 1)

    with pytest.raises(InvalidArgument, match=match):
        await loader.load(
            _encoder_result(packed, row_splits=row_splits),
            token_ids,
        )


async def test_handler_assembles_external_prompt_through_shared_loader() -> None:
    handler = _handler()
    expected = EmbedsPrompt(
        prompt_embeds=torch.ones((1, _HIDDEN), dtype=torch.bfloat16),
        prompt_token_ids=[_IMAGE_TOKEN_ID],
        prompt_is_token_ids=[False],
    )
    loader = SimpleNamespace(load=AsyncMock(return_value=expected))
    handler._external_encoder_prompt_loader = loader
    request = {
        "token_ids": [_IMAGE_TOKEN_ID],
        "encoder_result": _encoder_result(row_splits=(0, 3)),
    }

    prompt = await handler._assemble_external_encoder_prompt(request, "req-1")

    assert prompt is expected
    loader.load.assert_awaited_once_with(
        request["encoder_result"],
        request["token_ids"],
    )


async def test_handler_rejects_competing_raw_media() -> None:
    handler = _handler()
    request = {
        "token_ids": [_IMAGE_TOKEN_ID],
        "encoder_result": _encoder_result(row_splits=(0, 3)),
        "multi_modal_data": {"image_url": [{"Url": "unused"}]},
    }

    with pytest.raises(InvalidArgument, match="authoritative"):
        await handler._assemble_external_encoder_prompt(request, "req-1")


async def test_handler_propagates_external_encoder_runtime_failure() -> None:
    handler = _handler()
    handler._external_encoder_prompt_loader = SimpleNamespace(
        load=AsyncMock(side_effect=RuntimeError("allocation failed"))
    )
    request = {
        "token_ids": [_IMAGE_TOKEN_ID],
        "encoder_result": _encoder_result(row_splits=(0, 3)),
    }

    with pytest.raises(RuntimeError, match="allocation failed"):
        await handler._assemble_external_encoder_prompt(request, "req-1")


async def test_handler_rejects_external_result_on_disaggregated_worker() -> None:
    handler = _handler(DisaggregationMode.DECODE)

    chunks = [
        chunk
        async for chunk in handler._generate_token_mode(
            {"encoder_result": _encoder_result(row_splits=(0, 3))},
            MagicMock(),
            "req-1",
        )
    ]

    assert len(chunks) == 1
    assert "aggregated vLLM worker" in chunks[0]["finish_reason"]


async def test_aggregated_token_path_selects_external_prompt_assembly() -> None:
    handler = _handler()
    handler._assemble_external_encoder_prompt = AsyncMock(
        side_effect=InvalidArgument(
            "stop after selection",
        )
    )
    request = {"encoder_result": _encoder_result(row_splits=(0, 3))}

    with pytest.raises(InvalidArgument, match="stop after selection"):
        _ = [
            chunk
            async for chunk in handler._generate_token_mode(
                request,
                MagicMock(),
                "req-1",
            )
        ]
    handler._assemble_external_encoder_prompt.assert_awaited_once_with(
        request,
        "req-1",
    )
