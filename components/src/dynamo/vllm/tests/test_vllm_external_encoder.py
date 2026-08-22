# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for workflow-produced encoder results consumed by stock vLLM."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from vllm.inputs import EmbedsPrompt

from dynamo.common.external_encoder import ExternalEncoderResult
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
    *,
    row_splits: tuple[int, ...] = (0, 2, 3),
) -> dict:
    return ExternalEncoderResult(
        features={"opaque": "nixl-reference"},
        row_splits=row_splits,
        image_token_id=_IMAGE_TOKEN_ID,
    ).to_dict()


def _handler(mode: DisaggregationMode = DisaggregationMode.AGGREGATED):
    handler = object.__new__(DecodeWorkerHandler)
    handler.config = SimpleNamespace(
        disaggregation_mode=mode,
        engine_args=_engine_args(),
    )
    handler.model_config = _model_config()
    handler._external_encoder_prompt_loader = None
    handler._custom_encoder = None
    return handler


async def test_loader_imports_packed_features_and_builds_mixed_prompt():
    packed = torch.arange(12, dtype=torch.bfloat16).reshape(3, _HIDDEN)
    importer = SimpleNamespace(import_tensor=AsyncMock(return_value=packed))
    factory = MagicMock(return_value=importer)
    loader = ExternalEncoderPromptLoader(
        _model_config(),
        _engine_args(),
        importer_factory=factory,
    )

    prompt = await loader.load(
        _encoder_result(),
        [1, _IMAGE_TOKEN_ID, 2, _IMAGE_TOKEN_ID, 3],
    )

    assert prompt["prompt_token_ids"] == [1, 99, 99, 2, 99, 3]
    assert prompt["prompt_is_token_ids"] == [True, False, False, True, False, True]
    torch.testing.assert_close(prompt["prompt_embeds"][1:3], packed[:2])
    torch.testing.assert_close(prompt["prompt_embeds"][4], packed[2])
    factory.assert_called_once_with()
    importer.import_tensor.assert_awaited_once_with(_encoder_result()["features"])


async def test_loader_reuses_lazy_importer():
    packed = torch.ones((1, _HIDDEN), dtype=torch.bfloat16)
    importer = SimpleNamespace(import_tensor=AsyncMock(return_value=packed))
    factory = MagicMock(return_value=importer)
    loader = ExternalEncoderPromptLoader(
        _model_config(),
        _engine_args(),
        importer_factory=factory,
    )

    assert factory.call_count == 0
    result = _encoder_result(row_splits=(0, 1))
    await loader.load(result, [_IMAGE_TOKEN_ID])
    await loader.load(result, [_IMAGE_TOKEN_ID])

    factory.assert_called_once_with()
    assert importer.import_tensor.await_count == 2


@pytest.mark.parametrize(
    "model_config,engine_args,match",
    [
        (_model_config(multimodal=True), _engine_args(), "text-only decoder"),
        (_model_config(), _engine_args(enable_prompt_embeds=False), "prompt-embeds"),
    ],
)
def test_loader_rejects_incompatible_decoder_configuration(
    model_config, engine_args, match
):
    with pytest.raises(ValueError, match=match):
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
            "decoder dtype",
        ),
        (
            torch.ones((2, _HIDDEN), dtype=torch.bfloat16),
            (0, 3),
            "cover the imported tensor rows",
        ),
        (
            torch.ones((2, _HIDDEN), dtype=torch.bfloat16),
            (0, 0, 2),
            "0 rows",
        ),
    ],
)
async def test_loader_rejects_invalid_imported_tensor(packed, row_splits, match):
    importer = SimpleNamespace(import_tensor=AsyncMock(return_value=packed))
    loader = ExternalEncoderPromptLoader(
        _model_config(),
        _engine_args(),
        importer_factory=lambda: importer,
    )
    token_ids = [_IMAGE_TOKEN_ID] * (len(row_splits) - 1)

    with pytest.raises(ValueError, match=match):
        await loader.load(_encoder_result(row_splits=row_splits), token_ids)


async def test_handler_assembles_external_prompt_through_shared_loader():
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
        "encoder_result": _encoder_result(row_splits=(0, 1)),
    }

    prompt, error = await handler._assemble_external_encoder_prompt(request, "req-1")

    assert error is None
    assert prompt is expected
    loader.load.assert_awaited_once_with(
        request["encoder_result"],
        request["token_ids"],
    )


async def test_handler_rejects_competing_raw_media():
    handler = _handler()
    request = {
        "token_ids": [_IMAGE_TOKEN_ID],
        "encoder_result": _encoder_result(row_splits=(0, 1)),
        "multi_modal_data": {"image_url": [{"Url": "unused"}]},
    }

    prompt, error = await handler._assemble_external_encoder_prompt(request, "req-1")

    assert prompt is None
    assert error is not None
    assert "authoritative" in error["finish_reason"]


async def test_handler_rejects_external_result_on_disaggregated_vllm_worker():
    handler = _handler(DisaggregationMode.DECODE)

    chunks = [
        chunk
        async for chunk in handler._generate_token_mode(
            {"encoder_result": _encoder_result(row_splits=(0, 1))},
            MagicMock(),
            "req-1",
        )
    ]

    assert len(chunks) == 1
    assert "aggregated vLLM worker" in chunks[0]["finish_reason"]


async def test_aggregated_token_path_selects_external_prompt_assembly():
    handler = _handler()
    handler._assemble_external_encoder_prompt = AsyncMock(
        return_value=(
            None,
            {
                "finish_reason": "error: stop after selection",
                "index": 0,
                "token_ids": [],
            },
        )
    )
    request = {"encoder_result": _encoder_result(row_splits=(0, 1))}

    chunks = [
        chunk
        async for chunk in handler._generate_token_mode(
            request,
            MagicMock(),
            "req-1",
        )
    ]

    assert chunks[0]["finish_reason"] == "error: stop after selection"
    handler._assemble_external_encoder_prompt.assert_awaited_once_with(
        request,
        "req-1",
    )
