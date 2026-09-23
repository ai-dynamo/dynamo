# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for external encoder results consumed by stock vLLM."""

from collections.abc import Mapping
from threading import get_ident
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from vllm.inputs import EmbedsPrompt

from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm.constants import DisaggregationMode
from dynamo.vllm.handlers import BYPASS_REMOTE_PREFILL_ANNOTATION, DecodeWorkerHandler
from dynamo.vllm.multimodal_utils.custom_encoder import handoff as handoff_module
from dynamo.vllm.multimodal_utils.custom_encoder.adapter.linear import (
    LinearEmbedsAdapter,
)
from dynamo.vllm.multimodal_utils.custom_encoder.backend import VisionEncoderBackend
from dynamo.vllm.multimodal_utils.custom_encoder.handoff import (
    ExternalEncoderResult,
    encode_request_plane_tensor,
)
from dynamo.vllm.multimodal_utils.custom_encoder.handoff_consumer import (
    ExternalEncoderHandoffConsumer,
)

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
    handler._external_encoder_handoff_consumer = None
    handler._custom_encoder = None
    return handler


def test_consumer_builds_mixed_prompt_from_request_plane_features() -> None:
    packed = torch.arange(12, dtype=torch.bfloat16).reshape(3, _HIDDEN)
    consumer = ExternalEncoderHandoffConsumer(_model_config(), _engine_args())

    prompt = consumer.prepare_prompt(
        _encoder_result(packed),
        [1, _IMAGE_TOKEN_ID, 2, _IMAGE_TOKEN_ID, 3],
    )

    assert prompt["prompt_token_ids"] == [1, 99, 99, 2, 99, 3]
    assert prompt["prompt_is_token_ids"] == [True, False, False, True, False, True]
    torch.testing.assert_close(prompt["prompt_embeds"][1:3], packed[:2])
    torch.testing.assert_close(prompt["prompt_embeds"][4], packed[2])


class _AdapterBackend(VisionEncoderBackend):
    image_token_id = _IMAGE_TOKEN_ID

    def build(self, model_id: str) -> None:
        pass

    def forward_batch(self, items, target_bucket=None):
        raise NotImplementedError


def test_handoff_consumer_matches_inline_linear_adapter() -> None:
    artifacts = [
        torch.arange(8, dtype=torch.bfloat16).reshape(2, _HIDDEN),
        torch.full((1, _HIDDEN), 12, dtype=torch.bfloat16),
    ]
    token_ids = [1, _IMAGE_TOKEN_ID, 2, _IMAGE_TOKEN_ID, 3]
    adapter = LinearEmbedsAdapter(
        _AdapterBackend(),
        _model_config(),
        _engine_args(),
    )
    consumer = ExternalEncoderHandoffConsumer(_model_config(), _engine_args())

    inline_prompt = adapter.prepare_prompt(token_ids, artifacts)
    remote_prompt = consumer.prepare_prompt(
        ExternalEncoderResult.from_artifacts(
            artifacts,
            image_token_id=_IMAGE_TOKEN_ID,
        ).to_dict(),
        token_ids,
    )

    assert remote_prompt["prompt_token_ids"] == inline_prompt["prompt_token_ids"]
    assert remote_prompt["prompt_is_token_ids"] == inline_prompt["prompt_is_token_ids"]
    torch.testing.assert_close(
        remote_prompt["prompt_embeds"], inline_prompt["prompt_embeds"]
    )


async def test_handler_reconstructs_prompt_off_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_loop_thread = get_ident()
    decode_thread: int | None = None
    original_decode = handoff_module.decode_request_plane_tensor

    def tracked_decode(payload: Mapping[str, Any]) -> torch.Tensor:
        nonlocal decode_thread
        decode_thread = get_ident()
        return original_decode(payload)

    monkeypatch.setattr(
        handoff_module,
        "decode_request_plane_tensor",
        tracked_decode,
    )
    handler = _handler()
    handler._external_encoder_handoff_consumer = ExternalEncoderHandoffConsumer(
        _model_config(),
        _engine_args(),
    )

    await handler._assemble_external_encoder_prompt(
        {
            "encoder_result": _encoder_result(row_splits=(0, 3)),
            "token_ids": [_IMAGE_TOKEN_ID],
        },
        "req-1",
    )

    assert decode_thread is not None
    assert decode_thread != event_loop_thread


@pytest.mark.parametrize(
    "model_config,engine_args,match",
    [
        (_model_config(multimodal=True), _engine_args(), "text-only decoder"),
        (_model_config(), _engine_args(enable_prompt_embeds=False), "prompt-embeds"),
    ],
)
def test_consumer_rejects_incompatible_decoder_configuration(
    model_config: SimpleNamespace,
    engine_args: SimpleNamespace,
    match: str,
) -> None:
    with pytest.raises(RuntimeError, match=match):
        ExternalEncoderHandoffConsumer(model_config, engine_args)


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
    ],
)
def test_consumer_rejects_invalid_tensor(
    packed: torch.Tensor,
    row_splits: tuple[int, ...],
    match: str,
) -> None:
    consumer = ExternalEncoderHandoffConsumer(_model_config(), _engine_args())
    token_ids = [_IMAGE_TOKEN_ID] * (len(row_splits) - 1)

    with pytest.raises(InvalidArgument, match=match):
        consumer.prepare_prompt(
            _encoder_result(packed, row_splits=row_splits),
            token_ids,
        )


def test_consumer_rejects_empty_image_row_split() -> None:
    packed = torch.ones((2, _HIDDEN), dtype=torch.bfloat16)
    encoder_result = _encoder_result(packed, row_splits=(0, 2))
    encoder_result["row_splits"] = [0, 0, 2]
    consumer = ExternalEncoderHandoffConsumer(_model_config(), _engine_args())

    with pytest.raises(InvalidArgument, match="strictly increasing"):
        consumer.prepare_prompt(
            encoder_result,
            [_IMAGE_TOKEN_ID, _IMAGE_TOKEN_ID],
        )


def test_consumer_rejects_placeholder_count_mismatch() -> None:
    consumer = ExternalEncoderHandoffConsumer(_model_config(), _engine_args())

    with pytest.raises(InvalidArgument, match="placeholder tokens.*image tensors"):
        consumer.prepare_prompt(
            _encoder_result(),
            [_IMAGE_TOKEN_ID],
        )


async def test_handler_assembles_external_prompt_through_handoff_consumer() -> None:
    handler = _handler()
    expected = EmbedsPrompt(
        prompt_embeds=torch.ones((1, _HIDDEN), dtype=torch.bfloat16),
        prompt_token_ids=[_IMAGE_TOKEN_ID],
        prompt_is_token_ids=[False],
    )
    consumer = SimpleNamespace(prepare_prompt=MagicMock(return_value=expected))
    handler._external_encoder_handoff_consumer = consumer
    request = {
        "token_ids": [_IMAGE_TOKEN_ID],
        "encoder_result": _encoder_result(row_splits=(0, 3)),
        "extra_args": {"nvext": {"extra_fields": ["engine_data"]}},
    }

    prompt = await handler._assemble_external_encoder_prompt(request, "req-1")

    assert prompt is expected
    consumer.prepare_prompt.assert_called_once_with(
        request["encoder_result"],
        request["token_ids"],
    )


async def test_handler_rejects_external_result_in_text_mode() -> None:
    handler = _handler()
    handler.use_vllm_tokenizer = True
    handler._first_token_source = None
    handler._multimodal_request_processor = SimpleNamespace(
        validate_multimodal_request=MagicMock()
    )
    handler._generate_text_mode = MagicMock(
        side_effect=AssertionError("text generator must not run")
    )
    handler._generate_token_mode = MagicMock(
        side_effect=AssertionError("token generator must not run")
    )
    context = MagicMock()
    context.id.return_value = "req-1"

    chunks = [
        chunk
        async for chunk in handler.generate(
            {"encoder_result": _encoder_result(row_splits=(0, 3))},
            context,
        )
    ]

    assert chunks == [
        {
            "finish_reason": (
                "error: external encoder results require token-in/token-out mode"
            ),
            "index": 0,
            "token_ids": [],
        }
    ]
    handler._generate_text_mode.assert_not_called()
    handler._generate_token_mode.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("multi_modal_data", {"image_url": [{"Url": "unused"}]}),
        ("multi_modal_uuids", {"image": ["unused"]}),
        ("prompt_embeds", "unused"),
        ("mm_processor_kwargs", {}),
        ("mm_routing_info", {}),
        ("media_io_kwargs", {}),
    ],
)
async def test_handler_rejects_competing_top_level_multimodal_fields(
    field: str,
    value: object,
) -> None:
    handler = _handler()
    request = {
        "token_ids": [_IMAGE_TOKEN_ID],
        "encoder_result": _encoder_result(row_splits=(0, 3)),
        field: value,
    }

    with pytest.raises(InvalidArgument, match=field):
        await handler._assemble_external_encoder_prompt(request, "req-1")


@pytest.mark.parametrize(
    "field",
    [
        "mm_processor_kwargs",
        "mm_kwargs_shm",
        "mm_kwargs_nixl",
        "mm_hashes",
        "mm_hashes_by_modality",
        "mm_placeholders",
        "mm_placeholders_by_modality",
        "expanded_token_ids",
    ],
)
async def test_handler_rejects_competing_multimodal_extra_args(field: str) -> None:
    handler = _handler()
    request = {
        "token_ids": [_IMAGE_TOKEN_ID],
        "encoder_result": _encoder_result(row_splits=(0, 3)),
        "extra_args": {field: {}},
    }

    with pytest.raises(InvalidArgument, match=rf"extra_args\.{field}"):
        await handler._assemble_external_encoder_prompt(request, "req-1")


async def test_handler_propagates_external_encoder_runtime_failure() -> None:
    handler = _handler()
    handler._external_encoder_handoff_consumer = SimpleNamespace(
        prepare_prompt=MagicMock(side_effect=RuntimeError("allocation failed"))
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


async def test_decode_bypass_selects_external_prompt_assembly() -> None:
    handler = _handler(DisaggregationMode.DECODE)
    handler._assemble_external_encoder_prompt = AsyncMock(
        side_effect=InvalidArgument("stop after selection")
    )
    request = {
        "encoder_result": _encoder_result(row_splits=(0, 3)),
        "annotations": [BYPASS_REMOTE_PREFILL_ANNOTATION],
    }

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


async def test_aggregated_token_path_selects_external_prompt_assembly() -> None:
    handler = _handler()
    handler._assemble_external_encoder_prompt = AsyncMock(
        side_effect=InvalidArgument("stop after selection")
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
