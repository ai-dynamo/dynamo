# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from dynamo.common.multimodal.embedding_transfer import (
    LocalEmbeddingReceiver,
    LocalEmbeddingSender,
)
from dynamo.vllm.handlers import DecodeWorkerHandler
from dynamo.vllm.multimodal_utils.custom_encoder import (
    HandoffReplayGuard,
    VisionEncoderBackend,
    create_custom_encoder_adapter,
    receive_linear_embeds_prompt,
    stage_linear_embeds_prompt,
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


async def test_linear_embeds_handoff_round_trip_is_owned_and_single_use():
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        get_hidden_size=lambda: 4,
        is_multimodal_model=False,
    )
    prepared = _adapter().prepare_prompt(
        [1, 99, 2],
        [torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)],
    )
    sender = LocalEmbeddingSender()
    receiver = LocalEmbeddingReceiver()
    guard = HandoffReplayGuard()
    handoff, transfer_future = await stage_linear_embeds_prompt(
        prepared,
        sender,
        transfer_mode="local",
        decoder_model="decoder",
        decoder_revision="revision",
        image_token_id=99,
        model_config=model_config,
    )

    received = await receive_linear_embeds_prompt(
        handoff,
        receiver,
        guard,
        expected_transfer_mode="local",
        expected_decoder_model="decoder",
        expected_decoder_revision="revision",
        model_config=model_config,
    )
    await transfer_future

    assert torch.equal(received["prompt_embeds"], prepared["prompt_embeds"])
    assert received["prompt_token_ids"] == prepared["prompt_token_ids"]
    assert received["prompt_is_token_ids"] == prepared["prompt_is_token_ids"]
    with pytest.raises(ValueError, match="reused"):
        await receive_linear_embeds_prompt(
            handoff,
            receiver,
            guard,
            expected_transfer_mode="local",
            expected_decoder_model="decoder",
            expected_decoder_revision="revision",
            model_config=model_config,
        )


async def test_linear_embeds_handoff_rejects_decoder_mismatch_before_receive():
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        get_hidden_size=lambda: 4,
        is_multimodal_model=False,
    )
    prepared = _adapter().prepare_prompt(
        [99], [torch.ones((2, 4), dtype=torch.bfloat16)]
    )
    sender = LocalEmbeddingSender()
    receiver = LocalEmbeddingReceiver()
    handoff, _ = await stage_linear_embeds_prompt(
        prepared,
        sender,
        transfer_mode="local",
        decoder_model="decoder-a",
        decoder_revision=None,
        image_token_id=99,
        model_config=model_config,
    )

    with pytest.raises(ValueError, match="decoder model"):
        await receive_linear_embeds_prompt(
            handoff,
            receiver,
            HandoffReplayGuard(),
            expected_transfer_mode="local",
            expected_decoder_model="decoder-b",
            expected_decoder_revision=None,
            model_config=model_config,
        )


async def test_worker_route_sends_complete_request_once_and_consumes_terminal():
    class _Response:
        def data(self):
            return {
                "finish_reason": "stop",
                "token_ids": [],
                "encoder_result": {"schema_version": 1},
            }

    async def _stream():
        yield _Response()

    client = SimpleNamespace(round_robin=AsyncMock(return_value=_stream()))
    handler = object.__new__(DecodeWorkerHandler)
    handler.encode_worker_client = client
    handler._consume_custom_encoder_handoff = AsyncMock(return_value="prompt")
    request = {
        "token_ids": [1, 99, 2],
        "multi_modal_data": {"image_url": [{"Url": "data:image/png;base64,unique"}]},
    }
    context = Mock()

    prompt = await handler._request_custom_encoder_handoff(request, context)

    assert prompt == "prompt"
    client.round_robin.assert_awaited_once_with(request, context=context)
    handler._consume_custom_encoder_handoff.assert_awaited_once_with(
        {"schema_version": 1}
    )
