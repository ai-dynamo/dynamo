# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from dynamo.common.multimodal.embedding_transfer import (
    LocalEmbeddingReceiver,
    LocalEmbeddingSender,
    NixlWriteEmbeddingSender,
)
from dynamo.vllm.constants import EmbeddingTransferMode
from dynamo.vllm.handlers import DecodeWorkerHandler
from dynamo.vllm.multimodal_handlers import encode_worker_handler as encode_module
from dynamo.vllm.multimodal_utils.custom_encoder import (
    HandoffReplayGuard,
    Qwen3VLImageEncoding,
    VisionEncoderBackend,
    create_custom_encoder_adapter,
    receive_linear_visual_prompt,
    stage_linear_visual_prompt,
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


async def test_linear_visual_handoff_transfers_only_visual_rows_and_is_single_use():
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        get_hidden_size=lambda: 4,
        is_multimodal_model=False,
    )
    adapter = _adapter()
    compact = adapter.prepare_compact_prompt(
        [1, 99, 2],
        [torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)],
    )
    sender = LocalEmbeddingSender()
    receiver = LocalEmbeddingReceiver()
    guard = HandoffReplayGuard()
    handoff, transfer_future = await stage_linear_visual_prompt(
        compact,
        sender,
        transfer_mode="local",
        decoder_model="decoder",
        decoder_revision="revision",
        model_config=model_config,
    )

    assert handoff["visual_embeds"]["embeddings_shape"] == [2, 4]
    assert len(handoff["prompt_token_ids"]) == 4
    received = await receive_linear_visual_prompt(
        handoff,
        receiver,
        guard,
        expected_transfer_mode="local",
        expected_decoder_model="decoder",
        expected_decoder_revision="revision",
        model_config=model_config,
    )
    await transfer_future

    expected = adapter.prepare_prompt(
        [1, 99, 2],
        [torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)],
    )
    assert torch.equal(received["prompt_embeds"], expected["prompt_embeds"])
    assert received["prompt_token_ids"] == expected["prompt_token_ids"]
    assert received["prompt_is_token_ids"] == expected["prompt_is_token_ids"]
    with pytest.raises(ValueError, match="reused"):
        await receive_linear_visual_prompt(
            handoff,
            receiver,
            guard,
            expected_transfer_mode="local",
            expected_decoder_model="decoder",
            expected_decoder_revision="revision",
            model_config=model_config,
        )


async def test_linear_visual_handoff_rejects_decoder_mismatch_before_receive():
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        get_hidden_size=lambda: 4,
        is_multimodal_model=False,
    )
    compact = _adapter().prepare_compact_prompt(
        [99], [torch.ones((2, 4), dtype=torch.bfloat16)]
    )
    handoff, _ = await stage_linear_visual_prompt(
        compact,
        LocalEmbeddingSender(),
        transfer_mode="local",
        decoder_model="decoder-a",
        decoder_revision=None,
        model_config=model_config,
    )

    with pytest.raises(ValueError, match="decoder model"):
        await receive_linear_visual_prompt(
            handoff,
            LocalEmbeddingReceiver(),
            HandoffReplayGuard(),
            expected_transfer_mode="local",
            expected_decoder_model="decoder-b",
            expected_decoder_revision=None,
            model_config=model_config,
        )


def test_encode_worker_init_rolls_back_custom_encoder_on_sender_failure(monkeypatch):
    encoder = Mock()
    adapter = Mock()
    monkeypatch.setattr(
        encode_module,
        "load_custom_encoder",
        lambda *args, **kwargs: (encoder, adapter),
    )

    class FailingSender:
        def __init__(self):
            raise RuntimeError("sender init failed")

    monkeypatch.setitem(
        encode_module.EMBEDDING_SENDER_FACTORIES,
        EmbeddingTransferMode.NIXL_WRITE,
        FailingSender,
    )
    config = SimpleNamespace(
        custom_encoder_class="module.CustomEncoder",
        engine_args=SimpleNamespace(
            model="decoder",
            create_model_config=lambda: object(),
        ),
    )

    with pytest.raises(RuntimeError, match="sender init failed"):
        encode_module.EncodeWorkerHandler(
            config,
            EmbeddingTransferMode.NIXL_WRITE,
        )

    encoder.shutdown.assert_called_once_with()


async def test_encode_worker_cleanup_terminates_background_tasks_and_is_idempotent():
    class Agent:
        def __init__(self):
            self.sent = []
            self.deregistered = []

        def get_new_notifs(self):
            return {}

        def check_xfer_state(self, handle):
            assert handle == "handle"
            return "DONE"

        def send_notif(self, remote_agent_id, notif_msg):
            import msgspec

            self.sent.append((remote_agent_id, msgspec.msgpack.decode(notif_msg)))

        def deregister_memory(self, descriptor):
            self.deregistered.append(descriptor)

    handler = object.__new__(encode_module.EncodeWorkerHandler)
    handler._cleanup_complete = False
    handler._custom_encoder = Mock()
    handler._custom_encoder_adapter = Mock()
    sender = object.__new__(NixlWriteEmbeddingSender)
    sender.nixl_agent = Agent()
    sender.remote_agents = {"receiver": object()}
    source = torch.zeros(4, dtype=torch.float16)
    transfer_future = asyncio.get_running_loop().create_future()
    descriptor = object()
    sender.transfer_tracker = {41: (source, object(), transfer_future)}
    sender.transfer_created_at = {41: 0.0}
    sender.transfer_failures = {}
    sender.retired_transfer_ids = OrderedDict()
    sender.retired_transfer_limit = 4
    sender.pending_terminal_notifications = OrderedDict()
    sender.responder_lease_expirations = {}
    sender.pending_write_requests = OrderedDict()
    sender.pending_write_retry_after = {}
    sender.pending_write_retry_attempts = {}
    sender.inflight_transfers = {41: ["handle", 0.0, "receiver", 7]}
    sender.registered_descs = {
        (source.data_ptr(), source.get_device()): [descriptor, 1]
    }
    sender.transfer_queue = asyncio.Queue()
    sender.transfer_timeout = 60
    sender._closing = False
    sender._state_update_task = asyncio.create_task(sender._state_update())
    handler.embedding_sender = sender
    handler.send_complete_queue = asyncio.Queue()
    handler.send_complete_checker_task = asyncio.create_task(
        handler.check_complete(handler.send_complete_queue)
    )
    handler.send_complete_queue.put_nowait((transfer_future, source))
    encoder = handler._custom_encoder
    checker_task = handler.send_complete_checker_task
    sender_task = sender._state_update_task

    await handler.cleanup()
    await handler.cleanup()

    assert checker_task.done()
    assert sender_task.done()
    assert sender._state_update_task is None
    assert transfer_future.done()
    assert isinstance(transfer_future.exception(), RuntimeError)
    assert sender.nixl_agent.sent == [("receiver", ["terminal", 7, "DONE"])]
    assert sender.nixl_agent.deregistered == [descriptor]
    assert not sender.transfer_tracker
    assert not sender.registered_descs
    encoder.shutdown.assert_called_once_with()
