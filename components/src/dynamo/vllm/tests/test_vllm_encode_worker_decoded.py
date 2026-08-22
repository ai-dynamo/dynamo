# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for encode-worker multimodal helpers."""

import logging
from types import SimpleNamespace

import pytest
import torch

from dynamo.vllm.multimodal_handlers import encode_worker_handler
from dynamo.vllm.multimodal_handlers.encode_worker_handler import (
    EmbeddingItem,
    EncodeWorkerHandler,
)
from dynamo.vllm.multimodal_utils.embedding_cache import EmbeddingCache
from dynamo.vllm.multimodal_utils.protocol import MultiModalInput

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


def _handler(*, frontend_decoding: bool) -> EncodeWorkerHandler:
    handler = EncodeWorkerHandler.__new__(EncodeWorkerHandler)
    handler._enable_frontend_decoding = frontend_decoding
    handler._decoded_content_hash_warning_emitted = False
    handler.embedding_cache = EmbeddingCache()
    return handler


def _embedding_item(values: torch.Tensor) -> EmbeddingItem:
    return EmbeddingItem(key=None, image_grid_thw=[], embeddings=values)


def test_prepare_embedding_transfers_coalesces_uneven_images():
    first = torch.arange(8, dtype=torch.float16).reshape(1, 2, 4)
    second = torch.arange(8, 20, dtype=torch.float16).reshape(1, 3, 4)
    items = [_embedding_item(first), _embedding_item(second)]

    transfers, indices = encode_worker_handler._prepare_embedding_transfers(
        items, coalesce=True
    )

    assert indices == [0, None]
    assert len(transfers) == 1
    assert torch.equal(transfers[0], torch.cat((first, second), dim=1))

    split_transfers, split_indices = encode_worker_handler._prepare_embedding_transfers(
        items, coalesce=False
    )
    assert split_transfers[0] is first
    assert split_transfers[1] is second
    assert split_indices == [0, 1]


def test_prepare_embedding_transfers_reuses_combined_encoder_output():
    combined = torch.randn(1, 5, 4)
    items = [
        _embedding_item(combined[:, :2]),
        _embedding_item(combined[:, 2:]),
    ]

    transfers, indices = encode_worker_handler._prepare_embedding_transfers(
        items,
        coalesce=True,
        combined_embedding=combined,
    )

    assert len(transfers) == 1
    assert transfers[0] is combined
    assert indices == [0, None]


def test_split_encode_controls_qwen_transfer_coalescing(monkeypatch):
    model = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"

    monkeypatch.setattr(encode_worker_handler, "SPLIT_ENCODE", 0)
    assert encode_worker_handler._should_coalesce_embedding_transfers(model, 2)
    assert not encode_worker_handler._should_coalesce_embedding_transfers(model, 1)

    monkeypatch.setattr(encode_worker_handler, "SPLIT_ENCODE", 1)
    assert not encode_worker_handler._should_coalesce_embedding_transfers(model, 2)


def test_image_processor_receives_engine_mm_processor_kwargs(monkeypatch):
    expected = {"min_pixels": 65536, "max_pixels": 262144}
    sentinel = object()

    def mock_from_pretrained(model, **kwargs):
        assert model == "model"
        assert kwargs == {"trust_remote_code": True, **expected}
        return sentinel

    monkeypatch.setattr(
        encode_worker_handler.AutoImageProcessor,
        "from_pretrained",
        mock_from_pretrained,
    )
    engine_args = SimpleNamespace(
        model="model",
        trust_remote_code=True,
        mm_processor_kwargs=expected,
    )

    assert encode_worker_handler._load_image_processor(engine_args) is sentinel


def test_cache_key_for_url_image_is_unchanged():
    handler = _handler(frontend_decoding=False)
    group_input = MultiModalInput(image_url="https://example.com/a.png")

    assert handler._image_cache_key(group_input) == EmbeddingCache.generate_hash_key(
        "https://example.com/a.png"
    )


def test_cache_key_for_decoded_image_uses_content_hash():
    handler = _handler(frontend_decoding=True)
    group_input = MultiModalInput(
        image_decoded={"shape": [4, 4, 3], "content_hash": "0123456789abcdef"}
    )

    assert handler._image_cache_key(group_input) == "0123456789abcdef"


def test_decoded_image_without_hash_is_unkeyed_and_warns_once(caplog):
    handler = _handler(frontend_decoding=True)
    group_input = MultiModalInput(image_decoded={"shape": [4, 4, 3]})

    with caplog.at_level(logging.WARNING):
        assert handler._image_cache_key(group_input) is None
        assert handler._image_cache_key(group_input) is None

    assert caplog.text.count("missing or invalid canonical content_hash") == 1


def test_decoded_image_rejected_without_frontend_decoding():
    handler = _handler(frontend_decoding=False)
    group_input = MultiModalInput(
        image_decoded={"shape": [4, 4, 3], "content_hash": "0123456789abcdef"}
    )

    with pytest.raises(ValueError, match="not enabled on the encode worker"):
        handler._image_cache_key(group_input)


def test_empty_group_rejected():
    handler = _handler(frontend_decoding=True)

    with pytest.raises(ValueError, match="image_url or image_decoded"):
        handler._image_cache_key(MultiModalInput())
    with pytest.raises(ValueError, match="image_url or image_decoded"):
        handler._image_cache_key(None)


def test_group_with_url_and_decoded_image_rejected():
    handler = _handler(frontend_decoding=True)
    group_input = MultiModalInput(
        image_url="https://example.com/a.png",
        image_decoded={"content_hash": "0123456789abcdef"},
    )

    with pytest.raises(ValueError, match="Exactly one"):
        handler._image_cache_key(group_input)
