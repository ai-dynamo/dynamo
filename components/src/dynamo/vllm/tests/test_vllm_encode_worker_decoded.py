# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for frontend-decoded image handling in the encode worker."""

import logging

import pytest

from dynamo.vllm.multimodal_handlers.encode_worker_handler import EncodeWorkerHandler
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
