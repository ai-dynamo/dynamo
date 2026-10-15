# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for load_multimodal_embeddings in prefill_worker_utils."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch

from dynamo.common.memory.multimodal_embedding_cache_manager import (
    CachedEmbedding,
    MultimodalEmbeddingCacheManager,
)
from dynamo.vllm.multimodal_utils import prefill_worker_utils as mod
from dynamo.vllm.multimodal_utils.protocol import MultiModalGroup, MultiModalInput

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]

MODEL = "test-model"
DTYPE = torch.float16


def test_attach_coalesced_embedding_transfer_uses_group_shapes():
    combined = torch.arange(20, dtype=DTYPE).reshape(1, 5, 4)
    groups = [
        MultiModalGroup(embeddings_shape=(1, 2, 4)),
        MultiModalGroup(embeddings_shape=(1, 3, 4)),
    ]
    receiver = SimpleNamespace(release_tensor=Mock())
    pending = mod._PendingRelease(receiver)

    mod._attach_received_embedding_transfers(
        groups,
        transfer_group_indices=[0],
        loaded=[(7, combined)],
        pending=pending,
    )

    assert torch.equal(groups[0].loaded_embedding, combined[:, :2])
    assert torch.equal(groups[1].loaded_embedding, combined[:, 2:])
    assert groups[0].loaded_embedding.untyped_storage().data_ptr() == (
        combined.untyped_storage().data_ptr()
    )
    assert groups[1].loaded_embedding.untyped_storage().data_ptr() == (
        combined.untyped_storage().data_ptr()
    )

    pending.release_all()
    receiver.release_tensor.assert_called_once_with(7)


def test_attach_legacy_per_image_embedding_transfers():
    first = torch.randn(1, 2, 4, dtype=DTYPE)
    second = torch.randn(1, 3, 4, dtype=DTYPE)
    groups = [MultiModalGroup(), MultiModalGroup()]

    mod._attach_received_embedding_transfers(
        groups,
        transfer_group_indices=[0, 1],
        loaded=[(7, first), (8, second)],
        pending=None,
    )

    assert groups[0].loaded_embedding is first
    assert groups[1].loaded_embedding is second


def test_attach_coalesced_embedding_rejects_token_count_mismatch():
    groups = [
        MultiModalGroup(embeddings_shape=(1, 2, 4)),
        MultiModalGroup(embeddings_shape=(1, 2, 4)),
    ]

    with pytest.raises(RuntimeError, match="token count"):
        mod._attach_received_embedding_transfers(
            groups,
            transfer_group_indices=[0],
            loaded=[(7, torch.randn(1, 5, 4, dtype=DTYPE))],
            pending=None,
        )


@pytest.mark.asyncio
async def test_encode_worker_request_carries_image_cache_scope():
    """The prefill-to-encode hop must preserve frontend cache isolation."""
    captured_payloads = []

    class _Response:
        def __init__(self, payload: str):
            self._payload = payload

        def data(self) -> str:
            return self._payload

    async def _response_stream(payload: str):
        yield _Response(payload)

    async def _round_robin(payload: str, *, context=None):
        captured_payloads.append(payload)
        response = json.loads(payload)
        response["multimodal_inputs"][0]["serialized_request"] = {
            "embeddings_shape": [1, 2, 3],
            "embedding_dtype_str": "float16",
            "serialized_request": "opaque",
        }
        return _response_stream(json.dumps(response))

    client = Mock()
    client.instance_ids.return_value = ["encode-0"]
    client.round_robin = AsyncMock(side_effect=_round_robin)

    receiver = SimpleNamespace(
        receive_embeddings=AsyncMock(
            return_value=(7, torch.randn(1, 2, 3, dtype=DTYPE))
        ),
        release_tensor=Mock(),
    )
    groups, pending = await mod._fetch_from_encode_workers(
        client,
        ["https://example.com/image.png"],
        "req-1",
        receiver,
        cache_scope="session-42",
    )

    assert len(groups) == 1
    assert pending is not None
    assert json.loads(captured_payloads[0])["image_cache_scope"] == "session-42"
    pending.release_all()
    receiver.release_tensor.assert_called_once_with(7)


class TestMultimodalEmbeddingLoader:
    @pytest.mark.asyncio
    async def test_all_cached(self):
        """All URLs cached -> no encode worker call, returns accumulated mm_data."""
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        tensor = torch.randn(1, 10, dtype=DTYPE)
        grid = [[1, 2, 3]]
        url = "http://img1.png"
        key = mod.get_embedding_hash(url)
        cache.set(key, CachedEmbedding(tensor=tensor, image_grid_thw=grid))

        with patch.object(
            mod,
            "_fetch_from_encode_workers",
            new_callable=AsyncMock,
        ) as mock_fetch:
            embedding_loader = mod.MultiModalEmbeddingLoader(AsyncMock(), None, cache)
            mm_data = await embedding_loader.load_multimodal_embeddings(
                [url],
                "req-1",
                model=MODEL,
            )

        mock_fetch.assert_not_awaited()
        assert torch.equal(mm_data["image"], tensor)

    @pytest.mark.asyncio
    async def test_all_uncached_with_cache(self):
        """All URLs uncached with cache -> encode worker call, results cached."""
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        url = "http://img1.png"
        tensor = torch.randn(1, 10, dtype=DTYPE)
        fake_group = MultiModalGroup(
            multimodal_input=MultiModalInput(),
            image_grid_thw=[[1, 2, 3]],
            loaded_embedding=tensor,
        )

        with patch.object(
            mod,
            "_fetch_from_encode_workers",
            new_callable=AsyncMock,
            return_value=([fake_group], None),
        ) as mock_fetch:
            embedding_loader = mod.MultiModalEmbeddingLoader(AsyncMock(), None, cache)
            mm_data = await embedding_loader.load_multimodal_embeddings(
                [url],
                "req-1",
                model=MODEL,
                cache_scope="session-42",
            )

        mock_fetch.assert_awaited_once()
        assert mock_fetch.call_args.kwargs["cache_scope"] == "session-42"
        assert torch.equal(mm_data["image"], tensor)

        key = mod.get_embedding_hash(url)
        cached = cache.get(key)
        assert cached is not None
        assert torch.equal(cached.tensor, tensor)

    @pytest.mark.asyncio
    async def test_session_scope_partitions_embedding_cache(self):
        """The same URL in separate sessions must not share an embedding."""
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        url = "http://img1.png"
        first_tensor = torch.full((1, 10), 1.0, dtype=DTYPE)
        second_tensor = torch.full((1, 10), 2.0, dtype=DTYPE)
        groups = [
            MultiModalGroup(loaded_embedding=first_tensor),
            MultiModalGroup(loaded_embedding=second_tensor),
        ]

        with patch.object(
            mod,
            "_fetch_from_encode_workers",
            new_callable=AsyncMock,
            side_effect=[([groups[0]], None), ([groups[1]], None)],
        ) as mock_fetch:
            embedding_loader = mod.MultiModalEmbeddingLoader(
                AsyncMock(), None, cache, session_scoped_cache=True
            )
            first = await embedding_loader.load_multimodal_embeddings(
                [url], "req-1", model=MODEL, cache_scope="session-a"
            )
            second = await embedding_loader.load_multimodal_embeddings(
                [url], "req-2", model=MODEL, cache_scope="session-b"
            )
            first_again = await embedding_loader.load_multimodal_embeddings(
                [url], "req-3", model=MODEL, cache_scope="session-a"
            )

        assert mock_fetch.await_count == 2
        assert torch.equal(first["image"], first_tensor)
        assert torch.equal(second["image"], second_tensor)
        assert torch.equal(first_again["image"], first_tensor)
        assert cache.stats["entries"] == 2

    @pytest.mark.asyncio
    async def test_session_scoped_embedding_cache_bypasses_without_scope(self):
        """Missing scope must neither read nor populate the embedding cache."""
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        url = "http://img1.png"
        tensors = [
            torch.full((1, 10), 1.0, dtype=DTYPE),
            torch.full((1, 10), 2.0, dtype=DTYPE),
        ]
        groups = [MultiModalGroup(loaded_embedding=tensor) for tensor in tensors]

        with patch.object(
            mod,
            "_fetch_from_encode_workers",
            new_callable=AsyncMock,
            side_effect=[([groups[0]], None), ([groups[1]], None)],
        ) as mock_fetch:
            embedding_loader = mod.MultiModalEmbeddingLoader(
                AsyncMock(), None, cache, session_scoped_cache=True
            )
            first = await embedding_loader.load_multimodal_embeddings(
                [url], "req-1", model=MODEL
            )
            second = await embedding_loader.load_multimodal_embeddings(
                [url], "req-2", model=MODEL, cache_scope=" "
            )

        assert mock_fetch.await_count == 2
        assert torch.equal(first["image"], tensors[0])
        assert torch.equal(second["image"], tensors[1])
        assert cache.stats["entries"] == 0

    @pytest.mark.asyncio
    async def test_no_cache(self):
        """Without cache -> all URLs go to encode workers."""
        url = "http://img1.png"
        tensor = torch.randn(1, 10, dtype=DTYPE)
        fake_group = MultiModalGroup(
            multimodal_input=MultiModalInput(),
            loaded_embedding=tensor,
        )

        with patch.object(
            mod,
            "_fetch_from_encode_workers",
            new_callable=AsyncMock,
            return_value=([fake_group], None),
        ) as mock_fetch:
            embedding_loader = mod.MultiModalEmbeddingLoader(AsyncMock(), None, None)
            mm_data = await embedding_loader.load_multimodal_embeddings(
                [url],
                "req-1",
                model=MODEL,
            )

        mock_fetch.assert_awaited_once()
        assert torch.equal(mm_data["image"], tensor)

    @pytest.mark.asyncio
    async def test_decoded_item_cached_by_content_hash(self):
        """A frontend-decoded item reuses the canonical content hash as its
        cache key, so a second request skips the encode worker."""
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)
        content_hash = "0123456789abcdef"
        decoded_item = {"Decoded": {"shape": [4, 4, 3], "content_hash": content_hash}}
        tensor = torch.randn(1, 10, dtype=DTYPE)
        fake_group = MultiModalGroup(
            multimodal_input=MultiModalInput(),
            image_grid_thw=[[1, 2, 3]],
            loaded_embedding=tensor,
        )

        with patch.object(
            mod,
            "_fetch_from_encode_workers",
            new_callable=AsyncMock,
            return_value=([fake_group], None),
        ) as mock_fetch:
            embedding_loader = mod.MultiModalEmbeddingLoader(AsyncMock(), None, cache)
            mm_data = await embedding_loader.load_multimodal_embeddings(
                [decoded_item],
                "req-1",
                model=MODEL,
            )
            mm_data_again = await embedding_loader.load_multimodal_embeddings(
                [decoded_item],
                "req-2",
                model=MODEL,
            )

        mock_fetch.assert_awaited_once()
        assert mock_fetch.call_args[0][1] == [decoded_item]
        assert torch.equal(mm_data["image"], tensor)
        assert torch.equal(mm_data_again["image"], tensor)
        cached = cache.get(content_hash)
        assert cached is not None
        assert torch.equal(cached.tensor, tensor)

    def test_parse_image_item_variants(self):
        assert mod.parse_image_item("http://a.png") == ("http://a.png", None)
        assert mod.parse_image_item({"Url": "http://a.png"}) == (
            "http://a.png",
            None,
        )
        metadata = {"shape": [4, 4, 3], "content_hash": "0123456789abcdef"}
        assert mod.parse_image_item({"Decoded": metadata}) == (None, metadata)

        with pytest.raises(ValueError, match="Unsupported image item"):
            mod.parse_image_item({"Url": "http://a.png", "Decoded": metadata})
        with pytest.raises(ValueError, match="Unsupported image item"):
            mod.parse_image_item({"ignored": "value"})
        with pytest.raises(ValueError, match="Unsupported image item"):
            mod.parse_image_item(123)

    @pytest.mark.asyncio
    async def test_mixed_cache(self):
        """Mixed cache hits/misses -> only misses sent to encode workers."""
        cache = MultimodalEmbeddingCacheManager(capacity_bytes=1024 * 1024)

        url_cached = "http://cached.png"
        url_miss = "http://miss.png"
        cached_tensor = torch.randn(1, 10, dtype=DTYPE)
        miss_tensor = torch.randn(1, 10, dtype=DTYPE)

        key = mod.get_embedding_hash(url_cached)
        cache.set(key, CachedEmbedding(tensor=cached_tensor, image_grid_thw=None))

        fake_group = MultiModalGroup(
            multimodal_input=MultiModalInput(),
            image_grid_thw=None,
            loaded_embedding=miss_tensor,
        )

        with patch.object(
            mod,
            "_fetch_from_encode_workers",
            new_callable=AsyncMock,
            return_value=([fake_group], None),
        ) as mock_fetch:
            embedding_loader = mod.MultiModalEmbeddingLoader(AsyncMock(), None, cache)
            mm_data = await embedding_loader.load_multimodal_embeddings(
                [url_cached, url_miss],
                "req-1",
                model=MODEL,
            )

        mock_fetch.assert_awaited_once()
        call_args = mock_fetch.call_args
        assert call_args[0][1] == [url_miss]
        expected = torch.cat((cached_tensor, miss_tensor))
        assert torch.equal(mm_data["image"], expected)
