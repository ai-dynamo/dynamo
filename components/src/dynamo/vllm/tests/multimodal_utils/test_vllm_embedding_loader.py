# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for load_multimodal_embeddings in prefill_worker_utils."""

import asyncio
import gc
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch

from dynamo.common.memory.multimodal_embedding_cache_manager import (
    CachedEmbedding,
    MultimodalEmbeddingCacheManager,
)
from dynamo.common.multimodal.embedding_transfer import LocalEmbeddingReceiver
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
    receiver.release_tensor.assert_not_called()
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


def _embedding_transfer_client(transfer_request, group_shapes):
    async def _round_robin(payload, *, context=None):
        response = json.loads(payload)
        groups = response["multimodal_inputs"]
        groups[0]["serialized_request"] = transfer_request
        for group, shape in zip(groups, group_shapes, strict=True):
            group["embeddings_shape"] = shape
        encoded = json.dumps(response)

        async def _stream():
            yield SimpleNamespace(data=lambda: encoded)

        return _stream()

    client = Mock()
    client.instance_ids.return_value = ["encode-0"]
    client.round_robin = AsyncMock(side_effect=_round_robin)
    items = ["https://example.com/image.png"] * len(group_shapes)
    return client, items


def _local_transfer_fixture(tmp_path, *, coalesced=False, invalid_shape=False):
    from safetensors.torch import save_file

    expected = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    path = tmp_path / "embedding.safetensors"
    save_file({"ec_cache": expected}, path)
    receiver = LocalEmbeddingReceiver()
    group_shapes = (
        [[1, 2, 4], [1, 2 if invalid_shape else 1, 4]]
        if coalesced
        else [list(expected.shape)]
    )
    client, items = _embedding_transfer_client(
        {
            "embeddings_shape": list(expected.shape),
            "embedding_dtype_str": "float32",
            "serialized_request": str(path),
        },
        group_shapes,
    )
    return client, receiver, items, path, expected


@pytest.mark.asyncio
@pytest.mark.parametrize("coalesced", [False, True])
async def test_completed_local_transfers_release_files_and_preserve_tensors(
    tmp_path, coalesced
):
    client, receiver, items, path, expected = _local_transfer_fixture(
        tmp_path, coalesced=coalesced
    )
    groups, pending = await mod._fetch_from_encode_workers(
        client, items, "local-success", receiver
    )

    assert pending is None
    assert not path.exists()
    assert receiver.received_tensors == {}
    del receiver
    gc.collect()
    actual = torch.cat([group.loaded_embedding for group in groups], dim=1)
    torch.testing.assert_close(actual, expected)
    if coalesced:
        assert groups[0].loaded_embedding.untyped_storage().data_ptr() == (
            groups[1].loaded_embedding.untyped_storage().data_ptr()
        )


@pytest.mark.asyncio
async def test_local_transfer_attachment_error_releases_completed_file(tmp_path):
    client, receiver, items, path, _ = _local_transfer_fixture(
        tmp_path, coalesced=True, invalid_shape=True
    )
    with pytest.raises(RuntimeError, match="token count"):
        await mod._fetch_from_encode_workers(client, items, "local-invalid", receiver)

    assert not path.exists()
    assert receiver.received_tensors == {}


@pytest.fixture
def nixl_read_transfer(monkeypatch):
    from dynamo import nixl_connect
    from dynamo.common.multimodal.embedding_transfer import NixlReadEmbeddingReceiver

    # Exercise real receiver/descriptor/pool ownership with native calls mocked.
    monkeypatch.setattr(nixl_connect, "nixl_api", Mock())
    receiver = NixlReadEmbeddingReceiver(
        embedding_hidden_size=16, max_item_mm_token=8, max_items=1
    )
    descriptor = receiver.warmedup_descriptors.queue[0]
    expected = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    descriptor._data_ref[: expected.numel() * expected.element_size()].view(
        torch.float32
    ).copy_(expected.flatten())
    completion = AsyncMock()
    receiver.connector.begin_read = AsyncMock(
        return_value=SimpleNamespace(wait_for_completion=completion)
    )
    release = Mock(wraps=receiver.release_tensor)
    monkeypatch.setattr(receiver, "release_tensor", release)
    client, items = _embedding_transfer_client(
        {
            "embeddings_shape": list(expected.shape),
            "embedding_dtype_str": "float32",
            "serialized_request": {
                "descriptors": [{"ptr": 1, "size": 48, "device": "cpu"}],
                "operation_kind": 1,
                "notification_key": "test-completion",
                "nixl_metadata": "unused-by-mocked-native-read",
            },
        },
        [list(expected.shape)],
    )
    return client, receiver, items, expected, completion, release, descriptor


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["cache", "assembly", "ownership"])
async def test_completed_nixl_transfer_released_on_processing_error(
    nixl_read_transfer, monkeypatch, failure_stage
):
    client, receiver, items, _, _, release, _ = nixl_read_transfer
    cache = (
        MultimodalEmbeddingCacheManager(capacity_bytes=1024)
        if failure_stage == "cache"
        else None
    )
    loader = mod.MultiModalEmbeddingLoader(client, receiver, cache)
    model = "qwen2-vl-test" if failure_stage == "assembly" else MODEL
    if failure_stage == "assembly":
        expected_error = ValueError
        message = "No image grid"
    else:
        expected_error = RuntimeError
        message = "injected copy failure"
        monkeypatch.setattr(
            torch.Tensor, "clone", Mock(side_effect=RuntimeError(message))
        )

    with pytest.raises(expected_error, match=message):
        await loader.load_multimodal_embeddings(items, "nixl-error", model=model)

    release.assert_called_once_with(0)
    assert receiver.inuse_descriptors == {}
    assert receiver.warmedup_descriptors.qsize() == 1


@pytest.mark.asyncio
async def test_nixl_transfer_waits_for_completion_and_owns_returned_tensor(
    nixl_read_transfer,
):
    (
        client,
        receiver,
        items,
        expected,
        completion,
        release,
        descriptor,
    ) = nixl_read_transfer
    started = asyncio.Event()
    finish = asyncio.Event()

    async def _complete():
        started.set()
        await finish.wait()

    completion.side_effect = _complete
    loader = mod.MultiModalEmbeddingLoader(client, receiver)
    task = asyncio.create_task(
        loader.load_multimodal_embeddings(items, "nixl-success", model=MODEL)
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        release.assert_not_called()
        assert receiver.warmedup_descriptors.empty()
    finally:
        finish.set()
        output = await asyncio.wait_for(task, timeout=5)

    release.assert_called_once_with(0)
    assert receiver.inuse_descriptors == {}
    assert receiver.warmedup_descriptors.qsize() == 1
    descriptor._data_ref.zero_()
    torch.testing.assert_close(output["image"], expected)
