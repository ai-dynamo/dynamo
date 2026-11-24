# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for fetch_embeddings_from_encoder."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "Skipping to avoid errors during collection with '-m gpu_0'. "
        "CUDA/GPU not available, but tensorrt_llm import and the test require GPU.",
        allow_module_level=True,
    )
from tensorrt_llm.llmapi import DisaggregatedParams

from dynamo.common.memory.multimodal_embedding_cache_manager import (
    CachedEmbedding,
    MultimodalEmbeddingCacheManager,
)
from dynamo.trtllm.multimodal.embedding_fetcher import (
    fetch_embeddings_from_encoder,
    fetch_mm_items_from_encoder,
)
from dynamo.trtllm.multimodal.hasher import MultimodalHasher

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
]


def create_mock_encode_client(
    embeddings: list[torch.Tensor],
    processed_prompt: str = "prompt",
    prompt_token_ids: list[int] | None = None,
    grids: list[list[int]] | None = None,
) -> AsyncMock:
    """Create mock encode client that returns embeddings via CUDA IPC handles.

    With grids, each handle carries its image grid in its metadata.
    """

    def handle(i: int) -> dict[str, Any]:
        if grids is None:
            return {"name": f"h{i}"}
        return {"name": f"h{i}", "metadata": {"image_grid_thw": grids[i]}}

    class MockResponse:
        def data(self):
            return {
                "ep_disaggregated_params": {
                    "multimodal_embedding_handles": [
                        handle(i) for i in range(len(embeddings))
                    ]
                },
                "processed_prompt": processed_prompt,
                "prompt_token_ids": prompt_token_ids or [1, 2, 3],
            }

    async def mock_round_robin(req: dict[str, Any], context=None) -> Any:
        async def gen():
            yield MockResponse()

        return gen()

    client = AsyncMock()
    client.round_robin = mock_round_robin
    return client


@pytest.fixture
def encoder_cache() -> MultimodalEmbeddingCacheManager:
    """Create encoder cache with 10MB capacity."""
    return MultimodalEmbeddingCacheManager(capacity_bytes=10 * 1024 * 1024)


class TestFetchEmbeddingsFromEncoder:
    """Tests for fetch_embeddings_from_encoder function."""

    @pytest.mark.asyncio
    async def test_partial_cache_no_metadata_update(self, encoder_cache):
        """Cache path: request NOT updated with EPD metadata."""
        url1, url2 = "http://example.com/img1.jpg", "http://example.com/img2.jpg"
        embedding1, embedding2 = torch.ones(10, 256), torch.ones(10, 256) * 2

        encoder_cache.set(
            MultimodalHasher.hash_bytes(url1.encode()),
            CachedEmbedding(tensor=embedding1),
        )
        request: dict[str, Any] = {"messages": []}

        mock_client = create_mock_encode_client([embedding2])

        with patch(
            "dynamo.trtllm.multimodal.embedding_fetcher.extract_embeddings_from_handles",
            AsyncMock(return_value=[embedding2]),
        ):
            result = await fetch_embeddings_from_encoder(
                [url1, url2], request, mock_client, encoder_cache
            )

        assert len(result) == 2
        assert "_epd_processed_prompt" not in request

    @pytest.mark.asyncio
    async def test_all_cached_no_request_sent(self, encoder_cache):
        """All cached: no encode request sent."""
        url1, url2 = "http://example.com/img1.jpg", "http://example.com/img2.jpg"
        embedding1, embedding2 = torch.ones(10, 256), torch.ones(10, 256) * 2

        encoder_cache.set(
            MultimodalHasher.hash_bytes(url1.encode()),
            CachedEmbedding(tensor=embedding1),
        )
        encoder_cache.set(
            MultimodalHasher.hash_bytes(url2.encode()),
            CachedEmbedding(tensor=embedding2),
        )

        async def should_not_call(req: dict[str, Any]) -> None:
            raise AssertionError("Should not be called")

        mock_client = AsyncMock()
        mock_client.round_robin = should_not_call

        result = await fetch_embeddings_from_encoder(
            [url1, url2], {"messages": []}, mock_client, encoder_cache
        )

        assert len(result) == 2
        assert torch.equal(result[0], embedding1)

    @pytest.mark.asyncio
    async def test_cache_passes_each_image_grid(self, encoder_cache):
        """Each image keeps its own grid on a cache miss and on a later hit.

        A wide and a tall image can give the same number of embedding rows,
        so only the grid tells TRT-LLM how each image is laid out.
        """
        urls = ["http://example.com/wide.jpg", "http://example.com/tall.jpg"]
        embeddings = [torch.ones(2040, 8), torch.ones(2040, 8) * 2]
        grids = [[1, 68, 120], [1, 120, 68]]
        mock_client = create_mock_encode_client(embeddings, grids=grids)

        with patch(
            "dynamo.trtllm.multimodal.embedding_fetcher.extract_embeddings_from_handles",
            AsyncMock(return_value=embeddings),
        ) as extract:
            miss = await fetch_embeddings_from_encoder(
                urls, {"messages": []}, mock_client, encoder_cache
            )
            hit = await fetch_embeddings_from_encoder(
                urls[::-1], {"messages": []}, mock_client, encoder_cache
            )

        assert extract.await_count == 1
        for result, order in ((miss, [0, 1]), (hit, [1, 0])):
            assert [item["image_grid_thw"] for item in result] == [
                grids[i] for i in order
            ]
            for item, i in zip(result, order):
                assert torch.equal(item["mm_embeddings"], embeddings[i])
        for url, grid in zip(urls, grids):
            cached = encoder_cache.get(MultimodalHasher.hash_bytes(url.encode()))
            assert cached.image_grid_thw == grid

    @pytest.mark.asyncio
    async def test_cache_without_grid_keeps_bare_tensors(self, encoder_cache):
        """An encoder that gives no grid still gets bare tensors, as before."""
        url = "http://example.com/img.jpg"
        embedding = torch.ones(10, 256)
        mock_client = create_mock_encode_client([embedding])

        with patch(
            "dynamo.trtllm.multimodal.embedding_fetcher.extract_embeddings_from_handles",
            AsyncMock(return_value=[embedding]),
        ):
            miss = await fetch_embeddings_from_encoder(
                [url], {"messages": []}, mock_client, encoder_cache
            )
            hit = await fetch_embeddings_from_encoder(
                [url], {"messages": []}, mock_client, encoder_cache
            )

        for result in (miss, hit):
            assert len(result) == 1
            assert isinstance(result[0], torch.Tensor)
            assert torch.equal(result[0], embedding)
        cached = encoder_cache.get(MultimodalHasher.hash_bytes(url.encode()))
        assert cached.image_grid_thw is None

    @pytest.mark.asyncio
    async def test_session_scope_partitions_embedding_cache(
        self, monkeypatch, encoder_cache
    ):
        """The same URL in separate sessions must not share an embedding."""
        monkeypatch.setenv("DYN_MM_IMAGE_CACHE_SESSION_SCOPED", "1")
        url = "http://example.com/img.jpg"
        first_embedding = torch.ones(10, 256)
        second_embedding = torch.ones(10, 256) * 2
        mock_client = create_mock_encode_client([first_embedding])

        with patch(
            "dynamo.trtllm.multimodal.embedding_fetcher.extract_embeddings_from_handles",
            AsyncMock(side_effect=[[first_embedding], [second_embedding]]),
        ) as extract:
            first = await fetch_embeddings_from_encoder(
                [url],
                {"messages": [], "image_cache_scope": "session-a"},
                mock_client,
                encoder_cache,
            )
            second = await fetch_embeddings_from_encoder(
                [url],
                {"messages": [], "image_cache_scope": "session-b"},
                mock_client,
                encoder_cache,
            )
            first_again = await fetch_embeddings_from_encoder(
                [url],
                {"messages": [], "image_cache_scope": "session-a"},
                mock_client,
                encoder_cache,
            )

        assert extract.await_count == 2
        assert torch.equal(first[0], first_embedding)
        assert torch.equal(second[0], second_embedding)
        assert torch.equal(first_again[0], first_embedding)
        assert encoder_cache.stats["entries"] == 2

    @pytest.mark.asyncio
    async def test_session_scoped_embedding_cache_bypasses_without_scope(
        self, monkeypatch, encoder_cache
    ):
        """Missing scope must neither read nor populate the embedding cache."""
        monkeypatch.setenv("DYN_MM_IMAGE_CACHE_SESSION_SCOPED", "1")
        url = "http://example.com/img.jpg"
        embeddings = [torch.ones(10, 256), torch.ones(10, 256) * 2]
        mock_client = create_mock_encode_client([embeddings[0]])

        with patch(
            "dynamo.trtllm.multimodal.embedding_fetcher.extract_embeddings_from_handles",
            AsyncMock(side_effect=[[embeddings[0]], [embeddings[1]]]),
        ) as extract:
            first = await fetch_embeddings_from_encoder(
                [url], {"messages": []}, mock_client, encoder_cache
            )
            second = await fetch_embeddings_from_encoder(
                [url],
                {"messages": [], "image_cache_scope": " "},
                mock_client,
                encoder_cache,
            )

        assert extract.await_count == 2
        assert torch.equal(first[0], embeddings[0])
        assert torch.equal(second[0], embeddings[1])
        assert encoder_cache.stats["entries"] == 0

    @pytest.mark.asyncio
    async def test_no_cache_returns_disaggregated_params(self):
        """No cache: returns DisaggregatedParams directly, request updated with metadata."""
        request: dict[str, Any] = {"messages": []}

        # Pass one embedding so mock generates one handle (DisaggregatedParams requires non-empty handles)
        mock_client = create_mock_encode_client(
            [torch.ones(10, 256)],
            processed_prompt="test <image>",
            prompt_token_ids=[10, 20],
        )

        result = await fetch_embeddings_from_encoder(
            ["http://example.com/img.jpg"], request, mock_client, encoder_cache=None
        )

        assert isinstance(result, DisaggregatedParams)
        assert request["_epd_processed_prompt"] == "test <image>"
        assert request["_epd_prompt_token_ids"] == [10, 20]

    @pytest.mark.asyncio
    async def test_empty_urls_raises_error(self, encoder_cache):
        """Empty image_urls raises ValueError."""
        mock_client = AsyncMock()

        with pytest.raises(ValueError, match="image_urls must not be empty"):
            await fetch_embeddings_from_encoder([], {}, mock_client, encoder_cache)


class _ItemEncoder:
    """Encode worker stand-in for media items: one handle per item, with its
    metadata, and records each request it gets."""

    def __init__(self, metadata: dict[str, dict[str, Any]]):
        self.metadata = metadata
        self.requests: list[dict[str, Any]] = []

    async def round_robin(self, request: dict[str, Any], context=None) -> Any:
        self.requests.append(request)
        urls = [
            item["Url"]
            for items in request["multi_modal_data"].values()
            for item in items
        ]
        handles = [{"name": url, "metadata": dict(self.metadata[url])} for url in urls]

        async def gen():
            yield SimpleNamespace(
                data=lambda: {
                    "ep_disaggregated_params": {"multimodal_embedding_handles": handles}
                }
            )

        return gen()


def _extract_by_name(tensors: dict[str, torch.Tensor]) -> AsyncMock:
    return AsyncMock(side_effect=lambda handles: [tensors[h["name"]] for h in handles])


IMAGE, VIDEO = "http://example.com/a.jpg", "http://example.com/v.mp4"
METADATA = {
    IMAGE: {"image_grid_thw": [1, 4, 6], "multimodal_hash": [1] * 8},
    VIDEO: {
        "video_grid_thw": [2, 4, 4],
        "video_timestamps": [0.5, 1.5],
        "multimodal_hash": [2] * 8,
    },
}
TENSORS = {IMAGE: torch.ones(6, 8), VIDEO: torch.ones(8, 8) * 2}


class TestFetchMediaItems:
    """Tests for fetch_mm_items_from_encoder (TRT-LLM item handoff)."""

    @pytest.mark.asyncio
    async def test_cache_stores_and_returns_metadata_and_hash(self, encoder_cache):
        """Each item is cached with its grid, video timing and hash, under a
        key with its modality; a hit gives them back without the encoder."""
        encoder = _ItemEncoder(METADATA)
        media = {"image": [IMAGE], "video": [VIDEO]}

        with patch(
            "dynamo.trtllm.multimodal.embedding_fetcher.extract_embeddings_from_handles",
            _extract_by_name(TENSORS),
        ):
            miss = await fetch_mm_items_from_encoder(
                media, {"messages": []}, encoder, encoder_cache
            )
            hit = await fetch_mm_items_from_encoder(
                media, {"messages": []}, encoder, encoder_cache
            )

        assert len(encoder.requests) == 2  # the miss: one call per modality
        for result in (miss, hit):
            assert set(result) == {"image", "video"}
            for modality, url in (("image", IMAGE), ("video", VIDEO)):
                (item,) = result[modality]
                assert torch.equal(item.pop("mm_embeddings"), TENSORS[url])
                assert item == METADATA[url]
        # The same URL as a video is not the cached image.
        assert encoder_cache.stats["entries"] == 2
        with patch(
            "dynamo.trtllm.multimodal.embedding_fetcher.extract_embeddings_from_handles",
            _extract_by_name({IMAGE: TENSORS[VIDEO]}),
        ):
            encoder.metadata = {IMAGE: METADATA[VIDEO]}
            as_video = await fetch_mm_items_from_encoder(
                {"video": [IMAGE]}, {"messages": []}, encoder, encoder_cache
            )
        assert len(encoder.requests) == 3
        assert as_video["video"][0]["video_grid_thw"] == [2, 4, 4]

    @pytest.mark.asyncio
    async def test_item_count_mismatch_is_rejected(self, encoder_cache):
        """A response with fewer embeddings than items is not cached."""
        encoder = _ItemEncoder(METADATA)
        real_round_robin = encoder.round_robin

        async def drop_one(request, context=None):
            request["multi_modal_data"]["image_url"].pop()
            return await real_round_robin(request, context)

        encoder.round_robin = drop_one  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="returned 1 embeddings for 2 image"):
            await fetch_mm_items_from_encoder(
                {"image": [IMAGE, IMAGE]}, {"messages": []}, encoder, encoder_cache
            )
        assert encoder_cache.stats["entries"] == 0
