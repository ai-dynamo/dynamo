# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for fetch_embeddings_from_encoder."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import torch
from tensorrt_llm.llmapi import DisaggregatedParams

from dynamo.common.memory.encoder_cache_manager import EncoderCacheManager
from dynamo.trtllm.multimodal.embedding_fetcher import fetch_embeddings_from_encoder
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
) -> AsyncMock:
    """Create mock encode client that returns embeddings via CUDA IPC handles."""

    class MockResponse:
        def data(self):
            return {
                "ep_disaggregated_params": {
                    "multimodal_embedding_handles": [
                        f"h{i}" for i in range(len(embeddings))
                    ]
                },
                "processed_prompt": processed_prompt,
                "prompt_token_ids": prompt_token_ids or [1, 2, 3],
            }

    async def mock_round_robin(req: dict[str, Any]) -> Any:
        async def gen():
            yield MockResponse()

        return gen()

    client = AsyncMock()
    client.round_robin = mock_round_robin
    return client


@pytest.fixture
def encoder_cache() -> EncoderCacheManager:
    """Create encoder cache with 10MB capacity."""
    return EncoderCacheManager(capacity_bytes=10 * 1024 * 1024)


class TestFetchEmbeddingsFromEncoder:
    """Tests for fetch_embeddings_from_encoder function."""

    @pytest.mark.asyncio
    async def test_partial_cache_no_metadata_update(self, encoder_cache):
        """Cache path: request NOT updated with EPD metadata."""
        url1, url2 = "http://example.com/img1.jpg", "http://example.com/img2.jpg"
        embedding1, embedding2 = torch.ones(10, 256), torch.ones(10, 256) * 2

        encoder_cache.set(MultimodalHasher.hash_bytes(url1.encode()), embedding1)
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

        encoder_cache.set(MultimodalHasher.hash_bytes(url1.encode()), embedding1)
        encoder_cache.set(MultimodalHasher.hash_bytes(url2.encode()), embedding2)

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
