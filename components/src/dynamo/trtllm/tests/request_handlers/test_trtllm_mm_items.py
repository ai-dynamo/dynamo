# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E/P with a separate encode worker: what the encode worker gets and what
reaches TRT-LLM, with and without TRT-LLM's media item handoff."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "Skipping to avoid errors during collection with '-m gpu_0'. "
        "CUDA/GPU not available, but tensorrt_llm import and the test require GPU.",
        allow_module_level=True,
    )

from dynamo.common.memory.multimodal_embedding_cache_manager import (
    MultimodalEmbeddingCacheManager,
)
from dynamo.trtllm.multimodal_processor import MultimodalRequestProcessor
from dynamo.trtllm.request_handlers.aggregated_handler import AggregatedHandler
from dynamo.trtllm.request_handlers.handlers import PrefillHandler
from dynamo.trtllm.tests.request_handlers.utils import create_mock_context
from dynamo.trtllm.tests.utils import create_mock_request_handler_config

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
]

A, B = "http://example.com/a.jpg", "http://example.com/b.jpg"
VIDEO = "http://example.com/v.mp4"
# Encoder item metadata: a wide and a tall image, and a video with its timing.
METADATA = {
    A: {"image_grid_thw": [1, 4, 6], "multimodal_hash": [1] * 8},
    B: {"image_grid_thw": [1, 6, 4], "multimodal_hash": [2] * 8},
    VIDEO: {
        "video_grid_thw": [2, 4, 4],
        "video_timestamps": [0.5, 1.5],
        "multimodal_hash": [3] * 8,
    },
}
TENSORS = {A: torch.ones(6, 8), B: torch.ones(6, 8) * 2, VIDEO: torch.ones(8, 8) * 3}
PROMPT = "<image><image><video> Describe them."


class _Encoder:
    """Encode worker stand-in: one handle per media item in the request, with
    that item's metadata. Records each request it gets."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def round_robin(self, request: dict[str, Any], context=None) -> Any:
        self.requests.append(request)
        urls = [
            item["Url"]
            for items in request["multi_modal_data"].values()
            for item in items
        ]
        handles = [{"name": url, "metadata": dict(METADATA[url])} for url in urls]

        async def gen():
            yield SimpleNamespace(
                data=lambda: {
                    "ep_disaggregated_params": {
                        "multimodal_embedding_handles": handles
                    },
                    "processed_prompt": PROMPT,
                    "prompt_token_ids": [1, 2, 3],
                }
            )

        return gen()


def _request(images: list[str], video: bool = False) -> dict[str, Any]:
    """A request as the frontend forwards it."""
    mm_data: dict[str, Any] = {"image_url": [{"Url": url} for url in images]}
    if video:
        mm_data["video_url"] = [{"Url": VIDEO}]
    return {
        "token_ids": [1, 2, 3],
        "multi_modal_data": mm_data,
        "extra_args": {"formatted_prompt": PROMPT, "messages": []},
    }


def _handler(
    handler_type: type, mode: str, encoder: _Encoder, cache: bool, items: bool
):
    config = create_mock_request_handler_config(disaggregation_mode=mode)
    config.encode_client = encoder
    processor = MultimodalRequestProcessor(
        model_type="multimodal",
        model_dir="unused",
        max_file_size_mb=10,
        tokenizer=MagicMock(),
    )
    # As set from the TRT-LLM input processor's supports_mm_item_handoff.
    processor.supports_mm_item_handoff = items
    config.multimodal_processor = processor
    encoder_cache = MultimodalEmbeddingCacheManager(10 * 1024**2) if cache else None
    return handler_type(config, encoder_cache=encoder_cache)


async def _trtllm_inputs(handler, request: dict[str, Any]) -> Any:
    """Run the handler; return the inputs it builds for TRT-LLM."""
    seen = {}

    async def generate_locally(request, context, embeddings, ep_params):
        seen["inputs"] = await handler.multimodal_processor.process_openai_request(
            request, embeddings, ep_params
        )
        yield {}

    with patch(
        "dynamo.trtllm.multimodal.embedding_fetcher.extract_embeddings_from_handles",
        AsyncMock(side_effect=lambda handles: [TENSORS[h["name"]] for h in handles]),
    ), patch.object(handler, "generate_locally", generate_locally):
        async for _ in handler.generate(request, create_mock_context()):
            pass
    return seen["inputs"]


def _items(*urls: str) -> list[dict[str, Any]]:
    return [{"mm_embeddings": TENSORS[url], **METADATA[url]} for url in urls]


def _assert_items(actual: list[dict[str, Any]], expected: list[dict[str, Any]]):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert torch.equal(got["mm_embeddings"], want["mm_embeddings"])
        assert {k: v for k, v in got.items() if k != "mm_embeddings"} == {
            k: v for k, v in want.items() if k != "mm_embeddings"
        }


HANDLERS = [(PrefillHandler, "prefill"), (AggregatedHandler, "prefill_and_decode")]


@pytest.mark.asyncio
@pytest.mark.parametrize("cache", [True, False], ids=["cache", "no-cache"])
@pytest.mark.parametrize(("handler_type", "mode"), HANDLERS)
async def test_image_and_video_reach_trtllm(handler_type, mode, cache):
    """Images and the video both go to the encode worker, one call per
    modality and with no prompt, and both reach TRT-LLM with the prompt."""
    encoder = _Encoder()
    handler = _handler(handler_type, mode, encoder, cache=cache, items=True)

    inputs = await _trtllm_inputs(handler, _request([A, B], video=True))

    assert [r["multi_modal_data"] for r in encoder.requests] == [
        {"image_url": [{"Url": A}, {"Url": B}]},
        {"video_url": [{"Url": VIDEO}]},
    ]
    assert all("token_ids" not in r for r in encoder.requests)
    assert inputs["prompt"] == PROMPT
    embeddings = inputs["multi_modal_embeddings"]
    assert set(embeddings) == {"image", "video"}
    _assert_items(embeddings["image"], _items(A, B))
    _assert_items(embeddings["video"], _items(VIDEO))


@pytest.mark.asyncio
async def test_partial_hit_encodes_only_missed_items_in_prompt_order():
    """With A cached, a request for [B, A] sends only B to the encode worker,
    and TRT-LLM gets B then A, each with its own grid and hash."""
    encoder = _Encoder()
    handler = _handler(PrefillHandler, "prefill", encoder, cache=True, items=True)
    await _trtllm_inputs(handler, _request([A]))

    inputs = await _trtllm_inputs(handler, _request([B, A]))

    assert len(encoder.requests) == 2
    assert encoder.requests[1] == {"multi_modal_data": {"image_url": [{"Url": B}]}}
    assert set(inputs["multi_modal_embeddings"]) == {"image"}
    _assert_items(inputs["multi_modal_embeddings"]["image"], _items(B, A))


@pytest.mark.asyncio
@pytest.mark.parametrize(("handler_type", "mode"), HANDLERS)
async def test_without_item_handoff_the_full_prompt_path_is_kept(handler_type, mode):
    """A TRT-LLM build without the item handoff keeps the full-prompt encode:
    the encode worker gets the whole request with its token ids, as before."""
    encoder = _Encoder()
    handler = _handler(handler_type, mode, encoder, cache=False, items=False)
    request = _request([A], video=True)

    inputs = await _trtllm_inputs(handler, request)

    (sent,) = encoder.requests
    assert sent["token_ids"] == [1, 2, 3]
    assert sent["multi_modal_data"] == request["multi_modal_data"]
    assert inputs == {"prompt": PROMPT, "prompt_token_ids": [1, 2, 3]}
