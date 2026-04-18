# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens

from dynamo.sglang._compat import (
    _build_grouped_mm_token_regex,
    maybe_enable_grouped_video_token_regex,
)
from dynamo.sglang.protocol import (
    MultiModalGroup,
    MultiModalInput,
    PreprocessedRequest,
    SamplingOptions,
    SglangMultimodalRequest,
    StopConditions,
)
from dynamo.sglang.request_handlers.multimodal.encode_worker_handler import (
    MultimodalEncodeWorkerHandler,
)
from dynamo.sglang.request_handlers.multimodal.worker_handler import (
    EmbeddingsProcessor,
    _build_mm_items,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def test_extract_multimodal_inputs_supports_video_urls():
    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)

    modality, urls = handler._extract_multimodal_inputs(
        {
            "multi_modal_data": {
                "video_url": [
                    {"Url": "https://example.com/clip.mp4"},
                    "file:///tmp/local.mp4",
                ]
            }
        }
    )

    assert modality == Modality.VIDEO
    assert urls == ["https://example.com/clip.mp4", "file:///tmp/local.mp4"]


def test_extract_multimodal_inputs_rejects_mixed_image_and_video():
    handler = MultimodalEncodeWorkerHandler.__new__(MultimodalEncodeWorkerHandler)

    with pytest.raises(ValueError, match="Mixed image_url and video_url"):
        handler._extract_multimodal_inputs(
            {
                "multi_modal_data": {
                    "image_url": [{"Url": "https://example.com/image.png"}],
                    "video_url": [{"Url": "https://example.com/clip.mp4"}],
                }
            }
        )


def test_expand_placeholder_tokens_supports_video():
    expanded = MultimodalEncodeWorkerHandler._expand_placeholder_tokens(
        [101, 151656, 102],
        151656,
        [3],
        "video",
    )

    assert expanded == [101, 151656, 151656, 151656, 102]


def test_grouped_video_regex_upgrade_uses_wrapped_image_structure():
    class _Tokenizer:
        def convert_ids_to_tokens(self, token_ids):
            mapping = {
                151655: "<|image_pad|>",
                151656: "<|video_pad|>",
            }
            return [mapping[token_id] for token_id in token_ids]

    tokenizer = _Tokenizer()
    mm_processor = SimpleNamespace(
        _processor=SimpleNamespace(tokenizer=tokenizer),
        mm_tokens=MultimodalSpecialTokens(
            image_token="<|vision_start|><|image_pad|><|vision_end|>",
            video_token="<|video_pad|>",
            image_token_id=151655,
            video_token_id=151656,
            image_token_regex=_build_grouped_mm_token_regex(
                "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"
            ),
        ).build(SimpleNamespace(tokenizer=tokenizer)),
    )

    assert maybe_enable_grouped_video_token_regex(mm_processor) is True

    expanded_video = (
        "<|vision_start|><|video_pad|><|video_pad|><|video_pad|><|vision_end|>"
    )
    text_parts = re.split(
        mm_processor.mm_tokens.get_combined_regex(),
        f"prefix {expanded_video} suffix",
    )
    matched_parts = [
        part
        for part in text_parts
        if mm_processor.mm_tokens.get_modality_of_token(part) == Modality.VIDEO
    ]

    assert matched_parts == [expanded_video]


@pytest.mark.asyncio
async def test_build_mm_items_routes_video_to_video_data():
    embeddings = torch.arange(24, dtype=torch.float16).reshape(6, 4)

    class _FakeEmbeddingsProcessor:
        async def process_embeddings(self, request):
            return embeddings, 17

        @staticmethod
        def create_multimodal_item(embeddings, modality, model_specific_data):
            return EmbeddingsProcessor.create_multimodal_item(
                embeddings, modality, model_specific_data
            )

    request = SglangMultimodalRequest(
        request=PreprocessedRequest(
            token_ids=[151652, 151656, 151653],
            stop_conditions=StopConditions(max_tokens=32),
            sampling_options=SamplingOptions(temperature=0.0),
        ),
        multimodal_inputs=[
            MultiModalGroup(
                multimodal_input=MultiModalInput(),
                modality="VIDEO",
                model_specific_data={
                    "video_grid_thw": [2, 3, 4],
                    "second_per_grid_ts": 0.5,
                    "video_timestamps": [0.25, 0.75],
                },
            )
        ],
    )

    mm_kwargs, combined_embeddings, tensor_id = await _build_mm_items(
        request, _FakeEmbeddingsProcessor()
    )

    assert tensor_id == 17
    assert torch.equal(combined_embeddings, embeddings)
    assert "image_data" not in mm_kwargs
    assert "video_data" in mm_kwargs

    mm_item = mm_kwargs["video_data"][0]
    assert mm_item["modality"] == "VIDEO"
    assert torch.equal(mm_item["video_grid_thw"], torch.tensor([[2, 3, 4]]))
    assert torch.equal(
        mm_item["second_per_grid_ts"], torch.tensor([0.5], dtype=torch.float32)
    )
    assert mm_item["video_timestamps"] == [[0.25, 0.75]]
