# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MultimodalRequestProcessor embedding paths.

These tests ensure multi_modal_embeddings is always a dict with an "image" key
(list of tensors), as required by TRT-LLM's attach_multimodal_embeddings.
Regression test for "multimodal_embedding must be a dictionary" ValueError.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "Skipping to avoid errors during collection with '-m gpu_0'. "
        "CUDA/GPU not available, but tensorrt_llm import and the test require GPU.",
        allow_module_level=True,
    )
from dynamo.trtllm.multimodal_processor import MultimodalRequestProcessor

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
]


def _make_processor():
    """Minimal processor with mock tokenizer (no model load)."""
    return MultimodalRequestProcessor(
        model_type="llava",
        model_dir="/fake/dir",
        max_file_size_mb=10,
        tokenizer=MagicMock(),
    )


def _request_with_tokens(token_ids=None):
    return {"token_ids": token_ids or [1, 2, 3]}


class TestProcessOpenAIRequestNIXLEmbeddings:
    """NIXL path: embeddings from encode worker must be wrapped as dict with 'image' key."""

    @pytest.mark.asyncio
    async def test_embeddings_tensor_wrapped_as_dict_with_image_key(self):
        """Single tensor must become multi_modal_embeddings['image'] list (TRT-LLM contract)."""
        processor = _make_processor()
        request = _request_with_tokens()
        embeddings = torch.randn(10, 256)

        result = await processor.process_openai_request(
            request, embeddings=embeddings, ep_disaggregated_params=None
        )

        assert result is not None
        assert "multi_modal_embeddings" in result
        mm_emb = result["multi_modal_embeddings"]
        assert isinstance(
            mm_emb, dict
        ), "TRT-LLM requires multimodal_embedding to be a dictionary"
        assert "image" in mm_emb
        assert isinstance(mm_emb["image"], list)
        assert len(mm_emb["image"]) == 1
        assert torch.equal(mm_emb["image"][0], embeddings)

    @pytest.mark.asyncio
    async def test_embeddings_list_passed_through_as_image_list(self):
        """List of tensors must be multi_modal_embeddings['image'] unchanged."""
        processor = _make_processor()
        request = _request_with_tokens()
        embeddings = [torch.randn(10, 256), torch.randn(10, 256)]

        result = await processor.process_openai_request(
            request, embeddings=embeddings, ep_disaggregated_params=None
        )

        assert result is not None
        mm_emb = result["multi_modal_embeddings"]
        assert isinstance(mm_emb, dict)
        assert mm_emb["image"] == embeddings


class TestProcessOpenAIRequestPDEmbeddingPaths:
    """PD path: loaded embedding files must be set as dict with 'image' key."""

    @pytest.mark.asyncio
    async def test_embedding_paths_produce_dict_with_image_key(self):
        """Loading .pt from multi_modal_data must set multi_modal_embeddings = {'image': [...]}."""
        processor = _make_processor()
        mock_tensor = torch.randn(5, 128)
        request = {
            "token_ids": [1, 2, 3],
            "multi_modal_data": {
                "image_url": [{"Url": "https://example.com/emb.pt"}],
            },
        }

        with patch.object(
            processor,
            "load_tensor_from_path_or_url",
            return_value=mock_tensor,
        ):
            result = await processor.process_openai_request(
                request, embeddings=None, ep_disaggregated_params=None
            )

        assert result is not None
        assert "multi_modal_embeddings" in result
        mm_emb = result["multi_modal_embeddings"]
        assert isinstance(mm_emb, dict)
        assert "image" in mm_emb
        assert isinstance(mm_emb["image"], list)
        assert len(mm_emb["image"]) == 1
        assert torch.equal(mm_emb["image"][0], mock_tensor)
