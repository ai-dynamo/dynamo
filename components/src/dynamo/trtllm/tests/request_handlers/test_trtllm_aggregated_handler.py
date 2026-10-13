# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AggregatedHandler."""

from unittest.mock import patch

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "Skipping to avoid errors during collection with '-m gpu_0'. "
        "CUDA/GPU not available, but tensorrt_llm import and the test require GPU.",
        allow_module_level=True,
    )
from tensorrt_llm.llmapi import DisaggregatedParams

from dynamo.trtllm.request_handlers.aggregated_handler import AggregatedHandler
from dynamo.trtllm.tests.request_handlers.utils import (
    create_mock_context,
    create_mock_encoder_cache,
    run_generate_with_mock_fetch,
    setup_multimodal_config,
)
from dynamo.trtllm.tests.utils import create_mock_request_handler_config

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
]

FETCH_PATCH_PATH = (
    "dynamo.trtllm.request_handlers.aggregated_handler.fetch_embeddings_from_encoder"
)


class TestAggregatedHandlerGenerate:
    """Tests for AggregatedHandler.generate method."""

    @pytest.mark.asyncio
    async def test_embeddings_passed_to_generate_locally(self):
        """Cache path: List[Tensor] passed as embeddings."""
        config = create_mock_request_handler_config(
            disaggregation_mode="prefill_and_decode"
        )
        setup_multimodal_config(config, ["http://example.com/image.jpg"])
        handler = AggregatedHandler(config, encoder_cache=create_mock_encoder_cache())

        expected_embeddings = [torch.randn(10, 256)]

        embeddings, ep_params = await run_generate_with_mock_fetch(
            handler, FETCH_PATCH_PATH, expected_embeddings
        )

        assert embeddings is expected_embeddings
        assert ep_params is None

    @pytest.mark.asyncio
    async def test_disaggregated_params_passed_to_generate_locally(self):
        """No-cache path: DisaggregatedParams passed as ep_params."""
        config = create_mock_request_handler_config(
            disaggregation_mode="prefill_and_decode"
        )
        setup_multimodal_config(config, ["http://example.com/image.jpg"])
        handler = AggregatedHandler(config, encoder_cache=None)

        expected_params = DisaggregatedParams(request_type="context_only")

        embeddings, ep_params = await run_generate_with_mock_fetch(
            handler, FETCH_PATCH_PATH, expected_params
        )

        assert embeddings is None
        assert ep_params is expected_params

    @pytest.mark.asyncio
    async def test_text_path_returns_generate_locally_stream(self):
        """Without a remote encoder, generate returns generate_locally's stream
        itself instead of re-yielding each chunk through another generator."""
        handler = AggregatedHandler(create_mock_request_handler_config())
        request = {"token_ids": [1, 2, 3]}
        context = create_mock_context()
        chunks = [{"token_ids": [4]}, {"token_ids": [5]}]

        async def local_stream():
            for chunk in chunks:
                yield chunk

        stream = local_stream()
        with patch.object(
            handler, "generate_locally", return_value=stream
        ) as generate_locally:
            result = handler.generate(request, context)

        assert result is stream
        generate_locally.assert_called_once_with(request, context, None, None)
        assert [chunk async for chunk in result] == chunks
