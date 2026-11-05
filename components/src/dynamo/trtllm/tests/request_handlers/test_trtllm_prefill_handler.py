# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PrefillHandler."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "Skipping to avoid errors during collection with '-m gpu_0'. "
        "CUDA/GPU not available, but tensorrt_llm import and the test require GPU.",
        allow_module_level=True,
    )
from tensorrt_llm.llmapi import DisaggregatedParams

from dynamo.trtllm.request_handlers.handlers import PrefillHandler
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
    "dynamo.trtllm.request_handlers.handlers.fetch_embeddings_from_encoder"
)


class TestPrefillHandlerInit:
    """Tests for PrefillHandler initialization."""

    def test_init_with_encoder_cache(self):
        """Test PrefillHandler can be initialized with encoder_cache."""
        config = create_mock_request_handler_config(disaggregation_mode="prefill")
        cache = create_mock_encoder_cache()

        handler = PrefillHandler(config, encoder_cache=cache)

        assert handler.engine == config.engine
        assert handler._encoder_cache == cache


class TestPrefillHandlerGenerate:
    """Tests for PrefillHandler.generate method."""

    @pytest.mark.asyncio
    async def test_embeddings_passed_to_generate_locally(self):
        """Cache path: List[Tensor] passed as embeddings."""
        config = create_mock_request_handler_config(disaggregation_mode="prefill")
        setup_multimodal_config(config, ["http://example.com/image.jpg"])
        handler = PrefillHandler(config, encoder_cache=create_mock_encoder_cache())

        expected_embeddings = [torch.randn(10, 256)]

        embeddings, ep_params = await run_generate_with_mock_fetch(
            handler, FETCH_PATCH_PATH, expected_embeddings
        )

        assert embeddings is expected_embeddings
        assert ep_params is None

    @pytest.mark.asyncio
    async def test_disaggregated_params_passed_to_generate_locally(self):
        """No-cache path: DisaggregatedParams passed as ep_params."""
        config = create_mock_request_handler_config(disaggregation_mode="prefill")
        setup_multimodal_config(config, ["http://example.com/image.jpg"])
        handler = PrefillHandler(config, encoder_cache=None)

        expected_params = DisaggregatedParams(request_type="context_only")

        embeddings, ep_params = await run_generate_with_mock_fetch(
            handler, FETCH_PATCH_PATH, expected_params
        )

        assert embeddings is None
        assert ep_params is expected_params

    @pytest.mark.asyncio
    async def test_embedding_paths_log_keeps_signed_url_out(self, caplog):
        """The query of a presigned URL is a bearer credential; the NIXL
        embedding path must log only scheme, host and path."""
        signature = "c2f1e9d8b7a6deadbeef0123456789abcdef"
        signed_url = (
            "https://bucket.s3.amazonaws.com/emb.safetensors"
            "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
            f"&X-Amz-Signature={signature}"
        )
        config = create_mock_request_handler_config(disaggregation_mode="prefill")
        config.multimodal_processor = MagicMock()
        config.multimodal_processor.extract_prompt_and_media_from_request = MagicMock(
            return_value=("text", [], [signed_url])
        )
        config.encode_client = MagicMock()
        config.connector = MagicMock()
        handler = PrefillHandler(config, encoder_cache=None)

        class _StopAfterLog(Exception):
            pass

        with patch.object(
            handler,
            "remote_encode_with_nixl",
            AsyncMock(side_effect=_StopAfterLog),
        ) as remote_encode, caplog.at_level(logging.DEBUG):
            with pytest.raises(_StopAfterLog):
                async for _ in handler.generate({}, create_mock_context()):
                    pass

        remote_encode.assert_awaited_once()
        assert "PrefillHandler: embedding_paths=" in caplog.text
        assert "bucket.s3.amazonaws.com/emb.safetensors" in caplog.text
        assert signature not in caplog.text
        assert "X-Amz-" not in caplog.text
