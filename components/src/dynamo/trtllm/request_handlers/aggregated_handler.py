# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Handler for aggregated (prefill + decode) mode with optional encoder disaggregation."""

import logging
from typing import Optional

import nvtx

from dynamo._core import Context
from dynamo.common.memory.multimodal_embedding_cache_manager import (
    MultimodalEmbeddingCacheManager,
)
from dynamo.trtllm.multimodal.embedding_fetcher import fetch_embeddings_from_encoder
from dynamo.trtllm.request_handlers.handler_base import (
    HandlerBase,
    RequestHandlerConfig,
)


class AggregatedHandler(HandlerBase):
    """
    Handler for aggregated mode (prefill + decode in single worker).

    Supports optional encoder disaggregation (E_PD flow) when encode_client
    and encoder_cache are configured.
    """

    def __init__(
        self,
        config: RequestHandlerConfig,
        encoder_cache: Optional[MultimodalEmbeddingCacheManager] = None,
    ):
        super().__init__(config)
        self._encoder_cache = encoder_cache

    async def generate(self, request: dict, context: Context):
        """Generate response, optionally using remote encoder for multimodal."""
        _rng_agg = nvtx.start_range(message="agg:total")
        logging.debug(f"AggregatedHandler Request ID: {context.id()}")

        embeddings = None
        ep_disaggregated_params = None
        if self.multimodal_processor and self.encode_client:
            messages = request.get("extra_args", {}).get(
                "messages", request.get("messages", [])
            )
            _, image_urls, _ = self.multimodal_processor.extract_prompt_and_media(
                messages
            )
            if image_urls:
                logging.info(f"AggregatedHandler: image_urls={image_urls}")
                _rng_fetch = nvtx.start_range(message="agg:fetch_embeddings_from_encoder")
                result = await fetch_embeddings_from_encoder(
                    image_urls,
                    request,
                    self.encode_client,
                    self._encoder_cache,
                )
                nvtx.end_range(_rng_fetch)
                if isinstance(result, list):
                    embeddings = result
                else:
                    ep_disaggregated_params = result

        try:
            async for res in self.generate_locally(
                request, context, embeddings, ep_disaggregated_params
            ):
                yield res
        finally:
            nvtx.end_range(_rng_agg)
