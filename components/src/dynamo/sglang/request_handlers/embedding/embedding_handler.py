# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any, Dict, Optional

import sglang as sgl

from dynamo._core import Context
from dynamo.common.utils.otel_tracing import build_trace_headers
from dynamo.sglang.args import Config
from dynamo.sglang.protocol import EmbeddingRequest
from dynamo.sglang.publisher import DynamoSglangPublisher
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler


class EmbeddingWorkerHandler(BaseWorkerHandler):
    def __init__(
        self,
        engine: sgl.Engine,
        config: Config,
        publisher: Optional[DynamoSglangPublisher] = None,
        shutdown_event: Optional[asyncio.Event] = None,
    ):
        super().__init__(engine, config, publisher, None, shutdown_event)
        logging.info("Embedding worker handler initialized")

    def cleanup(self) -> None:
        super().cleanup()
        self.engine.shutdown()
        logging.info("Engine shutdown")

    async def generate(
        self, request: dict, context: Context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate embeddings for the given input.

        Args:
            request: Embedding request dictionary.
            context: Context object for cancellation handling.
        """
        logging.debug(f"Embedding request: {request}")

        # Parse the embedding request - should only receive EmbeddingRequest format
        embedding_request = EmbeddingRequest(**request)

        # Handle different input types
        prompt: str | list[Any]
        if isinstance(embedding_request.input, str):
            prompt = embedding_request.input
        elif isinstance(embedding_request.input, list):
            prompt = embedding_request.input
        else:
            raise TypeError(f"Invalid input type: {type(embedding_request.input)}")

        trace_header = build_trace_headers(context) if self.enable_trace else None
        trace_id = context.trace_id

        result = await self.engine.async_encode(
            prompt=prompt,
            external_trace_header=trace_header,
            rid=trace_id,
        )

        # Transform the response to OpenAI format
        response = self._transform_response(result, embedding_request.model)
        yield response

    def _transform_response(self, ret, model_name):
        """Transform SGLang response to OpenAI embedding format"""
        if not isinstance(ret, list):
            ret = [ret]

        embedding_objects = []
        prompt_tokens = 0

        for idx, ret_item in enumerate(ret):
            embedding_objects.append(
                {
                    "object": "embedding",
                    "embedding": ret_item["embedding"],
                    "index": idx,
                }
            )
            prompt_tokens += ret_item.get("meta_info", {}).get("prompt_tokens", 0)

        return {
            "object": "list",
            "data": embedding_objects,
            "model": model_name,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
        }
