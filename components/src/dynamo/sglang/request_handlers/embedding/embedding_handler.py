# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any, Dict, List, Optional

import sglang as sgl

from dynamo._core import Context
from dynamo.sglang.args import Config
from dynamo.sglang.protocol import EmbeddingRequest
from dynamo.sglang.publisher import DynamoSglangPublisher
from dynamo.sglang.request_handlers.embedding.metrics import (
    observe_embedding_batch_size,
    observe_embedding_input_tokens,
)
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

        # Lower-bound validation runs BEFORE async_encode so an obviously
        # bad request (e.g. ``dimensions=0``) is rejected as HTTP 400
        # without spending a full pooling forward pass. The upper-bound
        # check stays in ``_transform_response`` because it needs to
        # compare against the actual embedding length returned by SGLang.
        dimensions = embedding_request.dimensions
        if dimensions is not None and dimensions < 1:
            raise ValueError(f"dimensions must be >= 1, got {dimensions}")

        trace_header = context.trace_headers() if self.enable_trace else None
        trace_id = context.trace_id

        result = await self.engine.async_encode(
            prompt=prompt,
            external_trace_header=trace_header,
            rid=trace_id,
        )

        # Transform the response to OpenAI format
        response = self._transform_response(
            result,
            embedding_request.model,
            dimensions=dimensions,
        )
        yield response

    def _transform_response(
        self,
        ret: Any,
        model_name: str,
        dimensions: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Transform SGLang response to OpenAI embedding format.

        Applies the optional ``dimensions`` field for Matryoshka-style
        truncation (slice leading N).

        Note: ``encoding_format=base64`` is part of the OpenAI spec but
        cannot be honored at this layer alone -- the Rust frontend's
        response aggregator deserializes ``data[].embedding`` as
        ``Vec<f32>`` (inherited from the upstream ``async_openai``
        embeddings types), so a base64 string here would be rejected
        downstream. Supporting it end-to-end requires owning the
        embedding response type in ``lib/protocols`` and updating the
        aggregator. Tracked separately.
        """
        if not isinstance(ret, list):
            ret = [ret]

        embedding_objects = []
        prompt_tokens = 0

        for idx, ret_item in enumerate(ret):
            embedding: List[float] = list(ret_item["embedding"])
            if dimensions is not None:
                # Lower-bound (dimensions >= 1) is checked upfront in
                # ``generate`` before async_encode runs.
                if dimensions > len(embedding):
                    raise ValueError(
                        f"dimensions={dimensions} exceeds model embedding "
                        f"dimension {len(embedding)}"
                    )
                embedding = embedding[:dimensions]

            embedding_objects.append(
                {
                    "object": "embedding",
                    "embedding": embedding,
                    "index": idx,
                }
            )
            prompt_tokens += ret_item.get("meta_info", {}).get("prompt_tokens", 0)

        try:
            observe_embedding_batch_size(model_name, len(embedding_objects))
            observe_embedding_input_tokens(model_name, prompt_tokens)
        except Exception:
            logging.warning("Failed to record embedding metrics", exc_info=True)

        return {
            "object": "list",
            "data": embedding_objects,
            "model": model_name,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
        }
