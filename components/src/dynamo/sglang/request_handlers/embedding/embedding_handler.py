# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import logging
import struct
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


def _encode_floats_to_base64(floats: List[float]) -> str:
    """Encode an embedding vector as a base64 string per the OpenAI
    ``encoding_format=base64`` spec: raw little-endian ``float32`` bytes
    are concatenated and base64-encoded with the standard alphabet.

    Mirrors the Rust ``encode_floats_to_base64`` helper in
    ``lib/llm/src/preprocessor.rs`` so the two backend code paths
    produce identical bytes for the same input.
    """
    packed = struct.pack(f"<{len(floats)}f", *floats)
    return base64.b64encode(packed).decode("ascii")


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

        encoding_format = embedding_request.encoding_format

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
            encoding_format=encoding_format,
        )
        yield response

    def _transform_response(
        self,
        ret: Any,
        model_name: str,
        dimensions: Optional[int] = None,
        encoding_format: str = "float",
    ) -> Dict[str, Any]:
        """Transform SGLang response to the internal worker->frontend
        embedding format.

        - ``dimensions``: Matryoshka-style truncation; keeps the first N
          values of each embedding vector.
        - ``encoding_format``: validated upstream in ``generate`` for
          spec compliance, but no longer branches the worker's output.
          The ``embedding`` field is always emitted as a base64-encoded
          little-endian ``float32`` byte string; the Rust HTTP frontend
          decodes back to a JSON array of floats at the HTTP boundary
          when the client asked for float. Truncation runs before
          encoding so the base64 byte count matches the requested
          dimensionality.
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

            # Always emit base64 over the worker->frontend wire format,
            # mirroring the vLLM embedding handler. The Rust HTTP frontend
            # decodes back to a float array when the client's
            # ``encoding_format`` is float (the OpenAI default); when the
            # client asked for base64 the payload is passed through. JSON
            # float arrays of 15 x 1024 floats cost ~115 ms per request in
            # Python ``json.dumps`` + Rust ``serde_json`` parse across NATS;
            # base64 bytes avoid both halves of that cost.
            embedding_objects.append(
                {
                    "object": "embedding",
                    "embedding": _encode_floats_to_base64(embedding),
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
