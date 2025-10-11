# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import sglang as sgl

from dynamo._core import Component, Context
from dynamo.sglang.args import Config
from dynamo.sglang.protocol import EmbeddingRequest
from dynamo.sglang.publisher import DynamoSglangPublisher
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler


class EmbeddingWorkerHandler(BaseWorkerHandler):
    def __init__(
        self,
        component: Component,
        engine: sgl.Engine,
        config: Config,
        publisher: Optional[DynamoSglangPublisher] = None,
    ):
        super().__init__(component, engine, config, publisher)
        logging.info("Embedding worker handler initialized")

    def cleanup(self):
        self.engine.shutdown()
        logging.info("Engine shutdown")
        super().cleanup()

    async def generate(self, request: dict, context: Context):
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
        if isinstance(embedding_request.input, str):
            prompt = embedding_request.input
        elif isinstance(embedding_request.input, list):
            prompt = embedding_request.input
        else:
            raise TypeError(f"Invalid input type: {type(embedding_request.input)}")

        result = await self.engine.async_encode(prompt=prompt)

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
