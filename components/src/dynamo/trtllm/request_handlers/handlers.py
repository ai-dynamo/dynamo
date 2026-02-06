# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

from dynamo._core import Context
from dynamo.common.memory.encoder_cache_manager import EncoderCacheManager
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.trtllm.encode_helper import EncodeHelper
from dynamo.trtllm.multimodal.embedding_fetcher import fetch_embeddings_from_encoder
from dynamo.trtllm.request_handlers.handler_base import (
    HandlerBase,
    RequestHandlerConfig,
)

configure_dynamo_logging()


class RequestHandlerFactory:
    def __init__(self):
        self.handlers = {
            "prefill": PrefillHandler,
            "decode": DecodeHandler,
            "encode": EncodeHandler,
            "prefill_and_decode": AggregatedHandler,
        }

    def get_request_handler(self, config: RequestHandlerConfig) -> HandlerBase:
        if config.disaggregation_mode.value not in self.handlers:
            raise ValueError(
                f"Invalid disaggregation_mode '{config.disaggregation_mode.value}'"
            )
        if config.disaggregation_mode.value == "prefill":
            encoder_cache = None
            if config.encoder_cache_capacity_gb > 0:
                # Create encoder cache for prefill handler
                capacity_bytes = int(config.encoder_cache_capacity_gb * 1024**3)
                encoder_cache = EncoderCacheManager(capacity_bytes)
            return PrefillHandler(config, encoder_cache=encoder_cache)
        return self.handlers[config.disaggregation_mode.value](config)


def get_request_handler(config: RequestHandlerConfig) -> HandlerBase:
    return RequestHandlerFactory().get_request_handler(config)


class AggregatedHandler(HandlerBase):
    """
    Handler for the aggregated mode.
    """

    def __init__(self, config: RequestHandlerConfig):
        super().__init__(config)

    async def generate(self, request: dict, context: Context):
        logging.debug(f"New Request ID: {context.id()}")
        # Implement all steps locally.
        async for res in self.generate_locally(request, context):
            yield res


class EncodeHandler(HandlerBase):
    """
    Handler for the encode mode.
    """

    def __init__(self, config: RequestHandlerConfig):
        super().__init__(config)
        # Initialize to None by default to avoid AttributeError if multimodal_processor is not set
        self.model_dir = None
        self.model_type = None
        self.tokenizer = None
        if self.multimodal_processor:
            self.model_dir = self.multimodal_processor.model_dir
            self.model_type = self.multimodal_processor.model_type
            self.tokenizer = self.multimodal_processor.tokenizer

    async def generate(self, request: dict, context: Context):
        logging.debug(f"New Request ID: {context.id()}")
        if self.multimodal_processor is None:
            logging.error("encode handler: no multimodal_processor configured")
            raise RuntimeError("encode handler: no multimodal_processor configured")

        async for response in EncodeHelper.process_encode_request(
            request,
            self.multimodal_processor,
            self.connector,
            self.tokenizer,
            self.model_dir,
            self.model_type,
            self.engine,
        ):
            yield response
        return


class PrefillHandler(HandlerBase):
    """
    Handler for prefill-only workers in disaggregated serving.
    """

    def __init__(
        self,
        config: RequestHandlerConfig,
        encoder_cache: Optional[EncoderCacheManager] = None,
    ):
        super().__init__(config)
        self._encoder_cache = encoder_cache

    async def remote_encode_with_nixl(self, request: dict):
        """
        Call encode worker for NIXL flow to load embeddings and unpack the response.

        Args:
            request: Request dict

        Returns:
            Encoder's embeddings tensor to be used by the prefill worker
        """
        # Get response with shape info and readable metadata
        encode_response = None
        async for res in await self.encode_client.round_robin(request):
            encode_response = res.data()
            break

        if not encode_response:
            raise RuntimeError("Did not receive a response from the encode worker.")

        # Use utility function to handle NIXL reading and reconstruction
        return await EncodeHelper.read_embeddings_from_encode_response(
            encode_response, self.connector
        )

    async def generate(self, request: dict, context: Context):
        """
        Prefill worker: process prompt and return disaggregated_params.
        Frontend routes to decode workers automatically.
        """
        logging.debug(f"Prefill Request ID: {context.id()}")
        logging.debug(f"PrefillHandler.generate received request: {request}")
        embeddings_tensor = None
        ep_disaggregated_params = None

        if self.multimodal_processor:
            # Extract messages from extra_args (set by Rust preprocessor) or fall back to direct field
            messages = request.get("extra_args", {}).get(
                "messages", request.get("messages", [])
            )
            (
                _,
                image_urls,
                embedding_paths,
            ) = self.multimodal_processor.extract_prompt_and_media(messages)
            # Handle embedding paths (NIXL transfer of pre-computed embeddings)
            if embedding_paths:
                if self.encode_client and self.connector:
                    logging.info(f"PrefillHandler: embedding_paths={embedding_paths}")
                    embeddings_tensor = await self.remote_encode_with_nixl(request)
                else:
                    # We can still handle embedding_paths without NIXL:
                    # `MultimodalRequestProcessor.process_openai_request` will load the embeddings
                    # locally in the prefill worker as a fallback. The encode-worker+NIXL path is
                    # useful when you want a dedicated I/O stage and/or explicit RDMA transfer.
                    logging.info(
                        "PrefillHandler: no encode_client/connector; falling back to local embedding load"
                    )

            # Handle image URLs (full E-PD flow with MultimodalEncoder)
            elif image_urls:
                if self.encode_client:
                    logging.info(f"PrefillHandler: image_urls={image_urls}")
                    result = await fetch_embeddings_from_encoder(
                        image_urls,
                        request,
                        self.encode_client,
                        self._encoder_cache,
                    )
                    if isinstance(result, list):
                        # Cache path: got List[torch.Tensor]
                        embeddings_tensor = result
                    else:
                        # No-cache path: got DisaggregatedParams
                        ep_disaggregated_params = result

        # Normal flow: Generate the prefill response locally with embeddings
        response_count = 0
        async for res in self.generate_locally(
            request, context, embeddings_tensor, ep_disaggregated_params
        ):
            response_count += 1
            if response_count > 1:
                raise ValueError("Prefill response should be generated only once.")

            if context.is_stopped() or context.is_killed():
                return

            # Return response with disaggregated_params to frontend
            yield res


class DecodeHandler(HandlerBase):
    """
    Handler for decode-only workers in disaggregated serving.
    """

    def __init__(self, config: RequestHandlerConfig):
        super().__init__(config)

    async def generate(self, request: dict, context: Context):
        """
        Decode worker: generate tokens using disaggregated_params from prefill.
        If disaggregated_params is present, prefill was done. Otherwise generate normally.
        """
        logging.debug(f"Decode Request ID: {context.id()}")

        disaggregated_params = request.get("disaggregated_params")
        if disaggregated_params:
            logging.debug(
                f"Using disaggregated params from prefill for request {context.id()}"
            )

        # Generate tokens locally (with or without disaggregated_params)
        async for res in self.generate_locally(request, context):
            yield res
