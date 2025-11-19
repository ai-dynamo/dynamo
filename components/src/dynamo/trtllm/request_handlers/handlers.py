# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from dynamo._core import Context
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.trtllm.encode_helper import EncodeHelper
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

    async def generate(self, request: dict, context: Context):
        logging.debug(f"New Request ID: {context.id()}")
        if self.connector:
            # Use helper method to process embedding request
            async for response in EncodeHelper.process_embedding_request(
                request, self.multimodal_processor, self.connector
            ):
                yield response
            return
        else:
            logging.error("encode handler: no Dynamo NIXL connector found")
            raise RuntimeError("encode handler: no Dynamo NIXL connector found")

        if not request.get("streaming", False):
            yield request
            return

        yield request


class PrefillHandler(HandlerBase):
    """
    Handler for prefill-only workers in disaggregated serving.
    """

    def __init__(self, config: RequestHandlerConfig):
        super().__init__(config)

    async def remote_encode_with_nixl(self, request: dict):
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

        if self.multimodal_processor:
            # Extract messages from extra_args (set by Rust preprocessor) or fall back to direct field
            messages = request.get("extra_args", {}).get(
                "messages", request.get("messages", [])
            )
            _, _, embedding_paths = self.multimodal_processor.extract_prompt_and_media(
                messages
            )
            if embedding_paths:
                if self.encode_client and self.connector:
                    logging.debug(
                        "PrefillHandler calling Encode Worker via remote_encode_with_nixl"
                    )
                    embeddings_tensor = await self.remote_encode_with_nixl(request)

        # Generate prefill response locally and return disaggregated_params
        response_count = 0
        async for res in self.generate_locally(request, context, embeddings_tensor):
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
