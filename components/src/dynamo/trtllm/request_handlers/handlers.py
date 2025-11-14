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
        logging.info(f"=== ENCODE WORKER === Request ID: {context.id()}")
        logging.info(f"EncodeHandler.generate received request with model: {request.get('model')}")
        if self.connector:
            # Use helper method to process embedding request
            async for response in EncodeHelper.process_embedding_request(
                request, self.multimodal_processor, self.connector
            ):
                logging.info(f"EncodeHandler yielding response")
                yield response
            logging.info(f"=== ENCODE WORKER === Request {context.id()} completed")
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
        logging.info(f"=== PREFILL WORKER === Request ID: {context.id()}")
        logging.info(f"PrefillHandler.generate received request with model: {request.get('model')}")
        logging.info(f"PrefillHandler request keys: {list(request.keys())}")
        logging.info(f"PrefillHandler has 'messages': {'messages' in request}")
        logging.info(f"PrefillHandler has 'token_ids': {'token_ids' in request}")
        logging.info(f"PrefillHandler has 'extra_args': {'extra_args' in request}")
        logging.debug(f"Full request: {request}")
        embeddings_tensor = None

        if self.multimodal_processor:
            _, _, embedding_paths = self.multimodal_processor.extract_prompt_and_media(
                request.get("messages", [])
            )
            if embedding_paths:
                if self.encode_client and self.connector:
                    logging.info(
                        "PrefillHandler calling Encode Worker via remote_encode_with_nixl"
                    )
                    embeddings_tensor = await self.remote_encode_with_nixl(request)
                    logging.info("PrefillHandler received embeddings from Encode Worker")

        # Generate prefill response locally and return disaggregated_params
        response_count = 0
        async for res in self.generate_locally(request, context, embeddings_tensor):
            response_count += 1
            if response_count > 1:
                raise ValueError("Prefill response should be generated only once.")

            if context.is_stopped() or context.is_killed():
                return

            # Return response with disaggregated_params to frontend
            logging.info(f"PrefillHandler yielding response with disaggregated_params: {res.get('disaggregated_params') is not None}")
            yield res
        logging.info(f"=== PREFILL WORKER === Request {context.id()} completed")


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
        logging.info(f"=== DECODE WORKER === Request ID: {context.id()}")
        logging.info(f"DecodeHandler.generate received request with model: {request.get('model')}")

        disaggregated_params = request.get("disaggregated_params")
        if disaggregated_params:
            logging.info(
                f"DecodeHandler using disaggregated params from prefill for request {context.id()}"
            )
        else:
            logging.warning(
                f"DecodeHandler received request WITHOUT disaggregated_params for request {context.id()}"
            )

        # Generate tokens locally (with or without disaggregated_params)
        token_count = 0
        async for res in self.generate_locally(request, context):
            token_count += 1
            yield res
        logging.info(f"=== DECODE WORKER === Request {context.id()} completed, yielded {token_count} responses")
