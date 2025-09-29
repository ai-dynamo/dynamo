# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import logging

from dynamo._core import Context
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.trtllm.encode_helper import EncodeHelper
from dynamo.trtllm.request_handlers.handler_base import (
    DisaggregationMode,
    DisaggregationStrategy,
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

    def _validate_config(self, config: RequestHandlerConfig):
        if config.disaggregation_mode.value not in self.handlers:
            raise ValueError(
                f"Invalid disaggregation_mode '{config.disaggregation_mode.value}'"
            )

        if not config.next_client:
            if (
                config.disaggregation_mode == DisaggregationMode.PREFILL
                and config.disaggregation_strategy
                == DisaggregationStrategy.PREFILL_FIRST
            ):
                raise ValueError(
                    "Next client is required for the main worker when disaggregation_mode='prefill' and disaggregation_strategy='prefill_first'."
                )
            if (
                config.disaggregation_mode == DisaggregationMode.DECODE
                and config.disaggregation_strategy
                == DisaggregationStrategy.DECODE_FIRST
            ):
                raise ValueError(
                    "Next client is required for the decode worker when disaggregation_mode='decode' and disaggregation_strategy='decode_first'."
                )

    def get_request_handler(self, config: RequestHandlerConfig) -> HandlerBase:
        self._validate_config(config)
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
    Handler for the prefill mode.
    """

    def __init__(self, config: RequestHandlerConfig):
        super().__init__(config)

    async def remote_encode_with_nixl(self, request: dict):
        # 2. Get response with shape info and readable metadata
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

    async def remote_decode(self, request: dict, context: Context):
        async for res in await self.next_client.round_robin(request, context=context):
            yield res.data()

    async def generate(self, request: dict, context: Context):
        logging.debug(f"New Request ID: {context.id()}")
        logging.debug(f"PrefillHandler.generate received request: {request}")
        embeddings_tensor = None

        if self.multimodal_processor:
            _, _, embedding_paths = self.multimodal_processor.extract_prompt_and_media(
                request.get("messages", [])
            )
            # This check will be removed once TRTLLM Encoder is integrated.
            if embedding_paths:
                if self.encode_client and self.connector:
                    logging.debug(
                        "PrefillHandler calling Encode Worker via remote_encode_with_nixl"
                    )
                    embeddings_tensor = await self.remote_encode_with_nixl(request)
        # Generate the prefill response locally
        prefill_request = copy.deepcopy(request)
        prefill_response = None
        response_count = 0
        async for res in self.generate_locally(
            prefill_request, context, embeddings_tensor
        ):
            prefill_response = res
            response_count += 1
            if response_count > 1:
                raise ValueError("Prefill response should be generated only once.")

        if context.is_stopped() or context.is_killed():
            # Local generate abort monitor will print debug log, so only returning here.
            return

        if (
            self.disaggregation_strategy == DisaggregationStrategy.PREFILL_FIRST
            and not self.check_error(prefill_response)
        ):
            # If operating under prefill_first strategy, the prefill handler needs to trigger
            # the decode handler.
            if prefill_response is not None:
                request["disaggregated_params"] = prefill_response[
                    "disaggregated_params"
                ]
            async for res in self.remote_decode(request, context):
                yield res

            if context.is_stopped() or context.is_killed():
                logging.debug(f"Aborted Remote Request ID: {context.id()}")
                return
        else:
            # Return response to the decode handler.
            yield prefill_response


class DecodeHandler(HandlerBase):
    """
    Handler for the decode mode.
    """

    def __init__(self, config: RequestHandlerConfig):
        super().__init__(config)

    async def remote_prefill(self, request: dict, context: Context):
        async for res in await self.next_client.round_robin(request, context=context):
            yield res

    async def generate(self, request: dict, context: Context):
        logging.debug(f"New Request ID: {context.id()}")
        if self.disaggregation_strategy == DisaggregationStrategy.DECODE_FIRST:
            prefill_response = None
            # If operating under decode_first strategy, the decode handler needs to trigger
            # the prefill handler.
            response_count = 0
            # Do not yield the prefill response directly.
            # Instead, capture it and extract the state.
            async for res in self.remote_prefill(request, context):
                prefill_response = res
                response_count += 1
                if response_count > 1:
                    raise ValueError("Prefill response should be generated only once.")

            if context.is_stopped() or context.is_killed():
                logging.debug(f"Aborted Remote Request ID: {context.id()}")
                return

            response_data = (
                prefill_response.data() if prefill_response is not None else None
            )
            if prefill_response is not None and self.check_error(response_data):
                yield response_data
                return

            if prefill_response is not None and response_data is not None:
                request["disaggregated_params"] = response_data["disaggregated_params"]

        async for res in self.generate_locally(request, context):
            yield res
