# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import logging

from tensorrt_llm.llmapi import DisaggregatedParams
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
        if self.multimodal_processor:
            self.model_dir = self.multimodal_processor.model_dir
            self.model_type = self.multimodal_processor.model_type
            self.tokenizer = self.multimodal_processor.tokenizer

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
        if self.multimodal_processor:
            self.model_dir = self.multimodal_processor.model_dir
            self.model_type = self.multimodal_processor.model_type
            self.tokenizer = self.multimodal_processor.tokenizer

    async def generate(self, request: dict, context: Context):
        logging.debug(f"New Request ID: {context.id()}")
        if self.connector:
            # Use helper method to process embedding request
            async for response in EncodeHelper.process_encode_request(
                request, self.multimodal_processor, self.connector, self.tokenizer, self.model_dir, self.model_type, self.engine, 
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

    async def remote_encode_full_epd(self, request: dict):
        async for res in await self.encode_client.round_robin(request):
            encode_response = res.data()
            break

        if not encode_response:
            raise RuntimeError("Did not receive a response from the encode worker.")

        return encode_response

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
        ep_disaggregated_params = None

        if self.multimodal_processor:
            _, image_urls, embedding_paths = self.multimodal_processor.extract_prompt_and_media(
                request.get("messages", [])
            )
            # Handle embedding paths (NIXL transfer of pre-computed embeddings)
            if embedding_paths:
                if self.encode_client and self.connector:
                    logging.debug(
                        "PrefillHandler calling Encode Worker via remote_encode_with_nixl"
                    )
                    embeddings_tensor = await self.remote_encode_with_nixl(request)
            # Handle image URLs (full E-PD flow with MultimodalEncoder)
            elif image_urls:
                if self.encode_client and self.connector:
                    logging.info("========== PREFILL WORKER: Full EPD - Requesting from Encode Worker ==========")
                    encode_response = await self.remote_encode_full_epd(request)
                    
                    # Check if encode worker returned disaggregated params (full EPD flow)
                    if "ep_disaggregated_params" in encode_response:
                        params_dict = encode_response["ep_disaggregated_params"]
                        if params_dict is not None:
                            # Reconstruct DisaggregatedParams object from dict
                            ep_disaggregated_params = DisaggregatedParams(**params_dict)
                            logging.info("PREFILL WORKER: ✅ Received ep_disaggregated_params with multimodal handles")
                            logging.info(f"PREFILL WORKER: Input ep_disaggregated_params.multimodal_embedding_handles = {getattr(ep_disaggregated_params, 'multimodal_embedding_handles', 'NOT FOUND')}")
                            ep_disaggregated_params.request_type = "context_only"
                            
                            # Get the processed prompt from encoder (includes <image> tokens)
                            # Store it in the request so multimodal_processor can access it
                            if "processed_prompt" in encode_response:
                                request["_epd_processed_prompt"] = encode_response["processed_prompt"]
                                logging.info(f"PREFILL WORKER: Stored processed prompt from encoder: {request['_epd_processed_prompt']}")
                            else:
                                logging.warning("PREFILL WORKER: ⚠️ No processed_prompt in encode_response")
                            
                            # Store prompt_token_ids from encoder for consistency with decode worker
                            if "prompt_token_ids" in encode_response and encode_response["prompt_token_ids"]:
                                request["_epd_prompt_token_ids"] = encode_response["prompt_token_ids"]
                                logging.info(f"PREFILL WORKER: Stored prompt_token_ids from encoder (length={len(encode_response['prompt_token_ids'])})")
                            else:
                                logging.warning("PREFILL WORKER: ⚠️ No prompt_token_ids in encode_response")
                        else:
                            logging.warning("PREFILL WORKER: ⚠️ Received None ep_disaggregated_params from encode worker")
                    else:
                        logging.info("PREFILL WORKER: ❌ Did not receive ep_disaggregated_params from encode worker")
        # Normal flow: Generate the prefill response locally with embeddings
        prefill_request = copy.deepcopy(request)
        prefill_response = None
        response_count = 0
        async for res in self.generate_locally(
            prefill_request, context, embeddings_tensor, ep_disaggregated_params
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
        """
        Send request to prefill. Try router first if available, fallback to direct worker.
        """
        # Format request in PreprocessedRequest format with extra_args
        prefill_request = copy.deepcopy(request)

        # Try router first if available, fallback to worker
        if (
            self.next_router_client is not None
            and self.next_router_client.instance_ids()
        ):
            try:
                # Call router's generate endpoint which returns LLMEngineOutput
                async for res in await self.next_router_client.generate(
                    prefill_request, context=context
                ):
                    yield res
                return
            except Exception as e:
                logging.warning(
                    f"Prefill router call failed: {e}. Falling back to direct worker."
                )

        # Fallback to direct worker
        if self.next_client is not None:
            async for res in await self.next_client.round_robin(
                prefill_request, context=context
            ):
                yield res
        else:
            raise ValueError("No prefill router or worker available")

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
                # Also extract the processed prompt for full EPD flow
                if "_epd_processed_prompt" in response_data:
                    request["_epd_processed_prompt"] = response_data["_epd_processed_prompt"]
                    logging.info(f"DECODE WORKER: Received processed prompt from prefill: {request['_epd_processed_prompt']}")

        async for res in self.generate_locally(request, context):
            yield res
