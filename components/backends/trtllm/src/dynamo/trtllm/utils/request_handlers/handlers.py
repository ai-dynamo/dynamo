# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import logging

from dynamo.trtllm.utils.request_handlers.handler_base import (
    DisaggregationMode,
    DisaggregationStrategy,
    HandlerBase,
    RequestHandlerConfig,
)

# Configure detailed logging for disaggregation handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RequestHandlerFactory:
    def __init__(self):
        self.handlers = {
            "prefill": PrefillHandler,
            "decode": DecodeHandler,
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
        logger.debug("🔧 Initialized AggregatedHandler")
        logger.debug(f"   ➜ Mode: {config.disaggregation_mode.value}")

    async def generate(self, request: dict):
        logger.debug("=" * 80)
        logger.debug("🏭 AGGREGATED HANDLER - SINGLE WORKER PROCESSING")
        logger.debug("=" * 80)
        logger.debug("🔄 Processing both prefill and decode phases locally")

        # Implement all steps locally.
        async for res in self.generate_locally(request):
            yield res


class PrefillHandler(HandlerBase):
    """
    Handler for the prefill mode.
    """

    def __init__(self, config: RequestHandlerConfig):
        super().__init__(config)
        logger.debug("🔧 Initialized PrefillHandler")
        logger.debug(f"   ➜ Mode: {config.disaggregation_mode.value}")
        logger.debug(f"   ➜ Strategy: {config.disaggregation_strategy.value}")
        logger.debug(
            f"   ➜ Next client configured: {'Yes' if config.next_client else 'No'}"
        )

    async def remote_decode(self, request: dict):
        logger.debug("=" * 60)
        logger.debug("🌐 PREFILL → DECODE: REMOTE CALL")
        logger.debug("=" * 60)
        logger.debug("📡 Sending request to decode worker via round-robin")
        logger.debug(f"📦 Request keys being sent: {list(request.keys())}")

        decode_response_count = 0
        async for res in await self.next_client.round_robin(request):
            decode_response_count += 1
            logger.debug(f"📥 Received decode response #{decode_response_count}")
            logger.debug(f"📋 Decode response: {res.data()}")
            yield res.data()

        logger.debug(
            f"✅ Remote decode completed with {decode_response_count} responses"
        )

    async def generate(self, request: dict):
        logger.debug("=" * 80)
        logger.debug("🎯 PREFILL HANDLER - CONTEXT PROCESSING")
        logger.debug("=" * 80)
        logger.debug(f"📋 Strategy: {self.disaggregation_strategy.value}")

        # Generate the prefill response locally
        logger.debug("🔄 Step 1: Processing prefill phase locally")
        prefill_request = copy.deepcopy(request)
        prefill_response = None
        response_count = 0

        async for res in self.generate_locally(prefill_request):
            prefill_response = res
            response_count += 1
            logger.debug(f"📤 Prefill response #{response_count}: {res}")
            if response_count > 1:
                logger.error("❌ ERROR: Prefill should generate exactly one response")
                raise ValueError("Prefill response should be generated only once.")

        logger.debug(f"✅ Prefill phase completed in {response_count} iteration(s)")

        # Check for errors in prefill response
        is_error = self.check_error(prefill_response)
        logger.debug(f"🔍 Prefill error check: {'ERROR' if is_error else 'SUCCESS'}")

        if (
            self.disaggregation_strategy == DisaggregationStrategy.PREFILL_FIRST
            and not is_error
        ):
            logger.debug("=" * 60)
            logger.debug("🚀 PREFILL_FIRST STRATEGY: Triggering decode worker")
            logger.debug("=" * 60)
            logger.debug("📦 Step 2: Transferring state to decode worker")

            # If operating under prefill_first strategy, the prefill handler needs to trigger
            # the decode handler.
            if prefill_response is not None:
                logger.debug(
                    "🔄 Adding disaggregated_params to request for decode worker"
                )
                request["disaggregated_params"] = prefill_response[
                    "disaggregated_params"
                ]
                logger.debug(
                    f"📋 State transfer keys: {list(prefill_response['disaggregated_params'].keys())}"
                )

            logger.debug("📡 Initiating remote decode call...")
            async for res in self.remote_decode(request):
                logger.debug("📤 Forwarding decode response to client")
                yield res
        else:
            logger.debug("=" * 60)
            logger.debug("📤 DECODE_FIRST STRATEGY: Returning to decode worker")
            logger.debug("=" * 60)
            logger.debug(
                "🔄 Sending prefill response back to decode worker for local processing"
            )

            # Return response to the decode handler.
            yield prefill_response


class DecodeHandler(HandlerBase):
    """
    Handler for the decode mode.
    """

    def __init__(self, config: RequestHandlerConfig):
        super().__init__(config)
        logger.info("🔧 Initialized DecodeHandler")
        logger.debug(f"   ➜ Mode: {config.disaggregation_mode.value}")
        logger.debug(f"   ➜ Strategy: {config.disaggregation_strategy.value}")
        logger.debug(
            f"   ➜ Next client configured: {'Yes' if config.next_client else 'No'}"
        )

    async def remote_prefill(self, request: dict):
        logger.debug("=" * 60)
        logger.debug("🌐 DECODE → PREFILL: REMOTE CALL")
        logger.debug("=" * 60)
        logger.debug("📡 Sending request to prefill worker via round-robin")
        logger.debug(f"📦 Request keys being sent: {list(request.keys())}")

        prefill_response_count = 0
        async for res in await self.next_client.round_robin(request):
            prefill_response_count += 1
            logger.debug(f"📥 Received prefill response #{prefill_response_count}")
            logger.debug(f"📋 Prefill response data: {res}")
            yield res

        logger.info(
            f"✅ Remote prefill completed with {prefill_response_count} responses"
        )

    async def generate(self, request: dict):
        logger.debug("=" * 80)
        logger.debug("🎯 DECODE HANDLER - TOKEN GENERATION")
        logger.debug("=" * 80)
        logger.debug(f"📋 Strategy: {self.disaggregation_strategy.value}")

        if self.disaggregation_strategy == DisaggregationStrategy.DECODE_FIRST:
            logger.debug("=" * 60)
            logger.debug("🚀 DECODE_FIRST STRATEGY: Triggering prefill worker")
            logger.debug("=" * 60)
            logger.debug("🔄 Step 1: Requesting prefill processing from remote worker")

            prefill_response = None
            # If operating under decode_first strategy, the decode handler needs to trigger
            # the prefill handler.
            response_count = 0
            # FIX: Do not yield the prefill response directly.
            # Instead, capture it and extract the state.
            async for res in self.remote_prefill(request):
                prefill_response = res
                response_count += 1
                logger.debug(f"📥 Prefill response #{response_count}")
                if response_count > 1:
                    logger.error(
                        "❌ ERROR: Prefill should generate exactly one response"
                    )
                    raise ValueError("Prefill response should be generated only once.")

            logger.debug(f"✅ Remote prefill completed in {response_count} iteration(s)")

            response_data = (
                prefill_response.data() if prefill_response is not None else None
            )

            # Check for errors in prefill response
            if prefill_response is not None:
                is_error = self.check_error(response_data)
                logger.debug(
                    f"🔍 Prefill error check: {'ERROR' if is_error else 'SUCCESS'}"
                )

                if is_error:
                    logger.error(
                        "❌ Prefill worker returned error, forwarding to client"
                    )
                    # In case of an error, we might still need to yield it to terminate the stream.
                    # However, this error format might also be incompatible.
                    # For now, we yield it, but a more robust solution might be to raise an exception.
                    yield response_data
                    return

            if prefill_response is not None and response_data is not None:
                logger.debug(
                    "📦 Step 2: Extracting disaggregated_params from prefill response"
                )
                request["disaggregated_params"] = response_data["disaggregated_params"]
                logger.debug(
                    f"📋 Received state keys: {list(response_data['disaggregated_params'].keys())}"
                )
                logger.debug("✅ State successfully transferred from prefill worker")
        else:
            logger.debug("=" * 60)
            logger.debug("🔄 PREFILL_FIRST STRATEGY: Using existing state")
            logger.debug("=" * 60)
            logger.debug("📦 Disaggregated parameters should already be in request")

        logger.debug("=" * 60)
        logger.debug("🏭 DECODE PROCESSING: Generating tokens locally")
        logger.debug("=" * 60)
        logger.debug(
            "🔄 Step 3: Starting local decode generation with transferred state"
        )

        async for res in self.generate_locally(request):
            logger.debug("📤 Yielding decode response to client")
            yield res
