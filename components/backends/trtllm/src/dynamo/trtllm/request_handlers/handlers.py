# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from dynamo.trtllm.request_handlers.handler_base import (
    DisaggregationMode,
    DisaggregationStrategy,
    HandlerBase,
    RequestHandlerConfig,
)


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

    async def generate(self, request: dict):
        # Implement all steps locally.
        async for res in self.generate_locally(request):
            yield res


class PrefillHandler(HandlerBase):
    """
    Handler for the prefill mode.
    """

    def __init__(self, config: RequestHandlerConfig):
        super().__init__(config)

    async def remote_decode(self, request: dict):
        async for res in await self.next_client.round_robin(request):
            yield res.data()

    async def generate(self, request: dict):
        # Generate the prefill response locally
        prefill_request = copy.deepcopy(request)
        prefill_response = None
        response_count = 0
        async for res in self.generate_locally(prefill_request):
            prefill_response = res
            response_count += 1
            if response_count > 1:
                raise ValueError("Prefill response should be generated only once.")

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
            async for res in self.remote_decode(request):
                yield res
        else:
            # Return response to the decode handler.
            yield prefill_response


class DecodeHandler(HandlerBase):
    """
    Handler for the decode mode.
    """

    def __init__(self, config: RequestHandlerConfig):
        super().__init__(config)

    async def remote_prefill(self, request: dict):
        async for res in await self.next_client.round_robin(request):
            yield res

    async def generate(self, request: dict):
        if self.disaggregation_strategy == DisaggregationStrategy.DECODE_FIRST:
            prefill_response = None
            # If operating under decode_first strategy, the decode handler needs to trigger
            # the prefill handler.
            response_count = 0
            # Do not yield the prefill response directly.
            # Instead, capture it and extract the state.
            async for res in self.remote_prefill(request):
                prefill_response = res
                response_count += 1
                if response_count > 1:
                    raise ValueError("Prefill response should be generated only once.")

            response_data = (
                prefill_response.data() if prefill_response is not None else None
            )
            if prefill_response is not None and self.check_error(response_data):
                yield response_data
                return

            if prefill_response is not None and response_data is not None:
                request["disaggregated_params"] = response_data["disaggregated_params"]

        async for res in self.generate_locally(request):
            yield res
