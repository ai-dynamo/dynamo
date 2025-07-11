import copy
from dataclasses import dataclass

from utils.request_handlers.handler_base import HandlerBase


@dataclass
class RequestHandlerConfig:
    """
    Configuration for the request handler
    """

    component: object
    engine: object
    default_sampling_params: object
    publisher: object
    disaggregation_mode: str
    disaggregation_strategy: str
    next_client: object


class RequestHandlerFactory:
    def __init__(self):
        self.handlers = {
            "prefill": PrefillHandler,
            "decode": DecodeHandler,
            "prefill_and_decode": AggregatedHandler,
        }

    def _validate_config(self, config: RequestHandlerConfig):
        if config.disaggregation_mode not in self.handlers:
            raise ValueError(
                f"Invalid disaggregation_mode '{config.disaggregation_mode}'. "
                f"Supported modes: {list(self.handlers.keys())}"
            )

        if not config.next_client:
            if (
                config.disaggregation_mode == "prefill"
                and config.disaggregation_strategy == "prefill_first"
            ):
                raise ValueError(
                    "Next client is required for the main worker when disaggregation_mode='prefill' and disaggregation_strategy='prefill_first'."
                )
            if (
                config.disaggregation_mode == "decode"
                and config.disaggregation_strategy == "decode_first"
            ):
                raise ValueError(
                    "Next client is required for the decode worker when disaggregation_mode='decode' and disaggregation_strategy='decode_first'."
                )

    def get_request_handler(self, config: RequestHandlerConfig) -> HandlerBase:
        self._validate_config(config)
        return self.handlers[config.disaggregation_mode](config)


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

        if self.disaggregation_strategy == "prefill_first" and not self.check_error(prefill_response):
            # If operating under prefill_first strategy, the prefill handler needs to trigger
            # the decode handler.
            request["disaggregated_params"] = prefill_response["disaggregated_params"]
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
        if self.disaggregation_strategy == "decode_first":
            prefill_response = None
            # If operating under decode_first strategy, the decode handler needs to trigger
            # the prefill handler.
            response_count = 0
            async for res in self.remote_prefill(request):
                prefill_response = res
                response_count += 1
                if response_count > 1:
                    raise ValueError("Prefill response should be generated only once.")

            if self.check_error(prefill_response.data()):
                yield prefill_response.data()
                return
            request["disaggregated_params"] = prefill_response.data()["disaggregated_params"]


        async for res in self.generate_locally(request):
            yield res
