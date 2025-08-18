import logging

import msgspec
import sglang as sgl

from dynamo._core import Component
from dynamo.sgl.args import Config
from dynamo.sgl.request_handlers.handler_base import BaseWorkerHandler


class PrefillWorkerHandler(BaseWorkerHandler):
    def __init__(self, component: Component, engine: sgl.Engine, config: Config):
        super().__init__(component, engine, config)
        logging.info("Prefill worker handler initialized")

    async def generate(self, request: str):
        req = msgspec.json.decode(request, type=dict)

        results = await self.engine.async_generate(
            input_ids=req["request"]["token_ids"],
            sampling_params=req["sampling_params"],
            stream=True,
            bootstrap_host=req["bootstrap_host"],
            bootstrap_port=req["bootstrap_port"],
            bootstrap_room=req["bootstrap_room"],
        )

        async for result in results:
            yield result
