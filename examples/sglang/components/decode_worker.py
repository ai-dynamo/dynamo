from __future__ import annotations

import logging

from utils.protocol import MyRequestOutput, SGLangGenerateRequest
from utils.sglang import parse_sglang_args

import sglang as sgl
from dynamo.sdk import dynamo_endpoint, service

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1},
    workers=1,
)
class SGLangDecodeWorker:
    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_sglang_args(class_name, "")
        self.engine = sgl.Engine(server_args=self.engine_args)
        logger.warning("decode worker initialized")

    def shutdown_sglang_engine(self, signum, frame):
        logger.info("Shutting down SGLang engine")
        self.engine.shutdown()

    @dynamo_endpoint()
    async def generate(self, request: SGLangGenerateRequest):
        # TODO: remove
        logger.info(f"Generating with request: {request}")
        g = await self.engine.async_generate(
            input_ids=request.input_ids,
            sampling_params=request.sampling_params,
            stream=True,
            bootstrap_host=request.bootstrap_host,
            bootstrap_port=request.bootstrap_port,
            bootstrap_room=request.bootstrap_room,
        )

        async for result in g:
            yield MyRequestOutput(text=result).model_dump_json()
