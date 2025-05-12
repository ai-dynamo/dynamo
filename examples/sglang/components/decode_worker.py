from __future__ import annotations

import logging
import signal

from utils.protocol import DisaggPreprocessedRequest
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

        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, self.shutdown_sglang_engine)

        logger.warning("Decode worker initialized")

    def shutdown_sglang_engine(self, signum, frame):
        logger.info("Shutting down SGLang engine")
        self.engine.shutdown()

    @dynamo_endpoint()
    async def generate(self, req: DisaggPreprocessedRequest):
        g = await self.engine.async_generate(
            input_ids=req.request.token_ids,
            sampling_params=req.sampling_params,
            stream=True,
            bootstrap_host=req.bootstrap_host,
            bootstrap_port=req.bootstrap_port,
            bootstrap_room=req.bootstrap_room,
        )

        async for result in g:
            yield result
