from future import __annotations__
import asyncio
import uvloop
import logging
import signal

from dynamo.llm import ModelRuntimeConfig, ModelType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

import sglang as sgl
from dynamo.sgl.args import parse_cmd_line_args, Config, DisaggregationMode

configure_dynamo_logging()

@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    loop = asyncio.get_running_loop()

    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logging.info("Signal handlers set up for graceful shutdown")

    config = parse_cmd_line_args()
    if config.serving_strategy != DisaggregationMode.PREFILL:
        await init(runtime, config)
    else:
        await init_prefill(runtime, config)

async def init(runtime: DistributedRuntime, config: Config):
    # We use a request handler factory pattern here
    # depending on the disaggregation mode and pass in 
    # the correct arguments all the way through

    # i think i will just do decode -> prefill
    # i dont think it makes sense to try to support both atm

    component = runtime.namespace(config.dynamo_args.namespace).component(config.dynamo_args.component)
    await component.create_service()

    engine = sgl.Engine(config.server_args)

    generate_endpoint = component.endpoint(config.dynamo_args.endpoint)

    if config.serving_strategy == DisaggregationMode.DECODE:
        prefill_client = (
            await runtime.namespace(config.dynamo_args.namespace)
            .component("prefill") # todo naming? from "next field?"
            .endpoint("generate")
            .client()
        )
    
    # setup publisher pieces (like trtllm maybe async?)

    # figure out that new register_llm_runtime piece

    # asyncio gather
    
    pass

async def init_prefill(runtime: DistributedRuntime, config: Config):
    # simple init
    pass    

    



    







async def graceful_shutdown(runtime):
    logging.info("Received shutdown signal, shutting down DistributedRuntime")
    runtime.shutdown()
    logging.info("DistributedRuntime shutdown complete")

def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()