import asyncio
import logging
import signal

import sglang as sgl
import uvloop

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sgl.args import Config, DisaggregationMode, parse_args
from dynamo.sgl.publisher import setup_sgl_metrics

configure_dynamo_logging()


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    loop = asyncio.get_running_loop()

    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logging.info("Signal handlers will trigger a graceful shutdown of the runtime")

    config = parse_args()
    if config.serving_mode != DisaggregationMode.PREFILL:
        await init(runtime, config)
    else:
        await init_prefill(runtime, config)


async def init(runtime: DistributedRuntime, config: Config):
    # We use a request handler factory pattern here
    # depending on the disaggregation mode and pass in
    # the correct arguments all the way through

    server_args, dynamo_args = config.server_args, config.dynamo_args
    component = runtime.namespace(dynamo_args.namespace).component(
        dynamo_args.component
    )
    await component.create_service()

    engine = sgl.Engine(config.server_args)

    generate_endpoint = component.endpoint(dynamo_args.endpoint)

    # TODO: think about implementing DisaggregationStrategy for P->D
    if config.serving_mode == DisaggregationMode.DECODE:
        logging.info("Initializing prefill client")
        prefill_client = (
            await runtime.namespace(config.dynamo_args.namespace)
            .component("prefill")  # todo naming? from "next field?"
            .endpoint("generate")
            .client()
        )

    publisher, task = setup_sgl_metrics(component, engine)

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
