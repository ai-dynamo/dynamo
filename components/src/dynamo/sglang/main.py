# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import sys

import uvloop

from dynamo.common.config_dump import dump_config
from dynamo.common.constants import DisaggregationMode
from dynamo.common.utils.runtime import create_runtime
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sglang.args import parse_args
from dynamo.sglang.init_diffusion import (
    init_image_diffusion,
    init_llm_diffusion,
    init_video_diffusion,
)
from dynamo.sglang.init_embedding import init_embedding
from dynamo.sglang.init_llm import init_decode, init_prefill
from dynamo.sglang.init_multimodal import (
    init_multimodal_encode_worker,
    init_multimodal_prefill_worker,
    init_multimodal_processor,
    init_multimodal_worker,
)
from dynamo.sglang.shutdown import install_graceful_shutdown

configure_dynamo_logging()


async def worker():
    config = await parse_args(sys.argv[1:])
    dump_config(config.dynamo_args.dump_config_to, config)

    if config.server_args.load_format == "gms":
        from gpu_memory_service.integrations.sglang import setup_gms

        config.server_args.load_format = setup_gms(config.server_args)

    dynamo_args = config.dynamo_args
    shutdown_event = asyncio.Event()
    shutdown_endpoints: list = []
    runtime, loop = create_runtime(
        discovery_backend=dynamo_args.discovery_backend,
        request_plane=dynamo_args.request_plane,
        event_plane=dynamo_args.event_plane,
        use_kv_events=dynamo_args.use_kv_events,
    )

    run_deferred_handlers = install_graceful_shutdown(
        loop, runtime, shutdown_endpoints, shutdown_event
    )
    logging.info(
        "Signal handlers set up for graceful shutdown "
        "(discovery unregister + grace period, with chaining)"
    )

    if config.dynamo_args.image_diffusion_worker:
        await init_image_diffusion(
            runtime, config, shutdown_endpoints, run_deferred_handlers
        )
    elif config.dynamo_args.video_diffusion_worker:
        await init_video_diffusion(
            runtime, config, shutdown_endpoints, run_deferred_handlers
        )
    elif config.dynamo_args.embedding_worker:
        await init_embedding(
            runtime,
            config,
            shutdown_event,
            shutdown_endpoints,
            run_deferred_handlers,
        )
    elif config.dynamo_args.multimodal_processor:
        await init_multimodal_processor(
            runtime,
            config,
            shutdown_event,
            shutdown_endpoints,
            run_deferred_handlers,
        )
    elif config.dynamo_args.multimodal_encode_worker:
        await init_multimodal_encode_worker(
            runtime,
            config,
            shutdown_event,
            shutdown_endpoints,
            run_deferred_handlers,
        )
    elif config.dynamo_args.multimodal_worker:
        if config.serving_mode != DisaggregationMode.PREFILL:
            await init_multimodal_worker(
                runtime,
                config,
                shutdown_event,
                shutdown_endpoints,
                run_deferred_handlers,
            )
        else:
            await init_multimodal_prefill_worker(
                runtime,
                config,
                shutdown_event,
                shutdown_endpoints,
                run_deferred_handlers,
            )
    elif config.dynamo_args.llm_diffusion_worker:
        await init_llm_diffusion(
            runtime,
            config,
            shutdown_event,
            shutdown_endpoints,
            run_deferred_handlers,
        )
    elif config.serving_mode != DisaggregationMode.PREFILL:
        await init_decode(
            runtime,
            config,
            shutdown_event,
            shutdown_endpoints,
            run_deferred_handlers,
        )
    else:
        await init_prefill(
            runtime,
            config,
            shutdown_event,
            shutdown_endpoints,
            run_deferred_handlers,
        )


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
