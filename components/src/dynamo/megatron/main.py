# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo worker entrypoint for the Megatron backend

Connects to a separately-launched Megatron coordinator over ZMQ, registers an
endpoint with the Dynamo distributed runtime, and proxies streaming requests
to Megatron via the :class:`MegatronEngineClient`.
"""

from __future__ import annotations

import asyncio
import logging

import uvloop

from dynamo.common.utils.runtime import create_runtime
from dynamo.llm import (
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    WorkerType,
    register_model,
)
from dynamo.runtime.logging import configure_dynamo_logging

from dynamo.megatron.args import parse_args
from dynamo.megatron.engine_client import MegatronEngineClient
from dynamo.megatron.handlers import DecodeWorkerHandler

configure_dynamo_logging()
logger = logging.getLogger(__name__)


async def worker() -> None:
    config = parse_args()

    runtime, _loop = create_runtime(
        discovery_backend=config.discovery_backend,
        request_plane=config.request_plane,
        event_plane=config.event_plane,
    )

    generate_endpoint = runtime.endpoint(
        f"{config.namespace}.{config.component}.{config.endpoint}"
    )

    engine_client = MegatronEngineClient(config.coordinator_addr)
    engine_client.start()

    handler = DecodeWorkerHandler(engine_client)

    runtime_config = ModelRuntimeConfig()
    runtime_config.total_kv_blocks = 0
    runtime_config.max_num_seqs = 1
    runtime_config.max_num_batched_tokens = config.context_length

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(handler.generate, graceful_shutdown=True),
            register_model(
                ModelInput.Tokens,
                ModelType.Chat,
                generate_endpoint,
                config.model,
                config.served_model_name,
                context_length=config.context_length,
                runtime_config=runtime_config,
                worker_type=WorkerType.Aggregated,
                needs=[],
            ),
        )
    finally:
        engine_client.stop()
        runtime.shutdown()


def main() -> None:
    uvloop.run(worker())


if __name__ == "__main__":
    main()
