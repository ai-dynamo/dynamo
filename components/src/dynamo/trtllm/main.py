# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os

# Configure TLLM_LOG_LEVEL before importing tensorrt_llm
# This must happen before any tensorrt_llm imports
if "TLLM_LOG_LEVEL" not in os.environ and os.getenv(
    "DYN_SKIP_TRTLLM_LOG_FORMATTING"
) not in ("1", "true", "TRUE"):
    # This import is safe because it doesn't trigger tensorrt_llm imports
    from dynamo.runtime.logging import map_dyn_log_to_tllm_level

    dyn_log = os.environ.get("DYN_LOG", "info")
    tllm_level = map_dyn_log_to_tllm_level(dyn_log)
    os.environ["TLLM_LOG_LEVEL"] = tllm_level
import uvloop

from dynamo.common.utils.graceful_shutdown import install_signal_handlers
from dynamo.common.utils.runtime import create_runtime
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.trtllm.args import parse_args
from dynamo.trtllm.workers import init_worker

configure_dynamo_logging()
shutdown_endpoints: list = []


async def worker():
    config = parse_args()

    shutdown_event = asyncio.Event()
    runtime, loop = create_runtime(
        discovery_backend=config.discovery_backend,
        request_plane=config.request_plane,
        event_plane=config.event_plane,
        use_kv_events=config.use_kv_events,
    )

    install_signal_handlers(loop, runtime, shutdown_endpoints, shutdown_event)

    logging.info(f"Initializing the worker with config: {config}")
    await init_worker(runtime, config, shutdown_event, shutdown_endpoints)


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
