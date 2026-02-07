# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import signal

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

from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.trtllm.utils.trtllm_utils import cmd_line_args
from dynamo.trtllm.workers import init_worker

configure_dynamo_logging()


async def graceful_shutdown(runtime, shutdown_event):
    logging.info("Received shutdown signal, shutting down DistributedRuntime")
    shutdown_event.set()
    runtime.shutdown()
    logging.info("DistributedRuntime shutdown complete")


async def worker():
    config = cmd_line_args()

    loop = asyncio.get_running_loop()
    # Create shutdown event
    shutdown_event = asyncio.Event()

    # Set DYN_EVENT_PLANE environment variable based on config
    os.environ["DYN_EVENT_PLANE"] = config.event_plane

    # NATS is needed when:
    # 1. Request plane is NATS, OR
    # 2. Event plane is NATS AND use_kv_events is True
    enable_nats = config.request_plane == "nats" or (
        config.event_plane == "nats" and config.use_kv_events
    )

    runtime = DistributedRuntime(
        loop, config.store_kv, config.request_plane, enable_nats
    )

    # Set up signal handler for graceful shutdown
    def signal_handler():
        # Schedule the shutdown coroutine instead of calling it directly
        asyncio.create_task(graceful_shutdown(runtime, shutdown_event))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logging.info("Signal handlers set up for graceful shutdown")

    await init_worker(runtime, config, shutdown_event)


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
