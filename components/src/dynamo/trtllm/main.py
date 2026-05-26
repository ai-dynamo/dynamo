# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gc
import logging
from typing import Callable, Coroutine

import uvloop

from dynamo.common.utils.drain import prefill_drain_context
from dynamo.common.utils.graceful_shutdown import install_signal_handlers
from dynamo.common.utils.runtime import create_runtime
from dynamo.runtime.logging import configure_dynamo_logging, get_bool_env_var
from dynamo.trtllm.args import parse_args
from dynamo.trtllm.constants import DisaggregationMode
from dynamo.trtllm.workers import init_worker

configure_dynamo_logging()
logger = logging.getLogger(__name__)
shutdown_endpoints: list = []

_DRAIN_POLL_INTERVAL_S = 0.5
# `get_stats_async` raises TimeoutError when no stats are fresh and
# StopAsyncIteration when the iterator is exhausted; both are benign
# "try again" signals. RuntimeError covers the test stub.
_BENIGN_POLL_EXC = (asyncio.TimeoutError, StopAsyncIteration, RuntimeError)


def _make_drain_callback(
    engine_holder: list,
) -> Callable[[], Coroutine]:
    """Legacy-path drain matching :meth:`TensorRTLLMEngine.drain`.

    ``engine_holder`` is populated by ``init_llm_worker`` once the engine
    is ready; if still empty when SIGTERM fires we skip the drain.
    """

    async def _drain_in_flight_requests():
        if not engine_holder:
            logger.info("Engine not yet initialized; skipping drain")
            return
        engine = engine_holder[0]
        async with prefill_drain_context(logger) as ctx:
            while not ctx.expired():
                poll_timeout_s = min(2.0, ctx.remaining_s())
                try:
                    stats_iter = engine.llm.get_stats_async(timeout=poll_timeout_s)
                    stat = await asyncio.wait_for(
                        anext(stats_iter), timeout=poll_timeout_s
                    )
                    active = stat.get("numActiveRequests", 0)
                    queued = stat.get("numQueuedRequests", 0)
                    ctx.heartbeat(active=active, queued=queued)
                    if active + queued == 0:
                        return
                except _BENIGN_POLL_EXC as e:
                    logger.debug("Stats poll failed during drain: %s", e)
                    ctx.heartbeat()
                await asyncio.sleep(min(_DRAIN_POLL_INTERVAL_S, ctx.remaining_s()))

    return _drain_in_flight_requests


async def worker():
    config = parse_args()

    if get_bool_env_var("DYN_TRTLLM_SERVER_DISABLE_GC") or get_bool_env_var(
        "TRTLLM_SERVER_DISABLE_GC"
    ):
        gc.disable()
        logging.info(
            "Python cyclic GC disabled (DYN_TRTLLM_SERVER_DISABLE_GC or TRTLLM_SERVER_DISABLE_GC is set)"
        )

    shutdown_event = asyncio.Event()
    runtime, loop = create_runtime(
        discovery_backend=config.discovery_backend,
        request_plane=config.request_plane,
        event_plane=config.event_plane,
    )

    # Only prefill workers need a drain callback.  When a prefill worker shuts
    # down, decode workers may still be reading its GPU memory via NIXL RDMA.
    # The drain callback waits for in-flight requests to finish so that GPU
    # memory is not freed while transfers are active.
    engine_holder: list = []
    drain_callback = None
    if config.disaggregation_mode == DisaggregationMode.PREFILL:
        drain_callback = _make_drain_callback(engine_holder)

    install_signal_handlers(
        loop,
        runtime,
        shutdown_endpoints,
        shutdown_event,
        drain_callback=drain_callback,
    )

    logging.info(f"Initializing the worker with config: {config}")
    await init_worker(
        runtime,
        config,
        shutdown_event,
        shutdown_endpoints,
        engine_holder=engine_holder,
    )


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
