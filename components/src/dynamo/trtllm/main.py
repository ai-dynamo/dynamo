# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Callable, Coroutine

import uvloop

from dynamo.common.utils.graceful_shutdown import install_signal_handlers
from dynamo.common.utils.runtime import create_runtime
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.trtllm.args import parse_args
from dynamo.trtllm.constants import DisaggregationMode
from dynamo.trtllm.workers import init_worker

configure_dynamo_logging()
shutdown_endpoints: list = []

# Maximum time (seconds) to wait for in-flight requests to drain during shutdown.
_DRAIN_TIMEOUT_S = 30.0
_DRAIN_POLL_INTERVAL_S = 0.5


def _make_drain_callback(
    engine_holder: list,
) -> Callable[[], Coroutine]:
    """Create a drain callback that polls the TRT-LLM engine until idle.

    The engine_holder is a mutable list populated by init_llm_worker once the
    engine is ready.  If it is still empty when the signal fires (engine not yet
    initialized), draining is skipped.

    Returns None when the worker is not a prefill worker (drain is unnecessary).
    The caller checks disaggregation_mode *before* calling this helper.
    """

    async def _drain_in_flight_requests():
        if not engine_holder:
            logging.info("Engine not yet initialized; skipping drain")
            return

        engine = engine_holder[0]
        logging.info(
            "Draining in-flight requests (timeout=%.1fs) to allow "
            "NIXL KV transfers to complete before GPU memory is freed",
            _DRAIN_TIMEOUT_S,
        )
        deadline = asyncio.get_running_loop().time() + _DRAIN_TIMEOUT_S
        while asyncio.get_running_loop().time() < deadline:
            try:
                stats_iter = engine.llm.get_stats_async(timeout=2)
                stat = await anext(stats_iter)
                active = stat.get("numActiveRequests", 0)
                queued = stat.get("numQueuedRequests", 0)
                total = active + queued
                if total == 0:
                    logging.info("All in-flight requests drained")
                    return
                logging.info(
                    "Waiting for %d in-flight request(s) to complete "
                    "(active=%d, queued=%d)",
                    total,
                    active,
                    queued,
                )
            except Exception as e:
                # get_stats_async may fail if engine is already partially torn down
                logging.debug("Stats poll failed during drain: %s", e)
            await asyncio.sleep(_DRAIN_POLL_INTERVAL_S)

        logging.warning(
            "Drain timeout (%.1fs) reached; proceeding with shutdown. "
            "Some NIXL transfers may still be in flight.",
            _DRAIN_TIMEOUT_S,
        )

    return _drain_in_flight_requests


async def worker():
    config = parse_args()

    shutdown_event = asyncio.Event()
    runtime, loop = create_runtime(
        discovery_backend=config.discovery_backend,
        request_plane=config.request_plane,
        event_plane=config.event_plane,
    )

    # Only prefill workers need a drain callback.  When a prefill worker shuts
    # down, decode workers may still be reading its GPU memory via NIXL RDMA.
    # The drain callback waits for in-flight requests to finish so that GPU
    # memory is not freed while transfers are active (issue #7319).
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
