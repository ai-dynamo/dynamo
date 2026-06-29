# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gc
import logging
import sys
from collections.abc import Callable, Coroutine
from typing import Any

import uvloop

from dynamo.common.model_fetch import fetch_model
from dynamo.common.snapshot.lifecycle import (
    SnapshotConfig,
    configure_snapshot_capture_env,
)
from dynamo.common.utils.graceful_shutdown import install_signal_handlers
from dynamo.common.utils.runtime import create_runtime
from dynamo.runtime.logging import configure_dynamo_logging, get_bool_env_var
from dynamo.trtllm.args import parse_args
from dynamo.trtllm.constants import DisaggregationMode
from dynamo.trtllm.snapshot import (
    _should_prefetch_model_for_snapshot,
    _SnapshotRuntimeProxy,
    _validate_supported_snapshot_config,
)
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


async def worker(argv: list[str] | None = None):
    if argv is None:
        argv = sys.argv[1:]
    config = parse_args(argv)

    if get_bool_env_var("DYN_TRTLLM_SERVER_DISABLE_GC") or get_bool_env_var(
        "TRTLLM_SERVER_DISABLE_GC"
    ):
        gc.disable()
        logging.info(
            "Python cyclic GC disabled (DYN_TRTLLM_SERVER_DISABLE_GC or TRTLLM_SERVER_DISABLE_GC is set)"
        )

    shutdown_event = asyncio.Event()
    snapshot_config = SnapshotConfig.from_env()
    runtime: Any
    if snapshot_config is None:
        runtime, loop = create_runtime(
            discovery_backend=config.discovery_backend,
            request_plane=config.request_plane,
            event_plane=config.event_plane,
        )
    else:
        _validate_supported_snapshot_config(config)
        # Snapshot mode forces HF_HUB_OFFLINE=1 before TRT-LLM engine creation
        # so Hugging Face sockets are not captured. Warm the cache first when
        # TRT-LLM will load a normal HF model ID; external loaders such as GMS
        # own model acquisition themselves.
        if _should_prefetch_model_for_snapshot(config):
            await fetch_model(config.model)
        configure_snapshot_capture_env()
        # vLLM/SGLang snapshot paths build the engine before creating a runtime.
        # TRT-LLM's engine is built inside init_worker(), so pass a guarded
        # runtime proxy through that shared path and materialize the real runtime
        # only after the snapshot hook restores.
        runtime = _SnapshotRuntimeProxy(snapshot_config, argv=argv)
        loop = asyncio.get_running_loop()

    # Only prefill workers need a drain callback.  When a prefill worker shuts
    # down, decode workers may still be reading its GPU memory via NIXL RDMA.
    # The drain callback waits for in-flight requests to finish so that GPU
    # memory is not freed while transfers are active (issue #7319).
    engine_holder: list = []
    publisher_holder: list = []
    drain_callback = None
    if config.disaggregation_mode == DisaggregationMode.PREFILL:
        drain_callback = _make_drain_callback(engine_holder)

    async def cleanup_engine() -> None:
        if publisher_holder:
            await publisher_holder[0].cleanup()
            logging.info("TRT-LLM publisher cleanup complete")

        if not engine_holder:
            logging.info("TRT-LLM engine not initialized; skipping cleanup")
            return

        await engine_holder[0].cleanup()
        logging.info("TRT-LLM engine cleanup complete")

    install_signal_handlers(
        loop,
        runtime,
        shutdown_endpoints,
        shutdown_event,
        drain_callback=drain_callback,
        cleanup_callback=cleanup_engine,
    )

    logging.info(f"Initializing the worker with config: {config}")
    await init_worker(
        runtime,
        config,
        shutdown_event,
        shutdown_endpoints,
        engine_holder=engine_holder,
        publisher_holder=publisher_holder,
    )


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
