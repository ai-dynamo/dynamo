# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gc
import logging
import sys
from typing import Any

import uvloop

from dynamo.common.model_fetch import fetch_model
from dynamo.common.snapshot.lifecycle import (
    SnapshotConfig,
    configure_snapshot_capture_env,
)
from dynamo.common.utils.runtime import create_runtime
from dynamo.common.utils.worker_shutdown import WorkerShutdown
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
    shutdown_endpoints: list = []
    snapshot_config = SnapshotConfig.from_env()
    runtime: Any
    if snapshot_config is None:
        runtime, _ = create_runtime(
            discovery_backend=config.discovery_backend,
            request_plane=config.request_plane,
            event_plane=config.event_plane,
            response_plane=config.response_plane,
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

    shutdown = WorkerShutdown(
        runtime,
        shutdown_endpoints,
        shutdown_event,
        prefill=config.disaggregation_mode == DisaggregationMode.PREFILL,
    )

    logging.info(f"Initializing the worker with config: {config}")
    await shutdown.run(
        init_worker(
            runtime,
            config,
            shutdown_event,
            shutdown_endpoints,
            shutdown=shutdown,
        )
    )


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
