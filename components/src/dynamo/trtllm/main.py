# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gc
import logging
import os
from collections.abc import Callable, Coroutine
from typing import Any

import uvloop

from dynamo.common.utils.graceful_shutdown import install_signal_handlers
from dynamo.common.utils.runtime import create_runtime
from dynamo.common.utils.snapshot import CheckpointConfig, EngineSnapshotController
from dynamo.runtime.logging import configure_dynamo_logging, get_bool_env_var
from dynamo.trtllm.args import parse_args
from dynamo.trtllm.constants import DisaggregationMode
from dynamo.trtllm.workers import init_worker

configure_dynamo_logging()
shutdown_endpoints: list = []

# Maximum time (seconds) to wait for in-flight requests to drain during shutdown.
_DRAIN_TIMEOUT_S = 30.0
_DRAIN_POLL_INTERVAL_S = 0.5


class _NoOpSnapshotPauseController:
    """Pause controller for pre-runtime TRT-LLM engine snapshots.

    The snapshot hook runs after the TRT-LLM engine has loaded and before the
    Dynamo runtime endpoint is created. There is no endpoint to drain or
    unregister at this point, so keep the TRT-LLM/CUDA allocations resident and
    only run Python GC before CRIU/cuda-checkpoint capture.
    """

    async def pause(self, *_args: object) -> bool:
        gc.collect()
        return True

    async def resume(self) -> bool:
        return True

    def mark_resumed(self) -> None:
        return None


class _SnapshotRuntimeProxy:
    """Delay Dynamo runtime creation until after TRT-LLM snapshot restore.

    TRT-LLM initializes CUDA/OpenMPI state while creating the engine. For
    snapshot mode we want that state captured, but we do not want the Dynamo
    runtime endpoint and its network sockets in the checkpoint. This proxy is
    passed through the normal worker initialization path and materializes the
    real runtime only after restore.
    """

    def __init__(self, checkpoint_config: CheckpointConfig) -> None:
        self._checkpoint_config = checkpoint_config
        self._runtime: Any | None = None

    async def snapshot_before_endpoint(self, engine: Any, config: Any) -> None:
        if self._runtime is not None:
            return

        logging.info(
            "Checkpoint mode enabled: TRT-LLM engine is initialized before "
            "Dynamo runtime creation"
        )
        pause_controller = _NoOpSnapshotPauseController()
        snapshot_controller = EngineSnapshotController(
            engine=engine,
            pause_controller=pause_controller,
            checkpoint_config=self._checkpoint_config,
        )

        # This is the checkpoint/restore synchronization point. It writes the
        # "ready-for-checkpoint" sentinel after the TRT-LLM engine is resident,
        # waits for the external snapshot agent to either capture or restore the
        # process, and returns True only in the restored process. The real
        # DistributedRuntime must not exist before this await, otherwise NATS,
        # etcd, and endpoint sockets would be captured with the engine state.
        restored = await snapshot_controller.wait_for_restore()
        if not restored:
            logging.info(
                "Initial TRT-LLM snapshot captured successfully; exiting "
                "without destroying the engine"
            )
            os._exit(0)

        config.namespace, config.discovery_backend = (
            snapshot_controller.reload_restore_identity(
                config.namespace,
                config.discovery_backend,
            )
        )
        self._runtime, _ = create_runtime(
            discovery_backend=config.discovery_backend,
            request_plane=config.request_plane,
            event_plane=config.event_plane,
        )
        logging.info("Dynamo runtime created after TRT-LLM snapshot restore")

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise RuntimeError(
                "Dynamo runtime is not available until the TRT-LLM snapshot "
                "hook has restored and created it"
            )
        return self._runtime

    def shutdown(self) -> None:
        if self._runtime is not None:
            self._runtime.shutdown()

    def __getattr__(self, name: str) -> Any:
        if name == "snapshot_before_endpoint":
            raise AttributeError(name)

        # Future DistributedRuntime methods should fail fast before restore
        # instead of accidentally creating runtime-owned network state in the
        # snapshot. Once snapshot_before_endpoint materializes the real runtime,
        # attribute access delegates to it for both existing and newly-added
        # runtime APIs.
        return getattr(self._require_runtime(), name)


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

    if get_bool_env_var("DYN_TRTLLM_SERVER_DISABLE_GC") or get_bool_env_var(
        "TRTLLM_SERVER_DISABLE_GC"
    ):
        gc.disable()
        logging.info(
            "Python cyclic GC disabled (DYN_TRTLLM_SERVER_DISABLE_GC or TRTLLM_SERVER_DISABLE_GC is set)"
        )

    shutdown_event = asyncio.Event()
    checkpoint_config = CheckpointConfig.from_env()
    runtime: Any
    if checkpoint_config is None:
        runtime, loop = create_runtime(
            discovery_backend=config.discovery_backend,
            request_plane=config.request_plane,
            event_plane=config.event_plane,
        )
    else:
        # vLLM/SGLang snapshot paths build the engine before creating a runtime.
        # TRT-LLM's engine is built inside init_worker(), so pass a guarded
        # runtime proxy through that shared path and materialize the real runtime
        # only after the snapshot hook restores.
        runtime = _SnapshotRuntimeProxy(checkpoint_config)
        loop = asyncio.get_running_loop()

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
