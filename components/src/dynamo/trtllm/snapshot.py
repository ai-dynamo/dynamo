# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import logging
import os
from typing import Any

from dynamo.trtllm.constants import DisaggregationMode, Modality

_EXTERNAL_MODEL_LOAD_FORMATS = {"gms"}


def _should_prefetch_model_for_snapshot(config: Any) -> bool:
    if os.path.exists(config.model):
        return False
    return config.load_format not in _EXTERNAL_MODEL_LOAD_FORMATS


def _create_runtime(
    discovery_backend: str,
    request_plane: str,
    event_plane: str | None,
) -> tuple[Any, Any]:
    from dynamo.common.utils.runtime import create_runtime as _create_runtime

    return _create_runtime(
        discovery_backend=discovery_backend,
        request_plane=request_plane,
        event_plane=event_plane,
    )


def _create_engine_snapshot_controller(
    engine: Any,
    pause_controller: Any,
    snapshot_config: Any,
) -> Any:
    from dynamo.common.snapshot.lifecycle import EngineSnapshotController

    return EngineSnapshotController(
        engine=engine,
        pause_controller=pause_controller,
        snapshot_config=snapshot_config,
    )


async def _refresh_snapshot_restore_runtime_config(
    config: Any,
    argv: list[str] | None,
) -> Any:
    from dynamo.common.snapshot.restore_context import (
        parse_snapshot_restore_runtime_config,
        refresh_snapshot_restore_config,
    )

    return await refresh_snapshot_restore_config(
        config,
        lambda: parse_snapshot_restore_runtime_config(argv),
    )


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

    def __init__(
        self,
        snapshot_config: Any,
        argv: list[str] | None = None,
    ) -> None:
        self._snapshot_config = snapshot_config
        self._argv = list(argv) if argv is not None else None
        self._runtime: Any | None = None

    async def snapshot_before_endpoint(self, engine: Any, config: Any) -> None:
        if self._runtime is not None:
            return

        logging.info(
            "Checkpoint mode enabled: TRT-LLM engine is initialized before "
            "Dynamo runtime creation"
        )
        pause_controller = _NoOpSnapshotPauseController()
        snapshot_controller = _create_engine_snapshot_controller(
            engine=engine,
            pause_controller=pause_controller,
            snapshot_config=self._snapshot_config,
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

        config = await _refresh_snapshot_restore_runtime_config(
            config,
            self._argv,
        )
        self._runtime, _ = _create_runtime(
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


def _validate_supported_snapshot_config(config: Any) -> None:
    unsupported = [
        label
        for supported, label in (
            (config.modality == Modality.TEXT, f"modality={config.modality.value}"),
            (
                config.disaggregation_mode == DisaggregationMode.AGGREGATED,
                f"disaggregation_mode={config.disaggregation_mode.value}",
            ),
            (not config.encode_endpoint, "--encode-endpoint"),
            (not config.frontend_decoding, "--frontend-decoding"),
            (
                config.tensor_parallel_size == 1,
                f"tensor_parallel_size={config.tensor_parallel_size}",
            ),
            (
                config.pipeline_parallel_size == 1,
                f"pipeline_parallel_size={config.pipeline_parallel_size}",
            ),
            (
                config.gpus_per_node in (None, 1),
                f"gpus_per_node={config.gpus_per_node}",
            ),
            (not config.has_connector("kvbm"), "--connector kvbm"),
        )
        if not supported
    ]

    if unsupported:
        raise ValueError(
            "TRT-LLM Dynamo Snapshot currently supports only the single-GPU "
            "aggregated text worker path. Unsupported snapshot setting(s): "
            + ", ".join(unsupported)
        )
