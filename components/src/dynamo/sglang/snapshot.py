# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo Snapshot integration for SGLang workers."""

import gc
import logging
import time

import sglang as sgl

from dynamo.common.snapshot.lifecycle import (
    EngineSnapshotController,
    SnapshotConfig,
    configure_snapshot_capture_env,
)

from .pause import SGLangEnginePauseController

logger = logging.getLogger(__name__)


async def prepare_snapshot_engine(
    server_args,
) -> EngineSnapshotController[sgl.Engine] | None:
    """Single entry point for Dynamo Snapshot integration.

    Must be called BEFORE runtime creation so the engine can be snapshotted
    without active NATS/etcd connections.

    Returns:
        None when not in snapshot mode.
        A snapshot controller when restore completed and the caller should use
        the restored engine.

        If snapshotting completed successfully, this function exits the
        process with status 0.
    """
    snapshot_config = SnapshotConfig.from_env()
    if snapshot_config is None:
        return None

    configure_snapshot_capture_env()
    logger.info("Snapshot mode enabled (watcher-driven signals)")

    # Enable memory_saver so GPU memory can be released for CRIU.
    # When using GMS, weights use VA-stable unmap/remap (no CPU backup); GMS
    # forbids enable_weights_cpu_backup. Otherwise use CPU backup for weights.
    server_args.enable_memory_saver = True
    try:
        from gpu_memory_service.integrations.sglang import is_gms_active

        _using_gms = is_gms_active()
    except ImportError:
        _using_gms = False
    if not _using_gms:
        server_args.enable_weights_cpu_backup = True

    start_time = time.time()
    engine = sgl.Engine(server_args=server_args)
    logger.info(
        f"SGLang engine loaded in {time.time() - start_time:.2f}s (snapshot mode)"
    )

    gc.collect()

    snapshot_controller = EngineSnapshotController(
        engine=engine,
        pause_controller=SGLangEnginePauseController(engine),
        snapshot_config=snapshot_config,
    )
    if not await snapshot_controller.wait_for_restore():
        raise SystemExit(0)

    return snapshot_controller
