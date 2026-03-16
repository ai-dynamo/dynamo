# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo Snapshot integration for SGLang workers."""

import logging
import time

import sglang as sgl

from dynamo.common.utils.snapshot import EngineSnapshotController, get_checkpoint_config

from .request_handlers.handler_base import SGLangEngineQuiesceController

logger = logging.getLogger(__name__)

async def prepare_snapshot_engine(
    server_args, dynamo_args
) -> tuple[bool, EngineSnapshotController[sgl.Engine] | None]:
    """Single entry point for Dynamo Snapshot integration.

    Must be called BEFORE runtime creation so the engine can be checkpointed
    without active NATS/etcd connections.

    Returns:
        (should_exit, snapshot_controller) where:
        - (True, None): caller should return immediately (checkpoint already
          exists, or checkpoint completed successfully).
        - (False, None): not in checkpoint mode — cold-start normally.
        - (False, snapshot_controller): restore completed — caller should use
          snapshot_controller.engine.
    """
    should_exit, checkpoint_cfg = get_checkpoint_config()
    if should_exit:
        return True, None

    if checkpoint_cfg is None:
        return False, None

    logger.info("Checkpoint mode enabled (watcher-driven signals)")

    # Enable memory_saver + weights CPU backup so weights survive CRIU
    # (mirrors vLLM's enable_sleep_mode = True)
    server_args.enable_memory_saver = True
    server_args.enable_weights_cpu_backup = True

    start_time = time.time()
    engine = sgl.Engine(server_args=server_args)
    logger.info(
        f"SGLang engine loaded in {time.time() - start_time:.2f}s (checkpoint mode)"
    )

    snapshot_controller = EngineSnapshotController(
        engine=engine,
        quiesce_controller=SGLangEngineQuiesceController(engine),
        checkpoint_config=checkpoint_cfg,
    )
    if not await snapshot_controller.wait_for_restore():
        return True, None

    return False, snapshot_controller
