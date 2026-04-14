# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo Snapshot integration for TRT-LLM workers."""

import gc
import logging
import time
from collections.abc import Awaitable, Callable

from dynamo.common.utils.snapshot import CheckpointConfig, EngineSnapshotController
from dynamo.trtllm.engine import TensorRTLLMEngine

from .request_handlers.handler_base import TRTLLMEngineQuiesceController

logger = logging.getLogger(__name__)


async def prepare_snapshot_engine(
    setup_engine: Callable[[], Awaitable[TensorRTLLMEngine]],
) -> EngineSnapshotController[TensorRTLLMEngine] | None:
    """Single entry point for Dynamo Snapshot integration with TRT-LLM.

    Must be called BEFORE runtime creation so the engine can be checkpointed
    without active NATS/etcd connections.

    sleep_config is always enabled (set unconditionally in llm_worker.py),
    so the caller does not need to configure it separately.

    Returns:
        None when not in checkpoint mode.
        A snapshot controller when restore completed and the caller should use
        the restored engine.

        If checkpointing completed successfully, this function exits the
        process with status 0.
    """
    checkpoint_config = CheckpointConfig.from_env()
    if checkpoint_config is None:
        return None

    logger.info("Checkpoint mode enabled (watcher-driven signals)")

    start_time = time.time()
    engine = await setup_engine()
    logger.info(
        "TRT-LLM engine loaded in %.2fs (checkpoint mode)",
        time.time() - start_time,
    )

    gc.collect()

    snapshot_controller = EngineSnapshotController(
        engine=engine,
        quiesce_controller=TRTLLMEngineQuiesceController(engine),
        checkpoint_config=checkpoint_config,
    )
    if not await snapshot_controller.wait_for_restore():
        raise SystemExit(0)

    return snapshot_controller
