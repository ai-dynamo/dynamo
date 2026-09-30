# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import logging
import os
from collections.abc import Callable

from dynamo.common.snapshot.lifecycle import (
    EngineSnapshotController,
    SnapshotConfig,
    configure_snapshot_capture_env,
)

from .args import Config
from .handlers import VllmEnginePauseController
from .publisher import StatLoggerFactory
from .worker_factory import EngineSetupResult, SnapshotEngineSetupResult

logger = logging.getLogger(__name__)


async def prepare_snapshot_engine(
    config: Config,
    setup_vllm_engine: Callable[..., EngineSetupResult],
) -> EngineSnapshotController[SnapshotEngineSetupResult] | None:
    snapshot_config = SnapshotConfig.from_env()
    if snapshot_config is None:
        return None

    if config.headless:
        raise ValueError(
            "--headless is incompatible with snapshot mode "
            "(DYN_SNAPSHOT_CONTROL_DIR is set). "
            "Remove --headless or unset DYN_SNAPSHOT_CONTROL_DIR."
        )

    configure_snapshot_capture_env()
    logger.info("Snapshot mode enabled (watcher-driven signals)")
    config.engine_args.enable_sleep_mode = True

    stat_logger_factory = StatLoggerFactory(
        endpoint=None,
        embedding_worker=config.embedding_worker,
    )
    engine = setup_vllm_engine(config, stat_logger_factory)
    # Decide before the first pause: reaching this at pause time would raise
    # after sleep() had already released the engine's memory.
    checkpoint_hooks = all(
        hasattr(engine[0], hook)
        for hook in ("checkpoint_prepare", "checkpoint_restore")
    )
    if not checkpoint_hooks:
        logger.warning(
            "This vLLM build has no AsyncLLM.checkpoint_prepare/checkpoint_restore; "
            "snapshotting without communicator checkpointing. "
            "Requires vLLM 0.27.0 or newer."
        )

    gc.collect()
    snapshot_controller = EngineSnapshotController(
        engine=(engine, stat_logger_factory),
        pause_controller=VllmEnginePauseController(
            engine[0],
            prepare_for_process_checkpoint=checkpoint_hooks,
        ),
        snapshot_config=snapshot_config,
        pause_args=(None,),
    )
    if not await snapshot_controller.wait_for_restore():
        logger.info("vLLM snapshot captured successfully")
        os._exit(0)

    return snapshot_controller
