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
from dynamo.common.snapshot.restore_context import (
    parse_snapshot_restore_runtime_config,
    refresh_snapshot_restore_config,
)
from dynamo.common.utils.env import env_bool

from .args import Config
from .handlers import VllmEnginePauseController
from .worker_factory import EngineSetupResult

logger = logging.getLogger(__name__)


async def refresh_vllm_snapshot_restore_config(
    config: Config,
    argv: list[str] | None,
    *,
    restore_paused: bool,
) -> Config:
    config = await refresh_snapshot_restore_config(
        config,
        lambda: parse_snapshot_restore_runtime_config(argv),
    )
    if restore_paused:
        config.gms_shadow_mode = env_bool("DYN_VLLM_GMS_SHADOW_MODE")
    return config


async def prepare_snapshot_engine(
    config: Config,
    setup_vllm_engine: Callable[[Config], EngineSetupResult],
) -> EngineSnapshotController[EngineSetupResult] | None:
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

    engine = setup_vllm_engine(config)
    gc.collect()
    snapshot_controller = EngineSnapshotController(
        engine=engine,
        pause_controller=VllmEnginePauseController(engine[0]),
        snapshot_config=snapshot_config,
        pause_args=(None,),
    )
    if not await snapshot_controller.wait_for_restore():
        logger.info("vLLM snapshot captured successfully")
        os._exit(0)

    return snapshot_controller
