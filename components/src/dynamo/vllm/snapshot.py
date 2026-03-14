# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Callable

from dynamo.common.utils.snapshot import (
    EngineSnapshotController,
    get_checkpoint_config,
)

from .args import Config
from .handlers import VllmEngineQuiesceController
from .worker_factory import EngineSetupResult

logger = logging.getLogger(__name__)


async def prepare_snapshot_engine(
    config: Config,
    setup_vllm_engine: Callable[[Config], EngineSetupResult],
) -> tuple[bool, EngineSnapshotController[EngineSetupResult] | None]:
    should_exit, checkpoint_config = get_checkpoint_config()
    if should_exit:
        return True, None

    if checkpoint_config is None:
        return False, None

    if config.headless:
        raise ValueError(
            "--headless is incompatible with checkpoint mode "
            "(DYN_CHECKPOINT_SIGNAL_FILE is set). "
            "Remove --headless or unset DYN_CHECKPOINT_SIGNAL_FILE."
        )

    logger.info("Checkpoint mode enabled (watcher-driven signals)")
    config.engine_args.enable_sleep_mode = True

    engine = setup_vllm_engine(config)
    snapshot_controller = EngineSnapshotController(
        engine=engine,
        quiesce_controller=VllmEngineQuiesceController(engine[0]),
        checkpoint_config=checkpoint_config,
        quiesce_args=(None,),
    )
    if not await snapshot_controller.wait_for_restore():
        return True, None

    return False, snapshot_controller
