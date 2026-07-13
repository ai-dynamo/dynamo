# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import logging
import os
from collections.abc import Callable
from typing import Any

from dynamo.common.backend import PreRuntimeOutcome
from dynamo.common.model_fetch import fetch_model
from dynamo.common.snapshot.lifecycle import (
    EngineSnapshotController,
    SnapshotConfig,
    configure_snapshot_capture_env,
    unified_snapshot_outcome,
)
from dynamo.llm.exceptions import Cancelled, EngineShutdown, InvalidArgument

from .args import Config
from .handlers import VllmEnginePauseController
from .worker_factory import EngineSetupResult

logger = logging.getLogger(__name__)

_EXTERNAL_MODEL_LOAD_FORMATS = {"modelexpress", "mx"}


async def prepare_unified_snapshot(
    engine: Any,
    *,
    config: Config | None,
    argv: list[str] | None,
    context: Any,
) -> PreRuntimeOutcome:
    """Prepare a unified vLLM engine before DistributedRuntime creation."""

    snapshot_config = SnapshotConfig.from_env()
    if snapshot_config is None:
        return PreRuntimeOutcome.continue_startup()
    if config is None:
        raise InvalidArgument(
            "vLLM snapshot preparation is missing parsed configuration"
        )
    if config.headless:
        raise InvalidArgument(
            "--headless is incompatible with snapshot mode "
            "(DYN_SNAPSHOT_CONTROL_DIR is set)"
        )

    try:
        load_format = getattr(config.engine_args, "load_format", None)
        if (
            not os.path.exists(config.model)
            and load_format not in _EXTERNAL_MODEL_LOAD_FORMATS
        ):
            await fetch_model(config.model)

        configure_snapshot_capture_env()
        config.engine_args.enable_sleep_mode = True
        logger.info("Unified vLLM snapshot mode enabled")
        await engine._initialize_engine()
        if engine._pause_controller is None:
            raise RuntimeError("vLLM snapshot pause controller was not initialized")
    except (Cancelled, InvalidArgument):
        raise
    except Exception as exc:
        raise EngineShutdown(f"vLLM snapshot_prepare failed: {exc}") from exc
    gc.collect()

    controller = EngineSnapshotController(
        engine=engine,
        pause_controller=engine._pause_controller,
        snapshot_config=snapshot_config,
        pause_args=(None,),
    )
    return await unified_snapshot_outcome(
        controller,
        argv=argv,
        context=context,
        backend_name="vLLM",
    )


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
        raise SystemExit(0)

    return snapshot_controller
