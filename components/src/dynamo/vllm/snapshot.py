# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gc
import logging
import os
from collections.abc import Callable
from typing import Any

from dynamo.common.configuration.utils import split_served_model_names
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


class _ReleaseOnlyKvCache:
    """Engine client that discards only the KV cache before a snapshot.

    Snapshot captures the weights on the GPU, so they need no host copy.
    """

    def __init__(self, engine_client: Any):
        self._engine_client = engine_client
        # vLLM 0.33 (enterprise image) has no checkpoint_state, and its
        # checkpoint_prepare expects a released KV cache; remove once it ships 0.34.
        self._has_checkpoint_state = hasattr(engine_client, "checkpoint_state")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine_client, name)

    async def sleep(self, *_args: object) -> None:
        # From vLLM 0.34, checkpoint_prepare releases the KV cache itself and
        # refuses an engine that is already asleep.
        if not self._has_checkpoint_state:
            await self._engine_client.release_kv_cache_memory()

    async def checkpoint_prepare(self) -> None:
        refusal = await self._engine_client.checkpoint_prepare()
        if refusal is not None:
            raise RuntimeError(f"vLLM refused to prepare for a checkpoint: {refusal}")
        if not self._has_checkpoint_state:
            return
        state, error = await self._engine_client.checkpoint_state()
        while state == "preparing":
            await asyncio.sleep(0.05)
            state, error = await self._engine_client.checkpoint_state()
        if state != "prepared":
            raise RuntimeError(
                f"vLLM failed to prepare for a checkpoint ({state}): {error}"
            )


class _GmsCheckpoint:
    """Engine client that checkpoints a GMS engine after it sleeps.

    GMS restores weights outside the checkpoint, so they are released by
    sleeping. vLLM 0.34's checkpoint_prepare refuses a sleeping engine, so run
    the worker step that 0.33's checkpoint_prepare ran.
    """

    def __init__(self, engine_client: Any):
        self._engine_client = engine_client

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine_client, name)

    async def checkpoint_prepare(self) -> None:
        await self._engine_client.collective_rpc("checkpoint_prepare")

    async def checkpoint_restore(self) -> None:
        await self._engine_client.collective_rpc("checkpoint_restore")


def apply_restore_served_model_name(config: Config) -> None:
    """Serve the names the restore Pod sets in ``DYN_SERVED_MODEL_NAME``.

    Like ``--served-model-name``, the first name is primary and the rest are
    aliases. Unset keeps the names the snapshot was captured with.
    """
    served_names = split_served_model_names(os.environ.get("DYN_SERVED_MODEL_NAME"))
    if served_names:
        config.served_model_name, *config.served_model_aliases = served_names


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
    engine_client: _ReleaseOnlyKvCache | _GmsCheckpoint
    if (
        getattr(config.engine_args, "load_format", None) != "gms"
        and os.environ.get("DYN_GMS_USE_V1") != "true"
    ):
        engine_client = _ReleaseOnlyKvCache(engine[0])
    else:
        engine_client = _GmsCheckpoint(engine[0])

    gc.collect()
    snapshot_controller = EngineSnapshotController(
        engine=(engine, stat_logger_factory),
        pause_controller=VllmEnginePauseController(
            engine_client,
            prepare_for_process_checkpoint=True,
        ),
        snapshot_config=snapshot_config,
        pause_args=(None,),
    )
    if not await snapshot_controller.wait_for_restore():
        logger.info("vLLM snapshot captured successfully")
        os._exit(0)

    return snapshot_controller
