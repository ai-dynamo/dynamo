# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import logging
from collections.abc import Callable
from typing import Any

from dynamo.common.snapshot.lifecycle import (
    EngineSnapshotController,
    SnapshotConfig,
    configure_snapshot_capture_env,
)

from .args import Config
from .handlers import VllmEnginePauseController
from .snapshot_worker import (
    SNAPSHOT_WORKER_CLASS,
    configure_snapshot_worker,
)
from .worker_factory import EngineSetupResult

logger = logging.getLogger(__name__)


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
    snapshot_worker_configured = configure_snapshot_worker(config)

    engine = setup_vllm_engine(config)
    if snapshot_worker_configured:
        await _verify_snapshot_worker_identity(engine[0])

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


async def _verify_snapshot_worker_identity(engine_client: Any) -> None:
    try:
        identities = await engine_client.collective_rpc(
            "snapshot_worker_identity",
            timeout=30,
        )
    except Exception as exc:
        raise RuntimeError(
            "Dynamo vLLM Snapshot worker could not verify runtime identity "
            "via collective_rpc('snapshot_worker_identity')."
        ) from exc

    logger.info("vLLM SnapshotWorker runtime identities: %s", identities)
    if not identities:
        raise RuntimeError(
            "Dynamo vLLM Snapshot worker identity check returned no rank results."
        )

    for identity in identities:
        if not isinstance(identity, dict):
            raise RuntimeError(
                "Dynamo vLLM Snapshot worker identity check returned an invalid "
                f"rank result: {identity!r}."
            )
        if identity.get("qualified_class") != SNAPSHOT_WORKER_CLASS:
            raise RuntimeError(
                "Dynamo vLLM Snapshot worker expected "
                f"{SNAPSHOT_WORKER_CLASS}, but rank={identity.get('rank')!r} "
                f"local_rank={identity.get('local_rank')!r} "
                f"pid={identity.get('pid')!r} reported "
                f"qualified_class={identity.get('qualified_class')!r}."
            )
