# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from dynamo.common.snapshot.lifecycle import (
    EngineSnapshotController,
    SnapshotConfig,
    configure_snapshot_capture_env,
)

from .flashinfer_snapshot import DYN_VLLM_REQUIRE_FLASHINFER_SNAPSHOT_RESOURCES
from .snapshot_worker_config import (
    DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER,
    SNAPSHOT_WORKER_CLASS,
    configure_flashinfer_snapshot_worker,
)

if TYPE_CHECKING:
    from .args import Config
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
    configured_snapshot_worker = configure_flashinfer_snapshot_worker(config)
    if (
        os.environ.get(DYN_VLLM_REQUIRE_FLASHINFER_SNAPSHOT_RESOURCES) == "1"
        and not configured_snapshot_worker
    ):
        raise ValueError(
            f"{DYN_VLLM_REQUIRE_FLASHINFER_SNAPSHOT_RESOURCES}=1 requires "
            f"{DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER}=1 so Dynamo installs the "
            "FlashInfer-aware Snapshot worker before vLLM engine startup."
        )

    engine = setup_vllm_engine(config)
    await verify_snapshot_worker_identity(engine[0])
    gc.collect()
    from .handlers import VllmEnginePauseController

    snapshot_controller = EngineSnapshotController(
        engine=engine,
        pause_controller=VllmEnginePauseController(engine[0]),
        snapshot_config=snapshot_config,
        pause_args=(None,),
    )
    if not await snapshot_controller.wait_for_restore():
        raise SystemExit(0)

    return snapshot_controller


async def verify_snapshot_worker_identity(engine_client: Any) -> None:
    if os.environ.get(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER) != "1":
        return

    try:
        identities = await engine_client.collective_rpc(
            "snapshot_worker_identity", timeout=30
        )
    except Exception as exc:
        raise RuntimeError(
            "DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER=1 could not verify "
            "SnapshotWorker runtime identity via collective_rpc"
        ) from exc

    logger.info("vLLM SnapshotWorker runtime identities: %s", identities)
    if not identities:
        raise RuntimeError(
            "DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER=1 collective_rpc returned no "
            "SnapshotWorker runtime identities."
        )

    for identity in identities:
        if not isinstance(identity, dict):
            raise RuntimeError(
                "DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER=1 received invalid "
                f"SnapshotWorker identity result: {identity!r}."
            )
        qualified_class = identity.get("qualified_class")
        if qualified_class != SNAPSHOT_WORKER_CLASS:
            raise RuntimeError(
                "DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER=1 expected "
                f"{SNAPSHOT_WORKER_CLASS}, but rank={identity.get('rank')!r} "
                f"local_rank={identity.get('local_rank')!r} "
                f"pid={identity.get('pid')!r} "
                f"reported module={identity.get('module')!r} "
                f"class={identity.get('class')!r} "
                f"qualified_class={qualified_class!r}."
            )

        if os.environ.get(DYN_VLLM_REQUIRE_FLASHINFER_SNAPSHOT_RESOURCES) == "1":
            resource_count = identity.get("flashinfer_resource_count")
            if (
                isinstance(resource_count, bool)
                or not isinstance(resource_count, int)
                or resource_count <= 0
            ):
                raise RuntimeError(
                    f"{DYN_VLLM_REQUIRE_FLASHINFER_SNAPSHOT_RESOURCES}=1 requires "
                    "SnapshotWorker FlashInfer resources before ready-for-snapshot, "
                    f"but rank={identity.get('rank')!r} local_rank="
                    f"{identity.get('local_rank')!r} pid={identity.get('pid')!r} "
                    "reported invalid flashinfer_resource_count="
                    f"{resource_count!r}."
                )
