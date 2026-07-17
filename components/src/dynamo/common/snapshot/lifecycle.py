# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snapshot lifecycle helpers for Dynamo Snapshot."""

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Generic, TypeVar

from dynamo.common.snapshot.constants import (
    READY_FOR_SNAPSHOT_FILE,
    RESTORE_COMPLETE_FILE,
    SNAPSHOT_COMPLETE_FILE,
    SNAPSHOT_CONTROL_DIR_ENV,
)

logger = logging.getLogger(__name__)
EngineT = TypeVar("EngineT")

# Poll interval for the snapshot-control directory. Snapshot and restore
# latencies are seconds, so 100ms is negligible overhead.
SENTINEL_POLL_INTERVAL_SEC = 0.1


class SnapshotLifecycleError(RuntimeError):
    """Snapshot failure annotated with the lifecycle phase."""

    def __init__(self, phase: str, message: str):
        self.phase = phase
        super().__init__(message)


def is_snapshot_enabled() -> bool:
    return bool(os.environ.get(SNAPSHOT_CONTROL_DIR_ENV))


class SnapshotConfig:
    """Parsed snapshot configuration plus the sentinel-driven lifecycle."""

    def __init__(self, control_dir: str):
        self.control_dir = control_dir
        self.ready_file = os.path.join(control_dir, READY_FOR_SNAPSHOT_FILE)

    @classmethod
    def from_env(cls) -> "SnapshotConfig | None":
        control_dir = os.environ.get(SNAPSHOT_CONTROL_DIR_ENV)
        if not control_dir:
            return None

        return cls(control_dir=control_dir)

    async def run_lifecycle(
        self,
        pause_controller: Any,
        *pause_args: object,
        cancelled: Callable[[], Awaitable[object]] | None = None,
    ) -> bool:
        logger.info("Pausing model")
        try:
            await pause_controller.pause(*pause_args)
        except Exception as exc:
            raise SnapshotLifecycleError("snapshot_prepare", str(exc)) from exc

        try:
            try:
                with open(self.ready_file, "w", encoding="utf-8") as ready_file:
                    ready_file.write("ready")

                logger.info(
                    "Ready for snapshot. Polling for sentinel in %s "
                    "(snapshot-complete or restore-complete)",
                    self.control_dir,
                )

                event = await self._wait_for_sentinel_or_cancel(cancelled)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise SnapshotLifecycleError("snapshot_wait", str(exc)) from exc
        finally:
            self._cleanup_ready_and_sentinels()

        if event == "restore":
            logger.info("Restore sentinel detected")
            logger.info("Resuming model after restore")
            try:
                await pause_controller.resume()
                pause_controller.mark_resumed()
            except Exception as exc:
                raise SnapshotLifecycleError("restore_resume", str(exc)) from exc
            # The checkpoint is complete; post-restore model registration may
            # need normal Hugging Face cache/download behavior.
            os.environ.pop("HF_HUB_OFFLINE", None)
            return True

        logger.info("Snapshot completion sentinel detected")
        return False

    async def _wait_for_sentinel_or_cancel(
        self, cancelled: Callable[[], Awaitable[object]] | None
    ) -> str:
        if cancelled is None:
            return await self._wait_for_sentinel()

        sentinel_task = asyncio.create_task(self._wait_for_sentinel())
        cancel_task = asyncio.ensure_future(cancelled())
        try:
            done, _ = await asyncio.wait(
                {sentinel_task, cancel_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_task in done:
                sentinel_task.cancel()
                await asyncio.gather(sentinel_task, return_exceptions=True)
                raise asyncio.CancelledError("snapshot lifecycle cancelled")
            return sentinel_task.result()
        finally:
            for task in (sentinel_task, cancel_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(sentinel_task, cancel_task, return_exceptions=True)

    async def _wait_for_sentinel(self) -> str:
        snapshot_path = Path(self.control_dir) / SNAPSHOT_COMPLETE_FILE
        restore_path = Path(self.control_dir) / RESTORE_COMPLETE_FILE
        while True:
            if snapshot_path.exists():
                return "snapshot"
            if restore_path.exists():
                return "restore"
            await asyncio.sleep(SENTINEL_POLL_INTERVAL_SEC)

    def _cleanup_ready_and_sentinels(self) -> None:
        for name in (
            READY_FOR_SNAPSHOT_FILE,
            SNAPSHOT_COMPLETE_FILE,
            RESTORE_COMPLETE_FILE,
        ):
            path = os.path.join(self.control_dir, name)
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.exception("Failed to clean up %s at %s", name, path)


def configure_snapshot_capture_env() -> None:
    nccl_cumem_enable = os.environ.get("NCCL_CUMEM_ENABLE")
    if nccl_cumem_enable and nccl_cumem_enable != "0":
        logger.warning(
            "Overriding NCCL_CUMEM_ENABLE=%r with '0' for snapshot mode "
            "because cuda-checkpoint does not support cuMem-backed NCCL allocations",
            nccl_cumem_enable,
        )
    os.environ["NCCL_CUMEM_ENABLE"] = "0"

    nccl_nvls_enable = os.environ.get("NCCL_NVLS_ENABLE")
    if nccl_nvls_enable and nccl_nvls_enable != "0":
        logger.warning(
            "Overriding NCCL_NVLS_ENABLE=%r with '0' for snapshot mode "
            "to avoid NVLS and keep NCCL on the legacy P2P path",
            nccl_nvls_enable,
        )
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    nccl_ib_disable = os.environ.get("NCCL_IB_DISABLE")
    if nccl_ib_disable and nccl_ib_disable != "1":
        logger.warning(
            "Overriding NCCL_IB_DISABLE=%r with '1' for snapshot mode "
            "because CRIU and cuda-checkpoint cannot restore InfiniBand state",
            nccl_ib_disable,
        )
    os.environ["NCCL_IB_DISABLE"] = "1"

    nccl_ras_enable = os.environ.get("NCCL_RAS_ENABLE")
    if nccl_ras_enable and nccl_ras_enable != "0":
        logger.warning(
            "Overriding NCCL_RAS_ENABLE=%r with '0' for snapshot mode "
            "because NCCL RAS background state is not part of the snapshot contract",
            nccl_ras_enable,
        )
    os.environ["NCCL_RAS_ENABLE"] = "0"

    torch_nccl_monitoring = os.environ.get("TORCH_NCCL_ENABLE_MONITORING")
    if torch_nccl_monitoring and torch_nccl_monitoring != "0":
        logger.warning(
            "Overriding TORCH_NCCL_ENABLE_MONITORING=%r with '0' for "
            "snapshot mode because ProcessGroupNCCL monitoring can "
            "terminate restored processes",
            torch_nccl_monitoring,
        )
    os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
    os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "0")

    hf_hub_offline = os.environ.get("HF_HUB_OFFLINE")
    if hf_hub_offline and hf_hub_offline != "1":
        logger.warning(
            "Overriding HF_HUB_OFFLINE=%r with '1' in snapshot mode "
            "to avoid external Hugging Face sockets in snapshotted processes",
            hf_hub_offline,
        )
    os.environ["HF_HUB_OFFLINE"] = "1"


@dataclass
class EngineSnapshotController(Generic[EngineT]):
    engine: EngineT
    pause_controller: Any
    snapshot_config: SnapshotConfig
    pause_args: tuple[object, ...] = ()

    async def wait_for_restore(
        self, cancelled: Callable[[], Awaitable[object]] | None = None
    ) -> bool:
        return await self.snapshot_config.run_lifecycle(
            self.pause_controller,
            *self.pause_args,
            cancelled=cancelled,
        )


async def unified_snapshot_outcome(
    controller: EngineSnapshotController[Any],
    *,
    argv: list[str] | None,
    context: Any,
    backend_name: str,
):
    """Translate the shared sentinel protocol into the unified engine result."""

    from dynamo.common.backend import PreRuntimeOutcome
    from dynamo.common.snapshot.restore_context import load_restored_runtime_config
    from dynamo.llm.exceptions import Cancelled, EngineShutdown

    try:
        restored = await controller.wait_for_restore(cancelled=context.cancelled)
    except asyncio.CancelledError as exc:
        raise Cancelled(f"{backend_name} snapshot lifecycle cancelled") from exc
    except SnapshotLifecycleError as exc:
        raise EngineShutdown(f"{backend_name} {exc.phase} failed: {exc}") from exc

    if not restored:
        return PreRuntimeOutcome.exit_success()

    try:
        restored_runtime = load_restored_runtime_config(argv)
    except Exception as exc:
        raise EngineShutdown(f"{backend_name} restore_context failed: {exc}") from exc
    return PreRuntimeOutcome.prepared(restored_runtime)
