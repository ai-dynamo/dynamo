# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Dynamo snapshot helpers for checkpoint lifecycle."""

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from dynamo.common.snapshot.constants import (
    READY_FOR_SNAPSHOT_FILE,
    RESTORE_COMPLETE_FILE,
    SNAPSHOT_COMPLETE_FILE,
    SNAPSHOT_CONTROL_DIR_ENV,
)

logger = logging.getLogger(__name__)
EngineT = TypeVar("EngineT")

# Poll interval for the snapshot-control directory. Checkpoint and restore
# latencies are seconds, so 100ms is negligible overhead.
_SENTINEL_POLL_INTERVAL_SEC = 0.1


class CheckpointConfig:
    """Parsed checkpoint configuration plus the sentinel-driven lifecycle."""

    def __init__(self, control_dir: str):
        self.control_dir = control_dir
        self.ready_file = os.path.join(control_dir, READY_FOR_SNAPSHOT_FILE)

    @classmethod
    def from_env(cls) -> "CheckpointConfig | None":
        control_dir = os.environ.get(SNAPSHOT_CONTROL_DIR_ENV)
        if not control_dir:
            return None

        return cls(control_dir=control_dir)

    async def run_lifecycle(
        self,
        pause_controller: Any,
        *pause_args: object,
    ) -> bool:
        logger.info("Pausing model for checkpoint")
        await pause_controller.pause(self.control_dir, *pause_args)

        event = None
        try:
            with open(self.ready_file, "w", encoding="utf-8") as ready_file:
                ready_file.write("ready")

            logger.info(
                "Ready for checkpoint. Polling for sentinel in %s "
                "(snapshot-complete or restore-complete)",
                self.control_dir,
            )

            event = await self._wait_for_sentinel()
        finally:
            if event != "restore":
                self._cleanup_ready_and_sentinels()

        if event == "restore":
            logger.info("Restore sentinel detected")
            logger.info("Resuming model after restore")
            await pause_controller.resume()
            pause_controller.mark_resumed()
            self._cleanup_ready_and_sentinels()
            return True

        logger.info("Snapshot completion sentinel detected")
        return False

    async def _wait_for_sentinel(self) -> str:
        snapshot_path = Path(self.control_dir) / SNAPSHOT_COMPLETE_FILE
        restore_path = Path(self.control_dir) / RESTORE_COMPLETE_FILE
        while True:
            if snapshot_path.exists():
                return "checkpoint"
            if restore_path.exists():
                return "restore"
            await asyncio.sleep(_SENTINEL_POLL_INTERVAL_SEC)

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


@dataclass
class EngineSnapshotController(Generic[EngineT]):
    engine: EngineT
    pause_controller: Any
    checkpoint_config: CheckpointConfig
    pause_args: tuple[object, ...] = ()

    async def wait_for_restore(self) -> bool:
        return await self.checkpoint_config.run_lifecycle(
            self.pause_controller,
            *self.pause_args,
        )
