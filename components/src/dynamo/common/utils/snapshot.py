# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Dynamo snapshot helpers for checkpoint lifecycle."""

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from dynamo.common.utils.namespace import get_worker_namespace

logger = logging.getLogger(__name__)
PODINFO_ROOT = "/etc/podinfo"
KUBERNETES_REQUIRED_PODINFO_FILES = {
    "DYN_NAMESPACE": "dyn_namespace",
    "DYN_COMPONENT": "dyn_component",
    "DYN_PARENT_DGD_K8S_NAME": "dyn_parent_dgd_k8s_name",
    "DYN_PARENT_DGD_K8S_NAMESPACE": "dyn_parent_dgd_k8s_namespace",
}
KUBERNETES_OPTIONAL_PODINFO_FILES = {
    "DYN_NAMESPACE_WORKER_SUFFIX": "dyn_namespace_worker_suffix",
}
EngineT = TypeVar("EngineT")

# Must match snapshotprotocol.{SnapshotCompleteFile,RestoreCompleteFile,ReadyForCheckpointFile}.
SNAPSHOT_COMPLETE_FILE = "snapshot-complete"
RESTORE_COMPLETE_FILE = "restore-complete"
READY_FOR_CHECKPOINT_FILE = "ready-for-checkpoint"

# Poll interval for the snapshot-control directory. Checkpoint and restore
# latencies are seconds, so 100ms is negligible overhead.
_SENTINEL_POLL_INTERVAL_SEC = 0.1


class CheckpointConfig:
    """Parsed checkpoint configuration plus the sentinel-driven lifecycle."""

    def __init__(self, control_dir: str):
        self.control_dir = control_dir
        self.ready_file = os.path.join(control_dir, READY_FOR_CHECKPOINT_FILE)

    @classmethod
    def from_env(cls) -> "CheckpointConfig | None":
        control_dir = os.environ.get("DYN_SNAPSHOT_CONTROL_DIR")
        if not control_dir:
            return None

        return cls(control_dir=control_dir)

    async def run_lifecycle(
        self,
        quiesce_controller: Any,
        *quiesce_args: object,
    ) -> bool:
        logger.info("Quiescing model")
        await quiesce_controller.quiesce(self.control_dir, *quiesce_args)

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
            await quiesce_controller.resume()
            quiesce_controller.mark_resumed()
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
            READY_FOR_CHECKPOINT_FILE,
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
    quiesce_controller: Any
    checkpoint_config: CheckpointConfig
    quiesce_args: tuple[object, ...] = ()

    async def wait_for_restore(self) -> bool:
        return await self.checkpoint_config.run_lifecycle(
            self.quiesce_controller,
            *self.quiesce_args,
        )

    def reload_restore_identity(
        self,
        namespace: str,
        discovery_backend: str,
    ) -> tuple[str, str]:
        return reload_snapshot_restore_identity(namespace, discovery_backend)


def reload_snapshot_restore_identity(
    namespace: str,
    discovery_backend: str,
) -> tuple[str, str]:
    if discovery_backend != "kubernetes":
        logger.info(
            "Snapshot restore reusing configured discovery backend",
            extra={
                "dynamo_namespace": namespace,
                "discovery_backend": discovery_backend,
            },
        )
        return namespace, discovery_backend

    for env_name, podinfo_file in KUBERNETES_REQUIRED_PODINFO_FILES.items():
        podinfo_path = os.path.join(PODINFO_ROOT, podinfo_file)
        if not os.path.isfile(podinfo_path):
            raise RuntimeError(f"snapshot restore requires {podinfo_path}")

        with open(podinfo_path, encoding="utf-8") as podinfo:
            value = podinfo.read().strip()
        if not value:
            raise RuntimeError(f"snapshot restore requires a non-empty {podinfo_path}")

        os.environ[env_name] = value

    for env_name, podinfo_file in KUBERNETES_OPTIONAL_PODINFO_FILES.items():
        podinfo_path = os.path.join(PODINFO_ROOT, podinfo_file)
        if not os.path.isfile(podinfo_path):
            os.environ.pop(env_name, None)
            continue

        with open(podinfo_path, encoding="utf-8") as podinfo:
            value = podinfo.read().strip()
        if not value:
            os.environ.pop(env_name, None)
            continue

        os.environ[env_name] = value

    os.environ["DYN_DISCOVERY_BACKEND"] = "kubernetes"
    return get_worker_namespace(), "kubernetes"
