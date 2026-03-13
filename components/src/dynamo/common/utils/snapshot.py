# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Dynamo snapshot helpers for checkpoint lifecycle."""

import asyncio
import logging
import os
import signal
from typing import Any

from dynamo.common.utils.namespace import get_worker_namespace

_LOG = logging.getLogger(__name__)
PODINFO_ROOT = "/etc/podinfo"
PODINFO_FILES = {
    "DYN_NAMESPACE": "dyn_namespace",
    "DYN_NAMESPACE_WORKER_SUFFIX": "dyn_namespace_worker_suffix",
    "DYN_COMPONENT": "dyn_component",
    "DYN_PARENT_DGD_K8S_NAME": "dyn_parent_dgd_k8s_name",
    "DYN_PARENT_DGD_K8S_NAMESPACE": "dyn_parent_dgd_k8s_namespace",
}


class CheckpointConfig:
    """Parsed checkpoint configuration plus the watcher-driven lifecycle."""

    def __init__(self, ready_file: str, storage_type: str, location: str):
        self.ready_file = ready_file
        self.storage_type = storage_type
        self.location = location
        self._checkpoint_done = asyncio.Event()
        self._restore_done = asyncio.Event()

    @classmethod
    def from_env(cls) -> "CheckpointConfig | None":
        ready_file = os.environ.get("DYN_READY_FOR_CHECKPOINT_FILE")
        if not ready_file:
            return None

        location = os.environ.get("DYN_CHECKPOINT_LOCATION", "")
        if not location:
            checkpoint_path = os.environ.get("DYN_CHECKPOINT_PATH", "").rstrip("/")
            checkpoint_hash = os.environ.get("DYN_CHECKPOINT_HASH", "")
            if not checkpoint_path or not checkpoint_hash:
                raise EnvironmentError(
                    "Checkpoint mode requires either DYN_CHECKPOINT_LOCATION or both "
                    "DYN_CHECKPOINT_PATH and DYN_CHECKPOINT_HASH"
                )
            location = f"{checkpoint_path}/{checkpoint_hash}"

        return cls(
            ready_file=ready_file,
            storage_type=os.environ.get("DYN_CHECKPOINT_STORAGE_TYPE", "pvc"),
            location=location,
        )

    def checkpoint_exists(self) -> bool:
        if self.storage_type != "pvc":
            return False

        # The snapshot agent stages PVC checkpoints under <base>/tmp/<hash> and
        # only publishes self.location after a successful finalize/rename.
        if os.path.isdir(self.location):
            _LOG.info("Existing checkpoint found at %s, skipping", self.location)
            return True

        _LOG.info("No checkpoint at %s, creating new one", self.location)
        return False

    async def run_lifecycle(self, engine_client: Any, sleep_level: int) -> bool:
        _LOG.info("Putting model to sleep (level=%s)", sleep_level)
        await engine_client.sleep(level=sleep_level)

        self._install_signal_handlers()

        with open(self.ready_file, "w", encoding="utf-8") as ready_file:
            ready_file.write("ready")
        _LOG.info(
            "Ready for checkpoint. Waiting for watcher signal "
            "(SIGUSR1=checkpoint complete, SIGCONT=restore complete)"
        )

        try:
            event = await self._wait_for_watcher_signal()
            if event == "restore":
                _LOG.info("Restore signal detected (SIGCONT)")
                _LOG.info("Waking up model after restore")
                await engine_client.wake_up()
                return True

            _LOG.info("Checkpoint completion signal detected (SIGUSR1)")
            return False
        finally:
            self._remove_signal_handlers()
            try:
                os.unlink(self.ready_file)
            except OSError:
                pass

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGUSR1, self._checkpoint_done.set)
        loop.add_signal_handler(signal.SIGCONT, self._restore_done.set)

    def _remove_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        loop.remove_signal_handler(signal.SIGUSR1)
        loop.remove_signal_handler(signal.SIGCONT)

    async def _wait_for_watcher_signal(self) -> str:
        waiters = {
            asyncio.create_task(self._checkpoint_done.wait()): "checkpoint",
            asyncio.create_task(self._restore_done.wait()): "restore",
        }
        try:
            done, pending = await asyncio.wait(
                waiters.keys(), return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            winner = done.pop()
            await winner
            return waiters[winner]
        finally:
            for task in waiters:
                if not task.done():
                    task.cancel()


def get_checkpoint_config() -> tuple[bool, CheckpointConfig | None]:
    """Resolve checkpoint mode for checkpoint-job pods."""

    cfg = CheckpointConfig.from_env()
    if cfg is None:
        return False, None

    checkpoint_exists = cfg.checkpoint_exists()
    if checkpoint_exists:
        return True, None

    return False, cfg


def reload_snapshot_restore_identity() -> tuple[str, str]:
    namespace = None

    for env_name, podinfo_file in PODINFO_FILES.items():
        podinfo_path = os.path.join(PODINFO_ROOT, podinfo_file)
        if not os.path.isfile(podinfo_path):
            if env_name == "DYN_NAMESPACE":
                raise RuntimeError(
                    "snapshot restore requires /etc/podinfo/dyn_namespace"
                )
            os.environ.pop(env_name, None)
            continue

        with open(podinfo_path, encoding="utf-8") as podinfo:
            value = podinfo.read().strip()
        if not value:
            if env_name == "DYN_NAMESPACE":
                raise RuntimeError(
                    "snapshot restore requires a non-empty /etc/podinfo/dyn_namespace"
                )
            os.environ.pop(env_name, None)
            continue

        os.environ[env_name] = value
        if env_name == "DYN_NAMESPACE":
            namespace = value

    os.environ["DYN_DISCOVERY_BACKEND"] = "kubernetes"
    return get_worker_namespace(namespace), "kubernetes"
