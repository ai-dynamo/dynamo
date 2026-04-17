# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Dynamo snapshot helpers for checkpoint lifecycle."""

import asyncio
import logging
import os
import signal
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from dynamo.common.utils.namespace import get_worker_namespace
from gpu_memory_service.common.utils import get_checkpoint_save_complete_path

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


class CheckpointConfig:
    """Parsed checkpoint configuration plus the watcher-driven lifecycle."""

    def __init__(self, ready_file: str):
        self.ready_file = ready_file
        self._checkpoint_done = asyncio.Event()
        self._restore_done = asyncio.Event()

    @classmethod
    def from_env(cls) -> "CheckpointConfig | None":
        ready_file = os.environ.get("DYN_READY_FOR_CHECKPOINT_FILE")
        if not ready_file:
            return None

        configure_checkpoint_transport_env()
        return cls(ready_file=ready_file)

    async def run_lifecycle(
        self,
        quiesce_controller: Any,
        *quiesce_args: object,
    ) -> bool:
        logger.info("Quiescing model")
        await quiesce_controller.quiesce(*quiesce_args)
        await self._wait_for_gms_checkpoint_save()

        self._install_signal_handlers()
        try:
            with open(self.ready_file, "w", encoding="utf-8") as ready_file:
                ready_file.write("ready")
        except Exception:
            self._remove_signal_handlers()
            raise

        logger.info(
            "Ready for checkpoint. Waiting for watcher signal "
            "(SIGUSR1=checkpoint complete, SIGCONT=restore complete)"
        )

        try:
            event = await self._wait_for_watcher_signal()
            if event == "restore":
                logger.info("Restore signal detected (SIGCONT)")
                logger.info("Resuming model after restore")
                await quiesce_controller.resume()
                quiesce_controller.mark_resumed()
                return True

            logger.info("Checkpoint completion signal detected (SIGUSR1)")
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

    async def _wait_for_gms_checkpoint_save(self) -> None:
        ready_path = get_checkpoint_save_complete_path()
        if ready_path is None or os.path.exists(ready_path):
            return

        logger.info("Waiting for GMS checkpoint save to complete: %s", ready_path)
        while not os.path.exists(ready_path):
            await asyncio.sleep(0.1)


def configure_checkpoint_transport_env() -> None:
    gloo_ifname = os.environ.get("GLOO_SOCKET_IFNAME")
    if gloo_ifname and gloo_ifname != "lo":
        logger.warning(
            "Overriding GLOO_SOCKET_IFNAME=%r with 'lo' for checkpoint mode "
            "because CRIU cannot restore sockets bound to non-loopback addresses",
            gloo_ifname,
        )
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"

    nccl_ifname = os.environ.get("NCCL_SOCKET_IFNAME")
    if nccl_ifname and nccl_ifname != "lo":
        logger.warning(
            "Overriding NCCL_SOCKET_IFNAME=%r with 'lo' for checkpoint mode "
            "because CRIU cannot restore sockets bound to non-loopback addresses",
            nccl_ifname,
        )
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"

    nccl_cumem_enable = os.environ.get("NCCL_CUMEM_ENABLE")
    if nccl_cumem_enable and nccl_cumem_enable != "0":
        logger.warning(
            "Overriding NCCL_CUMEM_ENABLE=%r with '0' for checkpoint mode "
            "because cuda-checkpoint does not support cuMem-backed NCCL allocations",
            nccl_cumem_enable,
        )
    os.environ["NCCL_CUMEM_ENABLE"] = "0"

    nccl_p2p_disable = os.environ.get("NCCL_P2P_DISABLE")
    if nccl_p2p_disable and nccl_p2p_disable != "0":
        logger.warning(
            "Overriding NCCL_P2P_DISABLE=%r with '0' for checkpoint mode "
            "to keep NCCL on GPU P2P transport when topology allows it",
            nccl_p2p_disable,
        )
    os.environ["NCCL_P2P_DISABLE"] = "0"

    nccl_nvls_enable = os.environ.get("NCCL_NVLS_ENABLE")
    if nccl_nvls_enable and nccl_nvls_enable != "0":
        logger.warning(
            "Overriding NCCL_NVLS_ENABLE=%r with '0' for checkpoint mode "
            "to avoid NVLS and keep NCCL on the legacy P2P path",
            nccl_nvls_enable,
        )
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    nccl_ib_disable = os.environ.get("NCCL_IB_DISABLE")
    if nccl_ib_disable and nccl_ib_disable != "1":
        logger.warning(
            "Overriding NCCL_IB_DISABLE=%r with '1' for checkpoint mode "
            "because CRIU and cuda-checkpoint cannot restore InfiniBand state",
            nccl_ib_disable,
        )
    os.environ["NCCL_IB_DISABLE"] = "1"

    nccl_ras_enable = os.environ.get("NCCL_RAS_ENABLE")
    if nccl_ras_enable and nccl_ras_enable != "0":
        logger.warning(
            "Overriding NCCL_RAS_ENABLE=%r with '0' for checkpoint mode "
            "because NCCL RAS background state is not part of the checkpoint contract",
            nccl_ras_enable,
        )
    os.environ["NCCL_RAS_ENABLE"] = "0"

    torch_nccl_monitoring = os.environ.get("TORCH_NCCL_ENABLE_MONITORING")
    if torch_nccl_monitoring and torch_nccl_monitoring != "0":
        logger.warning(
            "Overriding TORCH_NCCL_ENABLE_MONITORING=%r with '0' for checkpoint mode "
            "because ProcessGroupNCCL monitoring can terminate restored processes",
            torch_nccl_monitoring,
        )
    os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
    os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "0")


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
