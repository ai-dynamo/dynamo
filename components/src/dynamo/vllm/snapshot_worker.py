# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM GPU worker subclass used by Dynamo Snapshot FlashInfer PoC."""

import logging
import os
from typing import Any

from vllm.device_allocator import get_mem_allocator_instance
from vllm.v1.worker.gpu_worker import Worker

from .flashinfer_collectives import (
    install_no_nccl_snapshot_guard,
    install_strict_flashinfer_collective_guard,
)
from .flashinfer_snapshot import (
    FlashInferResourceReport,
    inspect_flashinfer_peer_resources,
    pause_flashinfer_peer_resources,
    resume_flashinfer_peer_resources,
)
from .snapshot_worker_config import (
    flashinfer_only_collectives_enabled,
    no_nccl_snapshot_mode_enabled,
)

logger = logging.getLogger("vllm.dynamo.snapshot_worker")


class SnapshotWorker(Worker):
    """Worker that pauses FlashInfer peer resources around vLLM sleep/wake."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if no_nccl_snapshot_mode_enabled():
            install_no_nccl_snapshot_guard()
        if flashinfer_only_collectives_enabled():
            install_strict_flashinfer_collective_guard()
        super().__init__(*args, **kwargs)
        if no_nccl_snapshot_mode_enabled():
            _validate_no_nccl_worker_state(self)
        logger.info(
            "SnapshotWorker initialized rank=%s local_rank=%s pid=%s",
            _rank(self),
            _local_rank(self),
            os.getpid(),
        )

    def sleep(self, level: int = 1) -> None:
        logger.info(
            "SnapshotWorker sleep start rank=%s local_rank=%s pid=%s level=%s",
            _rank(self),
            _local_rank(self),
            os.getpid(),
            level,
        )
        report = pause_flashinfer_peer_resources(self)
        logger.info(
            "SnapshotWorker FlashInfer pause complete rank=%s local_rank=%s "
            "pid=%s level=%s resource_count=%s resources=%s",
            _rank(self),
            _local_rank(self),
            os.getpid(),
            level,
            report.count,
            _resource_summary(report),
        )
        try:
            super().sleep(level=level)
            logger.info(
                "SnapshotWorker base vLLM sleep complete rank=%s local_rank=%s "
                "pid=%s level=%s",
                _rank(self),
                _local_rank(self),
                os.getpid(),
                level,
            )
            self._synchronize_after_sleep(level)
        except Exception:
            logger.exception(
                "SnapshotWorker sleep failed after FlashInfer pause rank=%s "
                "local_rank=%s pid=%s level=%s; rolling back FlashInfer pause",
                _rank(self),
                _local_rank(self),
                os.getpid(),
                level,
            )
            _rollback_flashinfer_pause(self, level)
            raise

    def wake_up(self, tags: list[str] | None = None) -> None:
        logger.info(
            "SnapshotWorker wake_up start rank=%s local_rank=%s pid=%s tags=%s",
            _rank(self),
            _local_rank(self),
            os.getpid(),
            tags,
        )
        allocator = get_mem_allocator_instance()
        allocator.wake_up(tags)

        # Match vLLM 0.23.0 wake_up ordering except resume FlashInfer after
        # allocator wake-up and before saved buffers / KV-cache post hooks.
        report = resume_flashinfer_peer_resources(self)
        logger.info(
            "SnapshotWorker FlashInfer resume complete rank=%s local_rank=%s "
            "pid=%s tags=%s resource_count=%s resources=%s",
            _rank(self),
            _local_rank(self),
            os.getpid(),
            tags,
            report.count,
            _resource_summary(report),
        )

        if len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

        if tags is None or "kv_cache" in tags:
            self.model_runner.post_kv_cache_wake_up()
        logger.info(
            "SnapshotWorker wake_up complete rank=%s local_rank=%s pid=%s tags=%s",
            _rank(self),
            _local_rank(self),
            os.getpid(),
            tags,
        )

    def snapshot_worker_identity(self) -> dict[str, Any]:
        report = inspect_flashinfer_peer_resources(self)
        identity = {
            "module": self.__class__.__module__,
            "class": self.__class__.__qualname__,
            "qualified_class": _qualified_class_name(self),
            "rank": _rank(self),
            "local_rank": _local_rank(self),
            "pid": os.getpid(),
            "flashinfer_resource_count": report.count,
            "flashinfer_resources": _resource_summary(report),
        }
        logger.info("SnapshotWorker identity %s", identity)
        return identity

    def _synchronize_after_sleep(self, level: int) -> None:
        logger.info(
            "SnapshotWorker device synchronize start rank=%s local_rank=%s "
            "pid=%s level=%s",
            _rank(self),
            _local_rank(self),
            os.getpid(),
            level,
        )
        try:
            sync_backend = _synchronize_snapshot_device()
        except Exception:
            logger.exception(
                "SnapshotWorker device synchronize failed rank=%s local_rank=%s "
                "pid=%s level=%s",
                _rank(self),
                _local_rank(self),
                os.getpid(),
                level,
            )
            raise

        logger.info(
            "SnapshotWorker device synchronize complete rank=%s local_rank=%s "
            "pid=%s level=%s backend=%s",
            _rank(self),
            _local_rank(self),
            os.getpid(),
            level,
            sync_backend,
        )


def _rollback_flashinfer_pause(worker: Any, level: int) -> None:
    try:
        report = resume_flashinfer_peer_resources(worker)
        logger.info(
            "SnapshotWorker FlashInfer rollback resume complete rank=%s local_rank=%s "
            "pid=%s level=%s resource_count=%s resources=%s",
            _rank(worker),
            _local_rank(worker),
            os.getpid(),
            level,
            report.count,
            _resource_summary(report),
        )
    except Exception:
        logger.exception(
            "SnapshotWorker FlashInfer rollback resume failed rank=%s "
            "local_rank=%s pid=%s level=%s",
            _rank(worker),
            _local_rank(worker),
            os.getpid(),
            level,
        )
        return


def _validate_no_nccl_worker_state(worker: Any) -> None:
    model_runner = getattr(worker, "model_runner", None)
    vllm_config = getattr(model_runner, "vllm_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if parallel_config is not None:
        if getattr(parallel_config, "disable_custom_all_reduce", None) is not True:
            raise RuntimeError(
                "Dynamo no-NCCL snapshot mode requires "
                "parallel_config.disable_custom_all_reduce=True in worker "
                f"rank={_rank(worker)}."
            )
        if (
            getattr(
                parallel_config,
                "disable_nccl_for_dp_synchronization",
                None,
            )
            is not True
        ):
            raise RuntimeError(
                "Dynamo no-NCCL snapshot mode requires "
                "parallel_config.disable_nccl_for_dp_synchronization=True in "
                f"worker rank={_rank(worker)}."
            )

    for communicator in _iter_device_communicators():
        _validate_no_nccl_communicator(communicator)


def _iter_device_communicators() -> list[Any]:
    try:
        from vllm.distributed.parallel_state import get_world_group

        groups = [get_world_group()]
    except Exception:
        groups = []

    for name in (
        "get_tp_group",
        "get_pp_group",
        "get_dp_group",
        "get_dcp_group",
    ):
        try:
            import vllm.distributed.parallel_state as parallel_state

            group = getattr(parallel_state, name)()
        except Exception:
            continue
        groups.append(group)

    communicators = []
    seen: set[int] = set()
    for group in groups:
        communicator = getattr(group, "device_communicator", None)
        if communicator is None or id(communicator) in seen:
            continue
        communicators.append(communicator)
        seen.add(id(communicator))
    return communicators


def _validate_no_nccl_communicator(communicator: Any) -> None:
    active = []
    pynccl_comm = getattr(communicator, "pynccl_comm", None)
    if pynccl_comm is not None and not getattr(pynccl_comm, "disabled", False):
        active.append("PyNCCL")

    ca_comm = getattr(communicator, "ca_comm", None)
    if ca_comm is not None and not getattr(ca_comm, "disabled", False):
        active.append("CustomAllreduce")

    symm_mem_comm = getattr(communicator, "symm_mem_comm", None)
    if symm_mem_comm is not None and not getattr(symm_mem_comm, "disabled", False):
        active.append("SymmMem")

    if active:
        raise RuntimeError(
            "Dynamo no-NCCL snapshot mode found active NCCL-backed vLLM "
            f"communicator(s) {active!r} in "
            f"{getattr(communicator, 'unique_name', '<unnamed>')!r}."
        )


def _synchronize_snapshot_device() -> str:
    """Synchronize the active accelerator before marking snapshot readiness."""
    import torch

    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None:
        is_available = getattr(accelerator, "is_available", None)
        if is_available is None or is_available():
            accelerator.synchronize()
            return "torch.accelerator"

    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.synchronize()
        return "torch.cuda"

    return "skipped"


def _rank(worker: Any) -> Any:
    return getattr(worker, "rank", os.environ.get("RANK", "unknown"))


def _local_rank(worker: Any) -> Any:
    return getattr(worker, "local_rank", os.environ.get("LOCAL_RANK", "unknown"))


def _qualified_class_name(obj: Any) -> str:
    cls = obj.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _resource_summary(report: FlashInferResourceReport) -> list[dict[str, str]]:
    return [
        {
            "name": resource.name,
            "kind": resource.kind,
            "class": resource.class_name,
        }
        for resource in report.resources
    ]
