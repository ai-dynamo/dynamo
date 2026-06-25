# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo-local vLLM Snapshot worker hooks."""

from __future__ import annotations

import logging
import os
from typing import Any

from vllm.v1.worker.gpu_worker import Worker

logger = logging.getLogger("vllm.dynamo.snapshot_worker")

SNAPSHOT_WORKER_CLASS = "dynamo.vllm.snapshot_worker.SnapshotWorker"
_ALLOWED_WORKER_CLS = (None, "", "auto", SNAPSHOT_WORKER_CLASS)


def configure_snapshot_worker(config: Any) -> bool:
    """Install ``SnapshotWorker`` for Dynamo Snapshot mode."""

    engine_args = getattr(config, "engine_args", None)
    if engine_args is None:
        raise ValueError("Dynamo vLLM Snapshot worker requires engine_args.")

    worker_cls = getattr(engine_args, "worker_cls", None)
    if getattr(engine_args, "load_format", None) == "gms":
        raise ValueError(
            "Dynamo vLLM Snapshot worker is incompatible with --load-format gms."
        )
    if worker_cls not in _ALLOWED_WORKER_CLS:
        raise ValueError(
            "Dynamo vLLM Snapshot worker cannot override existing "
            f"worker_cls={worker_cls!r}."
        )

    engine_args.worker_cls = SNAPSHOT_WORKER_CLASS
    logger.info("Configured vLLM Snapshot worker_cls=%s", SNAPSHOT_WORKER_CLASS)
    return True


class SnapshotWorker(Worker):
    """vLLM GPU worker with optional vLLM/NCCL snapshot hooks."""

    def sleep(self, level: int = 1) -> None:
        logger.info("SnapshotWorker sleep start %s level=%s", _ctx(self), level)
        prepared = False
        try:
            self.checkpoint_prepare()
            prepared = True

            super().sleep(level=level)
        except Exception:
            logger.exception("SnapshotWorker sleep failed %s level=%s", _ctx(self), level)
            if prepared:
                _checkpoint_restore_best_effort(self)
            raise
        logger.info("SnapshotWorker sleep complete %s level=%s", _ctx(self), level)

    def wake_up(self, tags: list[str] | None = None) -> None:
        logger.info("SnapshotWorker wake_up start %s tags=%s", _ctx(self), tags)
        super().wake_up(tags=tags)
        self.checkpoint_restore()
        logger.info("SnapshotWorker wake_up complete %s tags=%s", _ctx(self), tags)

    def snapshot_worker_identity(self) -> dict[str, Any]:
        identity = {
            "qualified_class": _qualified_class_name(self),
            "rank": _rank(self),
            "local_rank": _local_rank(self),
            "pid": os.getpid(),
        }
        logger.info("SnapshotWorker identity %s", identity)
        return identity


def _checkpoint_restore_best_effort(worker: Worker) -> None:
    try:
        worker.checkpoint_restore()
    except Exception:
        logger.exception("SnapshotWorker rollback checkpoint_restore failed")


def _rank(worker: Any) -> Any:
    return getattr(worker, "rank", os.environ.get("RANK", "unknown"))


def _local_rank(worker: Any) -> Any:
    return getattr(worker, "local_rank", os.environ.get("LOCAL_RANK", "unknown"))


def _ctx(worker: Any) -> dict[str, Any]:
    return {"rank": _rank(worker), "local_rank": _local_rank(worker), "pid": os.getpid()}


def _qualified_class_name(obj: Any) -> str:
    cls = obj.__class__
    return f"{cls.__module__}.{cls.__qualname__}"
