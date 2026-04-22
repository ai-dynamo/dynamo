# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Propagate the GMS TensorRT-LLM integration into MPI worker child processes.

At TP>1, TRT-LLM spawns fresh Python interpreters via mpi4py.futures.MPIPoolExecutor
(see tensorrt_llm/llmapi/mpi_session.py::MpiPoolSession._start_mpi_pool). Those
children import tensorrt_llm directly and never import the Dynamo worker module,
so setup_gms() — which monkey-patches ModelLoader.load — only runs in the parent.
Every child rank then hits the unpatched load path and allocates its own full
weight shard, duplicating weights between active and shadow engines.

This module fixes that by monkey-patching MpiPoolSession._start_mpi_pool so the
MPIPoolExecutor is constructed with an ``initializer=worker_init_hook`` that runs
first in every child. The hook restores the GMS-relevant env vars from the
parent's snapshot and calls setup_gms() so that the child's own ModelLoader.load
is patched before TRT-LLM invokes it.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import sys
from functools import wraps
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)

_ENABLED_ENV = "DYN_GMS_TRTLLM_ENABLED"
# Env vars the children must inherit for the child-side setup_gms to behave
# identically to the parent. Upstream MpiPoolSession only forwards TRTLLM_*/TLLM_*.
_PROPAGATED_ENV_VARS: tuple[str, ...] = (
    _ENABLED_ENV,
    "DYN_GMS_SHADOW_MODE",
    "ENGINE_ID",
    "FAILOVER_LOCK_PATH",
    "GMS_SOCKET_DIR",
)

_extra_config_json: Optional[str] = None
_bootstrap_installed = False
_executor_finalize_hook_installed = False
_shadow_activation_fd: int | None = None


def set_extra_config(extra: "dict[str, Any] | None") -> None:
    """Stash the model_loader_extra_config so MPI children see the same lock mode."""
    global _extra_config_json
    _extra_config_json = json.dumps(extra) if extra else None


def _delay_commit_until_engine_init(extra: "dict[str, Any] | None") -> bool:
    return bool(extra and extra.get("gms_delay_commit_until_engine_init") is True)


def _is_shadow_standby() -> bool:
    return (
        os.environ.get("DYN_GMS_SHADOW_MODE") == "1"
        and os.environ.get("ENGINE_ID", "0") != "0"
    )


def _get_mpi_rank_and_comm() -> tuple[int, object | None]:
    try:
        from mpi4py import MPI  # type: ignore[import-not-found]

        return MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD
    except Exception:
        return 0, None


def _wait_for_shadow_activation() -> None:
    """Block standby after the temp estimator path, before final full-KV init.

    This mirrors the flock semantics used at the parent layer, but keeps the
    wait inside TRT-LLM's synchronous executor-construction path without
    changing the generic async failover-lock helper.
    """
    global _shadow_activation_fd

    if not _is_shadow_standby():
        return

    rank, comm = _get_mpi_rank_and_comm()
    if rank == 0 and _shadow_activation_fd is None:
        shadow_engine_id = os.environ.get("ENGINE_ID", "0")
        lock_path = os.environ.get("FAILOVER_LOCK_PATH", "/shared/failover.lock")
        logger.info(
            "[Shadow] Standby RO import complete, rank0 waiting for failover lock before KV init"
        )

        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"engine-{shadow_engine_id}".encode())
        _shadow_activation_fd = fd

        logger.info("[Shadow] Standby rank0 acquired failover lock, continuing KV init")

    if comm is not None:
        comm.Barrier()


def install_executor_finalize_hook() -> None:
    """Finalize delayed publishes and gate standby before final full-KV build."""
    global _executor_finalize_hook_installed
    if _executor_finalize_hook_installed:
        return

    try:
        import tensorrt_llm._torch.pyexecutor.py_executor_creator as _py_executor_creator
    except ImportError as exc:
        raise RuntimeError(
            "GMS delayed TRT-LLM publish hook requires py_executor_creator"
        ) from exc

    original_create_py_executor = _py_executor_creator.create_py_executor
    original_create_py_executor_instance = (
        _py_executor_creator.create_py_executor_instance
    )
    original_teardown_managers = _py_executor_creator.KvCacheCreator.teardown_managers
    original_gc_collect = _py_executor_creator.gc.collect

    @wraps(original_create_py_executor)
    def patched_create_py_executor(*args, **kwargs):
        wait_after_gc = False
        waited_for_activation = False

        def patched_teardown_managers(*teardown_args, **teardown_kwargs):
            nonlocal wait_after_gc

            result = original_teardown_managers(*teardown_args, **teardown_kwargs)
            if _is_shadow_standby():
                wait_after_gc = True
            return result

        def patched_gc_collect(*collect_args, **collect_kwargs):
            nonlocal wait_after_gc, waited_for_activation

            result = original_gc_collect(*collect_args, **collect_kwargs)
            if wait_after_gc and not waited_for_activation:
                _wait_for_shadow_activation()
                waited_for_activation = True
                wait_after_gc = False
            return result

        def patched_create_py_executor_instance(*instance_args, **instance_kwargs):
            executor = original_create_py_executor_instance(
                *instance_args, **instance_kwargs
            )
            original_start_worker = executor.start_worker
            finalized = False

            @wraps(original_start_worker)
            def patched_start_worker(*worker_args, **worker_kwargs):
                nonlocal finalized
                if not finalized:
                    from gpu_memory_service.integrations.trtllm import (
                        finalize_pending_gms_write,
                    )

                    finalize_pending_gms_write()
                    finalized = True
                return original_start_worker(*worker_args, **worker_kwargs)

            executor.start_worker = patched_start_worker
            return executor

        if _delay_commit_until_engine_init(
            json.loads(_extra_config_json) if _extra_config_json else None
        ):
            _py_executor_creator.create_py_executor_instance = (
                patched_create_py_executor_instance
            )
        if _is_shadow_standby():
            _py_executor_creator.KvCacheCreator.teardown_managers = (
                patched_teardown_managers
            )
            _py_executor_creator.gc.collect = patched_gc_collect
        try:
            return original_create_py_executor(*args, **kwargs)
        finally:
            _py_executor_creator.KvCacheCreator.teardown_managers = (
                original_teardown_managers
            )
            _py_executor_creator.gc.collect = original_gc_collect
            _py_executor_creator.create_py_executor_instance = (
                original_create_py_executor_instance
            )

    _py_executor_creator.create_py_executor = patched_create_py_executor
    _executor_finalize_hook_installed = True
    logger.info(
        "[GMS] Patched TensorRT-LLM create_py_executor for delayed publish and standby activation gating"
    )


def worker_init_hook(env_snapshot: dict, extra_config_json: Optional[str]) -> None:
    """Run once per MPI worker child process before any task executes.

    Must be picklable by reference (module-level function).
    """
    # Best-effort: try to identify the rank for logging. MPI_COMM_WORLD may not
    # be initialised yet in some mpi4py builds; swallow the error if so.
    try:
        from mpi4py import MPI  # type: ignore[import-not-found]

        rank = MPI.COMM_WORLD.Get_rank()
    except Exception:
        rank = -1

    for key, value in env_snapshot.items():
        os.environ[key] = value

    if os.environ.get(_ENABLED_ENV) != "1":
        return

    from gpu_memory_service.integrations.trtllm import setup_gms

    extra = json.loads(extra_config_json) if extra_config_json else None
    set_extra_config(extra)
    setup_gms(extra)
    if _delay_commit_until_engine_init(extra) or _is_shadow_standby():
        install_executor_finalize_hook()
    logger.info(
        "[GMS] MPI worker init hook applied (pid=%d rank=%d)", os.getpid(), rank
    )


def install_mpi_worker_bootstrap() -> None:
    """Monkey-patch MpiPoolSession to run worker_init_hook in every child. Idempotent."""
    global _bootstrap_installed
    if _bootstrap_installed:
        return

    try:
        import tensorrt_llm.llmapi.mpi_session as _mpi_session
    except ImportError as exc:
        raise RuntimeError(
            "GMS MPI worker bootstrap requires tensorrt_llm to be importable"
        ) from exc

    MpiPoolSession = _mpi_session.MpiPoolSession

    def _patched_start_mpi_pool(self) -> None:
        assert not self.mpi_pool, "MPI session already started"
        from mpi4py.futures import MPIPoolExecutor  # type: ignore[import-not-found]

        env = {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("TRTLLM", "TLLM"))
        }
        env_snapshot = {
            key: os.environ[key] for key in _PROPAGATED_ENV_VARS if key in os.environ
        }
        # Forward propagated vars through MPI's env= too so they're set before
        # Python starts (some libs cache at import time).
        env.update(env_snapshot)

        self.mpi_pool = MPIPoolExecutor(
            max_workers=self.n_workers,
            path=sys.path,
            env=env,
            initializer=worker_init_hook,
            initargs=(env_snapshot, _extra_config_json),
        )

    MpiPoolSession._start_mpi_pool = _patched_start_mpi_pool
    _bootstrap_installed = True
    logger.info(
        "[GMS] Patched TensorRT-LLM MpiPoolSession._start_mpi_pool to bootstrap GMS in MPI workers"
    )
