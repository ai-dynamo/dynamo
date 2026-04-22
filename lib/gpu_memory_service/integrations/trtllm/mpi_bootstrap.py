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
    "GMS_SOCKET_DIR",
)

_extra_config_json: Optional[str] = None
_bootstrap_installed = False
_executor_finalize_hook_installed = False


def set_extra_config(extra: "dict[str, Any] | None") -> None:
    """Stash the model_loader_extra_config so MPI children see the same lock mode."""
    global _extra_config_json
    _extra_config_json = json.dumps(extra) if extra else None


def _delay_commit_until_engine_init(extra: "dict[str, Any] | None") -> bool:
    return bool(extra and extra.get("gms_delay_commit_until_engine_init") is True)


def install_executor_finalize_hook() -> None:
    """Finalize delayed TRT-LLM GMS publishes in MPI children before start_worker()."""
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

    @wraps(original_create_py_executor)
    def patched_create_py_executor(*args, **kwargs):
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

        _py_executor_creator.create_py_executor_instance = (
            patched_create_py_executor_instance
        )
        try:
            return original_create_py_executor(*args, **kwargs)
        finally:
            _py_executor_creator.create_py_executor_instance = (
                original_create_py_executor_instance
            )

    _py_executor_creator.create_py_executor = patched_create_py_executor
    _executor_finalize_hook_installed = True
    logger.info(
        "[GMS] Patched TensorRT-LLM create_py_executor to finalize delayed publish before start_worker"
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
    setup_gms(extra)
    if _delay_commit_until_engine_init(extra):
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
