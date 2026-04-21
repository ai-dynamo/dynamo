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


def set_extra_config(extra: "dict[str, Any] | None") -> None:
    """Stash the model_loader_extra_config so MPI children see the same lock mode."""
    global _extra_config_json
    _extra_config_json = json.dumps(extra) if extra else None


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
            key: os.environ[key]
            for key in _PROPAGATED_ENV_VARS
            if key in os.environ
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
