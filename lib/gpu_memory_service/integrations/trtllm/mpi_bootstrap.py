# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Propagate the GMS TensorRT-LLM integration into MPI worker child processes.

At TP>1, TRT-LLM spawns fresh Python interpreters via mpi4py.futures.MPIPoolExecutor
(see tensorrt_llm/llmapi/mpi_session.py::MpiPoolSession._start_mpi_pool). Those
children import tensorrt_llm directly and never import the Dynamo worker module,
so setup_gms() — which monkey-patches ModelLoader.load — only runs in the parent.
Every child rank then hits the unpatched load path and allocates its own full
weight shard, duplicating weights between active and shadow engines.

This module fixes that in both TRT-LLM worker-launch paths:

- ``MpiPoolSession`` / ``MPIPoolExecutor`` children get
  ``initializer=worker_init_hook``.
- ``MpiCommSession`` / ``MPICommExecutor`` workers run a top-level
  ``worker_main`` wrapper that calls the same hook before delegating to
  TRT-LLM's real executor entrypoint.

In both cases the hook restores the GMS-relevant env vars from the parent's
snapshot and calls setup_gms() so that the child's own ModelLoader.load is
patched before TRT-LLM invokes it.
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
_EXTRA_CONFIG_ENV = "DYN_GMS_TRTLLM_EXTRA_CONFIG_JSON"
# Env vars the children must inherit for the child-side setup_gms to behave
# identically to the parent. Upstream MpiPoolSession only forwards TRTLLM_*/TLLM_*.
_PROPAGATED_ENV_VARS: tuple[str, ...] = (
    _ENABLED_ENV,
    _EXTRA_CONFIG_ENV,
    "CONTAINER_NAME",
    "DYN_GMS_SHADOW_MODE",
    "ENGINE_ID",
    "FAILOVER_LOCK_PATH",
    "GMS_SOCKET_DIR",
)

_extra_config_json: Optional[str] = None
_bootstrap_installed = False
_executor_finalize_hook_installed = False
_shadow_activation_fd: int | None = None
_comm_task_bootstrapped = False


def set_extra_config(extra: "dict[str, Any] | None") -> None:
    """Stash the model_loader_extra_config so MPI children see the same lock mode."""
    global _extra_config_json
    _extra_config_json = json.dumps(extra) if extra else None
    if _extra_config_json is None:
        os.environ.pop(_EXTRA_CONFIG_ENV, None)
    else:
        os.environ[_EXTRA_CONFIG_ENV] = _extra_config_json


def _current_bootstrap_context() -> tuple[dict[str, str], Optional[str]]:
    env_snapshot = {
        key: os.environ[key] for key in _PROPAGATED_ENV_VARS if key in os.environ
    }
    extra_config_json = _extra_config_json or env_snapshot.get(_EXTRA_CONFIG_ENV)
    return env_snapshot, extra_config_json


def _delay_commit_until_engine_init(extra: "dict[str, Any] | None") -> bool:
    return bool(extra and extra.get("gms_delay_commit_until_engine_init") is True)


def _get_shadow_engine_id() -> str:
    engine_id = os.environ.get("ENGINE_ID")
    if engine_id:
        return engine_id.removeprefix("engine-")

    container_name = os.environ.get("CONTAINER_NAME", "")
    if container_name.startswith("engine-"):
        return container_name.removeprefix("engine-")

    return "0"


def _is_shadow_standby(extra: "dict[str, Any] | None" = None) -> bool:
    if os.environ.get("DYN_GMS_SHADOW_MODE") == "1":
        return _get_shadow_engine_id() != "0"

    return bool(
        extra and extra.get("gms_read_only") is True and _get_shadow_engine_id() != "0"
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
        shadow_engine_id = _get_shadow_engine_id()
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
    original_try_prepare_estimation = (
        _py_executor_creator.KvCacheCreator.try_prepare_estimation
    )
    original_build_managers = _py_executor_creator.KvCacheCreator.build_managers
    original_teardown_managers = _py_executor_creator.KvCacheCreator.teardown_managers
    original_gc_collect = _py_executor_creator.gc.collect

    @wraps(original_create_py_executor)
    def patched_create_py_executor(*args, **kwargs):
        extra = json.loads(_extra_config_json) if _extra_config_json else None
        delay_finalize = _delay_commit_until_engine_init(extra)
        shadow_standby = _is_shadow_standby(extra)

        saw_estimating_build = False
        temp_estimator_executor_created = False
        wait_after_gc = False
        waited_for_activation = False
        active_temp_estimator_clamp: dict[str, object] | None = None

        def remember_attr(
            saved_attrs: list[tuple[object, str, object]],
            seen_attrs: set[tuple[int, str]],
            obj: object | None,
            attr: str,
            value: object,
        ) -> None:
            if obj is None or not hasattr(obj, attr):
                return

            key = (id(obj), attr)
            if key not in seen_attrs:
                saved_attrs.append((obj, attr, getattr(obj, attr)))
                seen_attrs.add(key)
            setattr(obj, attr, value)

        def clamp_engine_max_num_tokens(
            saved_attrs: list[tuple[object, str, object]],
            seen_attrs: set[tuple[int, str]],
            engine: object | None,
            clamp_tokens: int,
        ) -> None:
            if engine is None:
                return

            if hasattr(engine, "max_num_tokens"):
                remember_attr(
                    saved_attrs,
                    seen_attrs,
                    engine,
                    "max_num_tokens",
                    min(getattr(engine, "max_num_tokens"), clamp_tokens),
                )

            llm_args = getattr(engine, "llm_args", None)
            if llm_args is not None and hasattr(llm_args, "max_num_tokens"):
                remember_attr(
                    saved_attrs,
                    seen_attrs,
                    llm_args,
                    "max_num_tokens",
                    min(getattr(llm_args, "max_num_tokens"), clamp_tokens),
                )

        def reset_engine_cached_metadata(engine: object | None) -> None:
            if engine is None:
                return

            if getattr(engine, "attn_metadata", None) is not None:
                llm_args = getattr(engine, "llm_args", None)
                if (
                    llm_args is not None
                    and getattr(llm_args, "cuda_graph_config", None) is not None
                    and hasattr(engine, "_release_cuda_graphs")
                ):
                    engine._release_cuda_graphs()
                engine.attn_metadata = None

            if hasattr(engine, "spec_metadata"):
                engine.spec_metadata = None

        def apply_temp_estimator_clamp(kv_cache_creator: object) -> None:
            nonlocal active_temp_estimator_clamp

            if active_temp_estimator_clamp is not None:
                return

            clamp_tokens = min(
                getattr(kv_cache_creator, "_max_num_tokens"),
                getattr(kv_cache_creator, "_max_batch_size"),
            )
            saved_attrs: list[tuple[object, str, object]] = []
            seen_attrs: set[tuple[int, str]] = set()

            remember_attr(
                saved_attrs,
                seen_attrs,
                kv_cache_creator,
                "_max_num_tokens",
                clamp_tokens,
            )
            if hasattr(kv_cache_creator, "_dummy_reqs"):
                kv_cache_creator._dummy_reqs = None

            llm_args = getattr(kv_cache_creator, "_llm_args", None)
            if llm_args is not None and hasattr(llm_args, "max_num_tokens"):
                remember_attr(
                    saved_attrs,
                    seen_attrs,
                    llm_args,
                    "max_num_tokens",
                    min(getattr(llm_args, "max_num_tokens"), clamp_tokens),
                )

            engines = [
                getattr(kv_cache_creator, "_model_engine", None),
                getattr(kv_cache_creator, "_draft_model_engine", None),
            ]
            for engine in engines:
                clamp_engine_max_num_tokens(
                    saved_attrs, seen_attrs, engine, clamp_tokens
                )

            active_temp_estimator_clamp = {
                "tokens": clamp_tokens,
                "saved_attrs": saved_attrs,
                "kv_cache_creator": kv_cache_creator,
                "engines": [engine for engine in engines if engine is not None],
            }
            logger.info(
                "[Shadow] Clamping standby temp estimator token budget to %d",
                clamp_tokens,
            )

        def restore_temp_estimator_clamp() -> None:
            nonlocal active_temp_estimator_clamp

            if active_temp_estimator_clamp is None:
                return

            for obj, attr, value in reversed(
                active_temp_estimator_clamp["saved_attrs"]
            ):
                setattr(obj, attr, value)

            kv_cache_creator = active_temp_estimator_clamp["kv_cache_creator"]
            if hasattr(kv_cache_creator, "_dummy_reqs"):
                kv_cache_creator._dummy_reqs = None

            for engine in active_temp_estimator_clamp["engines"]:
                reset_engine_cached_metadata(engine)

            active_temp_estimator_clamp = None

        def patched_try_prepare_estimation(self, *est_args, **est_kwargs):
            if shadow_standby and active_temp_estimator_clamp is None:
                apply_temp_estimator_clamp(self)

            estimating_kv_cache = original_try_prepare_estimation(
                self, *est_args, **est_kwargs
            )
            if not estimating_kv_cache:
                restore_temp_estimator_clamp()
            return estimating_kv_cache

        def patched_build_managers(
            self, resources, estimating_kv_cache, *build_args, **build_kwargs
        ):
            nonlocal saw_estimating_build, waited_for_activation

            if shadow_standby and estimating_kv_cache:
                saw_estimating_build = True
            elif (
                shadow_standby
                and not waited_for_activation
                and not saw_estimating_build
            ):
                logger.info(
                    "[Shadow] Standby skipping KV estimation, waiting for failover lock before full KV init"
                )
                _wait_for_shadow_activation()
                waited_for_activation = True

            return original_build_managers(
                self,
                resources,
                estimating_kv_cache,
                *build_args,
                **build_kwargs,
            )

        def patched_teardown_managers(*teardown_args, **teardown_kwargs):
            nonlocal wait_after_gc

            result = original_teardown_managers(*teardown_args, **teardown_kwargs)
            if shadow_standby:
                wait_after_gc = True
            return result

        def patched_gc_collect(*collect_args, **collect_kwargs):
            nonlocal wait_after_gc, waited_for_activation

            result = original_gc_collect(*collect_args, **collect_kwargs)
            if wait_after_gc and not waited_for_activation:
                restore_temp_estimator_clamp()
                clear_cublas_workspaces = getattr(
                    getattr(getattr(_py_executor_creator, "torch", None), "_C", None),
                    "_cuda_clearCublasWorkspaces",
                    None,
                )
                if clear_cublas_workspaces is not None:
                    clear_cublas_workspaces()
                _wait_for_shadow_activation()
                waited_for_activation = True
                wait_after_gc = False
            return result

        def patched_create_py_executor_instance(*instance_args, **instance_kwargs):
            nonlocal temp_estimator_executor_created

            should_clamp_temp_estimator = (
                shadow_standby
                and active_temp_estimator_clamp is not None
                and saw_estimating_build
                and not temp_estimator_executor_created
            )

            if should_clamp_temp_estimator:
                instance_kwargs = dict(instance_kwargs)
                instance_kwargs["max_num_tokens"] = min(
                    instance_kwargs.get(
                        "max_num_tokens",
                        active_temp_estimator_clamp["tokens"],
                    ),
                    active_temp_estimator_clamp["tokens"],
                )

            executor = original_create_py_executor_instance(
                *instance_args, **instance_kwargs
            )

            if should_clamp_temp_estimator:
                temp_estimator_executor_created = True

            if not delay_finalize:
                return executor

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

        if delay_finalize or shadow_standby:
            _py_executor_creator.create_py_executor_instance = (
                patched_create_py_executor_instance
            )
        if shadow_standby:
            _py_executor_creator.KvCacheCreator.try_prepare_estimation = (
                patched_try_prepare_estimation
            )
            _py_executor_creator.KvCacheCreator.build_managers = patched_build_managers
            _py_executor_creator.KvCacheCreator.teardown_managers = (
                patched_teardown_managers
            )
            _py_executor_creator.gc.collect = patched_gc_collect
        try:
            return original_create_py_executor(*args, **kwargs)
        finally:
            restore_temp_estimator_clamp()
            _py_executor_creator.KvCacheCreator.try_prepare_estimation = (
                original_try_prepare_estimation
            )
            _py_executor_creator.KvCacheCreator.build_managers = original_build_managers
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
    from gpu_memory_service.integrations.trtllm.utils import configure_gms_lock_mode

    extra = json.loads(extra_config_json) if extra_config_json else None
    if os.environ.get("DYN_GMS_SHADOW_MODE") == "1":
        extra = configure_gms_lock_mode(extra or {})
    set_extra_config(extra)
    setup_gms(extra)
    if _delay_commit_until_engine_init(extra) or _is_shadow_standby(extra):
        install_executor_finalize_hook()
    logger.info(
        "[GMS] MPI worker init hook applied (pid=%d rank=%d)", os.getpid(), rank
    )


def run_bootstrapped_mpi_task(
    env_snapshot: dict,
    extra_config_json: Optional[str],
    task,
    *args,
    **kwargs,
):
    """Ensure MPICommSession tasks see the same bootstrap as MPIPool workers."""

    global _comm_task_bootstrapped
    if not _comm_task_bootstrapped:
        worker_init_hook(env_snapshot, extra_config_json)
        _comm_task_bootstrapped = True
    return task(*args, **kwargs)


def bootstrapped_proxy_worker_main(*args, **kwargs):
    """Run TRT-LLM's worker_main through worker_init_hook on MPIComm paths.

    External MPI worker nodes do not import Dynamo's llm_worker module before
    accepting tasks through MPICommExecutor, so the parent-side
    ``MpiCommSession.submit`` monkey-patch never reaches those processes. Patch
    the pickled task itself instead: GenerationExecutorProxy submits
    ``tensorrt_llm.executor.proxy.worker_main``, so replacing that symbol with a
    top-level wrapper ensures remote MPI workers import this module and execute
    ``worker_init_hook`` before TRT-LLM enters ``worker_main``.
    """

    env_snapshot, extra_config_json = _current_bootstrap_context()

    from tensorrt_llm.executor.worker import worker_main as original_worker_main

    return run_bootstrapped_mpi_task(
        env_snapshot,
        extra_config_json,
        original_worker_main,
        *args,
        **kwargs,
    )


def install_mpi_worker_bootstrap() -> None:
    """Monkey-patch MpiPoolSession to run worker_init_hook in every child. Idempotent."""
    global _bootstrap_installed
    if _bootstrap_installed:
        return

    try:
        import tensorrt_llm.executor.proxy as _executor_proxy
        import tensorrt_llm.llmapi.mpi_session as _mpi_session
    except ImportError as exc:
        raise RuntimeError(
            "GMS MPI worker bootstrap requires tensorrt_llm to be importable"
        ) from exc

    MpiPoolSession = _mpi_session.MpiPoolSession
    MpiCommSession = getattr(_mpi_session, "MpiCommSession", None)

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
    _executor_proxy.worker_main = bootstrapped_proxy_worker_main
    if MpiCommSession is not None:
        original_submit = MpiCommSession.submit

        @wraps(original_submit)
        def _patched_submit(self, task, *args, **kwargs):
            env_snapshot = {
                key: os.environ[key]
                for key in _PROPAGATED_ENV_VARS
                if key in os.environ
            }
            return original_submit(
                self,
                run_bootstrapped_mpi_task,
                env_snapshot,
                _extra_config_json,
                task,
                *args,
                **kwargs,
            )

        MpiCommSession.submit = _patched_submit
    _bootstrap_installed = True
    logger.info(
        "[GMS] Patched TensorRT-LLM MPI sessions and proxy worker_main to bootstrap GMS in MPI workers"
    )
