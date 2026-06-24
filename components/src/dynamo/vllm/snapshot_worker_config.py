# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM Snapshot worker configuration helpers."""

import importlib
import logging
import os
import sys
from typing import Any

from dynamo.common.snapshot.constants import SNAPSHOT_CONTROL_DIR_ENV

logger = logging.getLogger(__name__)

DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER = "DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER"
DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES = "DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES"
DYN_VLLM_NO_NCCL_SNAPSHOT = "DYN_VLLM_NO_NCCL_SNAPSHOT"
SNAPSHOT_WORKER_CLASS = "dynamo.vllm.snapshot_worker.SnapshotWorker"
GMS_WORKER_CLASS = "gpu_memory_service.integrations.vllm.worker.GMSWorker"
_ALLOWED_EXISTING_WORKER_CLS = (None, "", "auto")
_REQUIRED_NO_NCCL_ENV = {
    "VLLM_DISABLE_PYNCCL": "1",
    "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
    "VLLM_USE_NCCL_SYMM_MEM": "0",
    "VLLM_DISTRIBUTED_USE_SPLIT_GROUP": "0",
}
FLASHINFER_NVLINK_ALL2ALL_BACKENDS = frozenset(
    {
        "flashinfer_all2allv",
        "flashinfer_nvlink_one_sided",
        "flashinfer_nvlink_two_sided",
    }
)
DEFAULT_NO_NCCL_ALL2ALL_BACKEND = "flashinfer_nvlink_one_sided"


def flashinfer_only_collectives_enabled() -> bool:
    return os.environ.get(DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES) == "1"


def no_nccl_snapshot_mode_enabled() -> bool:
    """Return whether Dynamo should force vLLM's snapshot-safe no-NCCL path."""
    return (
        _explicit_no_nccl_snapshot_mode_enabled()
        or flashinfer_only_collectives_enabled()
    )


def flashinfer_snapshot_worker_config_required() -> bool:
    return (
        os.environ.get(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER) == "1"
        and _flashinfer_snapshot_worker_mode_enabled()
    )


def configure_flashinfer_snapshot_worker(config: Any) -> bool:
    """Install Dynamo's FlashInfer-aware Snapshot worker when opted in.

    Args:
        config: Dynamo vLLM config with an ``engine_args`` object.

    Returns:
        ``True`` when ``worker_cls`` was set, otherwise ``False``.

    Raises:
        ValueError: If the opt-in conflicts with an existing worker class or
            GMS load-format worker.
    """
    if os.environ.get(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER) != "1":
        return False

    engine_args = getattr(config, "engine_args", None)
    if engine_args is None:
        raise ValueError(
            "DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER=1 requires vLLM engine_args."
        )

    if getattr(engine_args, "load_format", None) == "gms":
        raise ValueError(
            "DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER=1 is incompatible with "
            "--load-format gms because GMS also installs a custom worker_cls."
        )

    existing_worker_cls = getattr(engine_args, "worker_cls", None)
    if existing_worker_cls == SNAPSHOT_WORKER_CLASS:
        _validate_snapshot_worker_class(SNAPSHOT_WORKER_CLASS)
        logger.info(
            "vLLM Snapshot worker_cls already configured: %s",
            SNAPSHOT_WORKER_CLASS,
        )
        return True

    if existing_worker_cls not in _ALLOWED_EXISTING_WORKER_CLS:
        raise ValueError(
            "DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER=1 cannot override existing "
            f"vLLM worker_cls={existing_worker_cls!r}."
        )

    _validate_snapshot_worker_class(SNAPSHOT_WORKER_CLASS)
    engine_args.worker_cls = SNAPSHOT_WORKER_CLASS
    logger.info("Configured vLLM Snapshot worker_cls=%s", SNAPSHOT_WORKER_CLASS)
    return True


def configure_flashinfer_snapshot_worker_before_engine_config(config: Any) -> bool:
    """Install SnapshotWorker before vLLM freezes worker_cls into VllmConfig.

    Normal serving paths call this before ``create_engine_config``. To avoid
    changing non-strict serving pods that merely inherit the snapshot-worker
    env, it only installs outside snapshot mode when explicit no-NCCL snapshot
    mode is enabled.
    """
    if _flashinfer_snapshot_worker_mode_enabled():
        configured = configure_flashinfer_snapshot_worker(config)
        if no_nccl_snapshot_mode_enabled() and not configured:
            raise ValueError(
                f"{_no_nccl_snapshot_mode_name()}=1 requires "
                f"{DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER}=1 so vLLM child "
                "workers instantiate Dynamo's SnapshotWorker and install the "
                "worker-local no-NCCL guard before vLLM Worker initialization."
            )
        return configured
    return False


def configure_no_nccl_snapshot_before_engine_config(config: Any) -> bool:
    """Apply no-NCCL snapshot settings before vLLM creates ``VllmConfig``."""
    if not no_nccl_snapshot_mode_enabled():
        return False

    configured = configure_flashinfer_snapshot_worker_before_engine_config(config)
    if not configured:
        raise ValueError(
            f"{_no_nccl_snapshot_mode_name()}=1 requires "
            f"{DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER}=1."
        )

    engine_args = getattr(config, "engine_args", None)
    if engine_args is None:
        raise ValueError(f"{_no_nccl_snapshot_mode_name()}=1 requires vLLM engine_args.")

    _set_engine_arg_if_present(engine_args, "disable_custom_all_reduce", True)
    _set_engine_arg_if_present(
        engine_args, "disable_nccl_for_dp_synchronization", True
    )
    if _explicit_no_nccl_snapshot_mode_enabled():
        _force_no_nccl_all2all_backend(engine_args)
    configure_no_nccl_snapshot_env()

    # Patch the in-process/default platform too. SnapshotWorker repeats this
    # in each spawned vLLM worker before Worker.__init__ initializes dist.
    from .flashinfer_collectives import patch_vllm_distributed_backend_for_snapshot

    patch_vllm_distributed_backend_for_snapshot()
    return True


def configure_no_nccl_snapshot_env() -> None:
    """Force vLLM env vars that prevent PyNCCL/NCCL symmetric-memory setup."""
    reason = f"{_no_nccl_snapshot_mode_name()}=1"
    for name, value in _REQUIRED_NO_NCCL_ENV.items():
        current = os.environ.get(name)
        if current is not None and current != value:
            logger.warning(
                "%s forcing %s=%s for snapshot-safe no-NCCL serving "
                "(was %r).",
                reason,
                name,
                value,
                current,
            )
        os.environ[name] = value
    _clear_vllm_env_cache()


def validate_no_nccl_snapshot_config(engine_args: Any, vllm_config: Any) -> None:
    """Fail closed if vLLM did not preserve the no-NCCL snapshot settings."""
    if not no_nccl_snapshot_mode_enabled():
        return

    validate_flashinfer_snapshot_worker_config(engine_args, vllm_config)

    parallel_config = getattr(vllm_config, "parallel_config", None)
    if parallel_config is None:
        raise ValueError(
            f"{_no_nccl_snapshot_mode_name()}=1 requires "
            "vllm_config.parallel_config."
        )

    _require_bool(
        parallel_config,
        "disable_custom_all_reduce",
        True,
        f"{_no_nccl_snapshot_mode_name()}=1",
    )
    _require_bool(
        parallel_config,
        "disable_nccl_for_dp_synchronization",
        True,
        f"{_no_nccl_snapshot_mode_name()}=1",
    )
    if _explicit_no_nccl_snapshot_mode_enabled():
        _require_no_nccl_all2all_backend(parallel_config)

    for name, value in _REQUIRED_NO_NCCL_ENV.items():
        if os.environ.get(name) != value:
            raise ValueError(
                f"{_no_nccl_snapshot_mode_name()}=1 requires {name}={value!r}; "
                f"got {os.environ.get(name)!r}."
            )


def configure_gms_worker_cls(engine_args: Any) -> None:
    existing_worker_cls = getattr(engine_args, "worker_cls", None)
    if existing_worker_cls not in _ALLOWED_EXISTING_WORKER_CLS:
        if existing_worker_cls == GMS_WORKER_CLASS:
            return
        raise ValueError(
            "--load-format gms cannot override existing vLLM "
            f"worker_cls={existing_worker_cls!r} with {GMS_WORKER_CLASS!r}."
        )
    engine_args.worker_cls = GMS_WORKER_CLASS


def validate_flashinfer_snapshot_worker_config(
    engine_args: Any, vllm_config: Any
) -> None:
    if not flashinfer_snapshot_worker_config_required():
        return

    engine_worker_cls = getattr(engine_args, "worker_cls", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    parallel_worker_cls = getattr(parallel_config, "worker_cls", None)
    if (
        engine_worker_cls == SNAPSHOT_WORKER_CLASS
        and parallel_worker_cls == SNAPSHOT_WORKER_CLASS
    ):
        return

    raise ValueError(
        f"{DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER}=1 requires vLLM "
        f"worker_cls={SNAPSHOT_WORKER_CLASS!r} after create_engine_config; "
        f"actual engine_args.worker_cls={engine_worker_cls!r}, "
        f"vllm_config.parallel_config.worker_cls={parallel_worker_cls!r}."
    )


def _require_no_nccl_all2all_backend(parallel_config: Any) -> None:
    backend = getattr(parallel_config, "all2all_backend", None)
    if backend == DEFAULT_NO_NCCL_ALL2ALL_BACKEND:
        return

    raise ValueError(
        f"{DYN_VLLM_NO_NCCL_SNAPSHOT}=1 requires "
        f"parallel_config.all2all_backend={DEFAULT_NO_NCCL_ALL2ALL_BACKEND!r} "
        "for the TP=1 DEP/EP snapshot PoC; got "
        f"{backend!r}."
    )


def _validate_snapshot_worker_class(worker_cls: str) -> None:
    module_name, _, class_name = worker_cls.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"Invalid vLLM Snapshot worker_cls={worker_cls!r}.")

    try:
        module = importlib.import_module(module_name)
        configured_cls = getattr(module, class_name)
    except Exception as exc:
        raise ValueError(
            "DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER=1 configured "
            f"worker_cls={worker_cls!r}, but it could not be imported: {exc}"
        ) from exc

    try:
        gpu_worker = importlib.import_module("vllm.v1.worker.gpu_worker")
        base_worker_cls = getattr(gpu_worker, "Worker")
    except Exception as exc:
        logger.debug(
            "Could not import vLLM Worker base class while validating %s: %s",
            worker_cls,
            exc,
        )
        return

    try:
        is_worker_subclass = issubclass(configured_cls, base_worker_cls)
    except TypeError as exc:
        raise ValueError(
            f"Configured vLLM Snapshot worker_cls={worker_cls!r} is not a class."
        ) from exc

    if not is_worker_subclass:
        raise ValueError(
            f"Configured vLLM Snapshot worker_cls={worker_cls!r} does not "
            "subclass vllm.v1.worker.gpu_worker.Worker."
        )


def _flashinfer_snapshot_worker_mode_enabled() -> bool:
    return bool(os.environ.get(SNAPSHOT_CONTROL_DIR_ENV)) or (
        no_nccl_snapshot_mode_enabled()
    )


def _set_engine_arg_if_present(engine_args: Any, name: str, value: bool) -> None:
    if engine_args is None or not hasattr(engine_args, name):
        return
    try:
        setattr(engine_args, name, value)
    except Exception:
        logger.debug("Could not set engine_args.%s=%s.", name, value)


def _force_no_nccl_all2all_backend(engine_args: Any) -> None:
    if engine_args is None:
        return

    try:
        setattr(engine_args, "all2all_backend", DEFAULT_NO_NCCL_ALL2ALL_BACKEND)
    except Exception:
        logger.debug(
            "Could not set engine_args.all2all_backend=%s.",
            DEFAULT_NO_NCCL_ALL2ALL_BACKEND,
        )
        return

    logger.info(
        "%s=1 forced vLLM --all2all-backend=%s for the TP=1 DEP/EP "
        "snapshot PoC.",
        _no_nccl_snapshot_mode_name(),
        DEFAULT_NO_NCCL_ALL2ALL_BACKEND,
    )


def _require_bool(config: Any, name: str, value: bool, reason: str) -> None:
    actual = getattr(config, name, None)
    if actual is not value:
        raise ValueError(
            f"{reason} requires parallel_config.{name}={value!r}; "
            f"got {actual!r}."
        )


def _clear_vllm_env_cache() -> None:
    envs = sys.modules.get("vllm.envs")
    if envs is None:
        return
    cache_clear = getattr(getattr(envs, "__getattr__", None), "cache_clear", None)
    if callable(cache_clear):
        cache_clear()


def _no_nccl_snapshot_mode_name() -> str:
    if os.environ.get(DYN_VLLM_NO_NCCL_SNAPSHOT) == "1":
        return DYN_VLLM_NO_NCCL_SNAPSHOT
    return DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES


def _explicit_no_nccl_snapshot_mode_enabled() -> bool:
    return os.environ.get(DYN_VLLM_NO_NCCL_SNAPSHOT) == "1"
