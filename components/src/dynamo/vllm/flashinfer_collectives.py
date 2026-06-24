# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict FlashInfer-only vLLM collective guard for snapshot mode."""

from __future__ import annotations

import functools
import importlib
import json
import logging
import math
import os
import sys
from collections.abc import Callable, Iterator
from typing import Any

from .snapshot_worker_config import (
    DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES,
    DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER,
    FLASHINFER_NVLINK_ALL2ALL_BACKENDS,
    configure_no_nccl_snapshot_env,
    flashinfer_only_collectives_enabled,
    no_nccl_snapshot_mode_enabled,
    validate_flashinfer_snapshot_worker_config,
)

logger = logging.getLogger(__name__)

_REQUIRED_VLLM_ENV = {
    "VLLM_ALLREDUCE_USE_FLASHINFER": "1",
    "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
    "VLLM_USE_NCCL_SYMM_MEM": "0",
    "VLLM_DISABLE_PYNCCL": "1",
}
_DEFAULT_FLASHINFER_ALLREDUCE_THRESHOLD_MB = 128
_VLLM_FLASHINFER_ALLREDUCE_THRESHOLDS_ENV = (
    "VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB"
)
_DYN_VLLM_FLASHINFER_AVOID_NCCL_PROCESS_GROUP = (
    "DYN_VLLM_FLASHINFER_AVOID_NCCL_PROCESS_GROUP"
)
_BLOCKED_CUDA_COLLECTIVES = (
    "all_gather",
    "all_gatherv",
    "broadcast",
    "gather",
    "reduce_scatter",
    "reduce_scatterv",
    "recv",
    "send",
    "batch_isend_irecv",
)
_TORCH_DISTRIBUTED_COLLECTIVES = (
    "all_reduce",
    "broadcast",
    "gather",
    "send",
    "recv",
    "isend",
    "irecv",
    "all_gather",
    "all_gather_into_tensor",
    "all_to_all",
    "all_to_all_single",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "scatter",
    "reduce",
    "batch_isend_irecv",
)
_CUDA_COMMUNICATOR_MODULE = "vllm.distributed.device_communicators.cuda_communicator"
_TORCH_DISTRIBUTED_MODULE = "torch.distributed"
_GUARD_INSTALLED_ATTR = "_dynamo_flashinfer_only_collectives_installed"
_TORCH_DIST_GUARD_INSTALLED_ATTR = (
    "_dynamo_flashinfer_only_collectives_torch_dist_installed"
)
_NO_NCCL_CUDA_GUARD_INSTALLED_ATTR = "_dynamo_no_nccl_snapshot_guard_installed"
_NO_NCCL_TORCH_GUARD_INSTALLED_ATTR = (
    "_dynamo_no_nccl_snapshot_torch_dist_installed"
)
_VLLM_PLATFORM_BACKEND_PATCHED_ATTR = "_dynamo_flashinfer_gloo_backend_patched"
_VLLM_PLATFORM_ORIGINAL_BACKEND_ATTR = "_dynamo_original_dist_backend"
_VLLM_PASSCONFIG_PATCHED_ATTR = "_dynamo_flashinfer_threshold_patch_installed"
_VLLM_PASSCONFIG_ORIGINAL_DEFAULTS_ATTR = (
    "_dynamo_original_default_fi_allreduce_fusion_max_size_mb"
)
_ORIGINAL_ALL_REDUCE_ATTR = "_dynamo_original_all_reduce"
_ORIGINAL_COLLECTIVE_PREFIX = "_dynamo_original_"
_ORIGINAL_TORCH_DIST_PREFIX = "_dynamo_original_"
_TORCH_DIST_GROUP_ARG_INDEX = {
    "all_reduce": 2,
    "broadcast": 2,
    "gather": 3,
    "send": 2,
    "recv": 2,
    "isend": 2,
    "irecv": 2,
    "all_gather": 2,
    "all_gather_into_tensor": 2,
    "all_to_all": 2,
    "all_to_all_single": 2,
    "reduce_scatter": 3,
    "reduce_scatter_tensor": 3,
    "scatter": 3,
    "reduce": 3,
}


def configure_flashinfer_only_collectives(
    engine_args: Any, vllm_config: Any
) -> bool:
    """Configure vLLM for strict FlashInfer-only CUDA collectives.

    This is intentionally opt-in for the FlashInfer snapshot E2E. It mutates
    the already-created vLLM config before ``AsyncLLM`` construction and
    installs a guard that rejects NCCL/PyNCCL/custom-allreduce fallbacks.

    Args:
        engine_args: vLLM engine arguments used to build ``vllm_config``.
        vllm_config: vLLM config returned by ``create_engine_config``.

    Returns:
        ``True`` when strict mode was enabled, otherwise ``False``.

    Raises:
        ValueError: If strict mode is enabled without the Snapshot worker hook
            that installs the child-process guard.
        ValueError: If strict mode cannot prove the configured all2all path is
            FlashInfer NVLink when MoE all2all is relevant.
        RuntimeError: If the CUDA communicator guard cannot be installed.
    """
    if not flashinfer_only_collectives_enabled():
        return False

    if os.environ.get(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER) != "1":
        raise ValueError(
            f"{DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES}=1 requires "
            f"{DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER}=1 so vLLM child workers "
            "instantiate Dynamo's SnapshotWorker and install the same strict "
            "collective guard before CUDA communicator use."
        )

    parallel_config = getattr(vllm_config, "parallel_config", None)
    if parallel_config is None:
        raise ValueError(
            f"{DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES}=1 requires "
            "vllm_config.parallel_config."
        )

    validate_flashinfer_snapshot_worker_config(engine_args, vllm_config)

    _set_config_flag(
        parallel_config,
        "disable_custom_all_reduce",
        True,
        f"{DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES}=1",
    )
    _set_config_flag(
        parallel_config,
        "disable_nccl_for_dp_synchronization",
        True,
        f"{DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES}=1",
    )
    _set_engine_arg_if_present(engine_args, "disable_custom_all_reduce", True)
    _set_engine_arg_if_present(engine_args, "disable_nccl_for_dp_synchronization", True)

    _validate_strict_all2all_backend(parallel_config, vllm_config)
    _configure_vllm_collective_env(parallel_config, engine_args)
    patch_vllm_flashinfer_allreduce_thresholds()
    patch_vllm_distributed_backend_for_snapshot()
    install_strict_flashinfer_collective_guard()

    logger.info(
        "Configured strict FlashInfer-only vLLM collectives for snapshot mode "
        "(tp=%s, dp=%s, all2all_backend=%r).",
        getattr(parallel_config, "tensor_parallel_size", None),
        getattr(parallel_config, "data_parallel_size", None),
        getattr(parallel_config, "all2all_backend", None),
    )
    return True


def install_strict_flashinfer_collective_guard() -> bool:
    """Patch vLLM/torch CUDA collective paths to fail closed.

    Returns:
        ``True`` if this call installed any strict-mode patch, ``False`` if
        all currently importable vLLM/torch objects were already patched or
        unavailable.
    """
    threshold_installed = patch_vllm_flashinfer_allreduce_thresholds()
    backend_patched = patch_vllm_distributed_backend_for_snapshot()
    communicator_installed = _install_cuda_communicator_guard()
    torch_dist_installed = _install_torch_distributed_guard()
    return (
        threshold_installed
        or backend_patched
        or communicator_installed
        or torch_dist_installed
    )


def install_no_nccl_snapshot_guard() -> bool:
    """Install concise fail-closed guards for no-NCCL snapshot mode."""
    configure_no_nccl_snapshot_env()
    backend_patched = patch_vllm_distributed_backend_for_snapshot()
    communicator_guard = _install_no_nccl_cuda_communicator_guard()
    torch_guard = _install_no_nccl_torch_backend_guard()
    return backend_patched or communicator_guard or torch_guard


def patch_vllm_flashinfer_allreduce_thresholds() -> bool:
    """Merge vLLM FlashInfer allreduce thresholds with the env override.

    vLLM parses ``VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB`` but this
    checkout's FlashInfer allreduce communicator reads
    ``PassConfig.default_fi_allreduce_fusion_max_size_mb()`` directly. Patch
    that static method so runtime communicator construction sees the same
    override as the compilation config.

    Returns:
        ``True`` when this call installed the patch, ``False`` if it was
        already installed or vLLM's ``PassConfig`` is not importable yet.

    Raises:
        ValueError: If the env var is present but not a JSON object mapping
            world sizes to positive numeric MB thresholds.
    """
    _parse_flashinfer_threshold_overrides()

    try:
        compilation_config = importlib.import_module("vllm.config.compilation")
        pass_config_cls = getattr(compilation_config, "PassConfig")
    except Exception:
        logger.debug(
            "vLLM PassConfig is not importable; skipping FlashInfer "
            "allreduce threshold patch for now.",
            exc_info=True,
        )
        return False

    if getattr(pass_config_cls, _VLLM_PASSCONFIG_PATCHED_ATTR, False):
        return False

    original = getattr(pass_config_cls, "default_fi_allreduce_fusion_max_size_mb")
    setattr(pass_config_cls, _VLLM_PASSCONFIG_ORIGINAL_DEFAULTS_ATTR, original)

    def default_fi_allreduce_fusion_max_size_mb() -> dict[int, float]:
        defaults = _call_vllm_threshold_default(original)
        overrides = _parse_flashinfer_threshold_overrides()
        if not overrides:
            return defaults
        return {**defaults, **overrides}

    pass_config_cls.default_fi_allreduce_fusion_max_size_mb = staticmethod(
        default_fi_allreduce_fusion_max_size_mb
    )
    setattr(pass_config_cls, _VLLM_PASSCONFIG_PATCHED_ATTR, True)

    logger.info(
        "Patched vLLM PassConfig.default_fi_allreduce_fusion_max_size_mb to "
        "honor %s=%r.",
        _VLLM_FLASHINFER_ALLREDUCE_THRESHOLDS_ENV,
        os.environ.get(_VLLM_FLASHINFER_ALLREDUCE_THRESHOLDS_ENV),
    )
    return True


def patch_vllm_distributed_backend_for_snapshot() -> bool:
    """Patch vLLM's worker process-group backend from NCCL to Gloo.

    Snapshot no-NCCL mode keeps model traffic on non-NCCL paths (for DEP, the
    FlashInfer MoE all2all backend) while vLLM bootstrap/model-parallel torch
    process groups use CPU/Gloo so vLLM does not allocate NCCL state before
    checkpointing.

    Returns:
        ``True`` when a vLLM platform backend was changed to ``"gloo"``.
        ``False`` when disabled, already patched, already safe, or vLLM's
        platform module is not importable yet.
    """
    if not _avoid_nccl_process_group_enabled():
        logger.info(
            "%s=0; leaving vLLM platform distributed backend unchanged.",
            _DYN_VLLM_FLASHINFER_AVOID_NCCL_PROCESS_GROUP,
        )
        return False

    try:
        platforms = importlib.import_module("vllm.platforms")
        current_platform = getattr(platforms, "current_platform")
    except Exception:
        logger.debug(
            "vLLM current_platform is not importable; skipping strict "
            "FlashInfer snapshot distributed backend patch for now.",
            exc_info=True,
        )
        return False

    patched = False
    for target in _platform_backend_patch_targets(current_platform):
        if getattr(target, _VLLM_PLATFORM_BACKEND_PATCHED_ATTR, False):
            continue
        backend = getattr(target, "dist_backend", None)
        if backend == "gloo":
            continue
        try:
            setattr(target, _VLLM_PLATFORM_ORIGINAL_BACKEND_ATTR, backend)
            setattr(target, "dist_backend", "gloo")
            setattr(target, _VLLM_PLATFORM_BACKEND_PATCHED_ATTR, True)
        except Exception as exc:
            raise RuntimeError(
                "Dynamo no-NCCL vLLM snapshot mode could not patch "
                "vLLM platform distributed backend to 'gloo'."
            ) from exc
        patched = True
        logger.warning(
            "Dynamo no-NCCL vLLM snapshot mode patched %s.dist_backend "
            "from %r to 'gloo' to avoid NCCL process-group initialization. "
            "Use vLLM --all2all-backend=flashinfer_nvlink_one_sided for "
            "DEP/EP MoE traffic instead of TP allreduce for this path.",
            _qualified_type_name(target),
            backend,
        )

    return patched


def _install_no_nccl_cuda_communicator_guard() -> bool:
    try:
        cuda_communicator = importlib.import_module(_CUDA_COMMUNICATOR_MODULE)
        communicator_cls = getattr(cuda_communicator, "CudaCommunicator")
    except Exception as exc:
        raise RuntimeError(
            "Dynamo no-NCCL vLLM snapshot mode could not install the vLLM "
            "CUDA communicator guard."
        ) from exc

    if getattr(communicator_cls, _NO_NCCL_CUDA_GUARD_INSTALLED_ATTR, False):
        return False

    original_init = communicator_cls.__init__

    @functools.wraps(original_init)
    def init_without_nccl(self: Any, *args: Any, **kwargs: Any) -> None:
        configure_no_nccl_snapshot_env()
        _validate_no_nccl_vllm_globals()
        original_init(self, *args, **kwargs)
        _validate_no_nccl_cuda_communicator(self)

    setattr(
        communicator_cls,
        f"{_ORIGINAL_COLLECTIVE_PREFIX}no_nccl_init",
        original_init,
    )
    communicator_cls.__init__ = init_without_nccl
    setattr(communicator_cls, _NO_NCCL_CUDA_GUARD_INSTALLED_ATTR, True)
    logger.info("Installed Dynamo no-NCCL vLLM CUDA communicator guard.")
    return True


def _install_no_nccl_torch_backend_guard() -> bool:
    try:
        distributed = importlib.import_module(_TORCH_DISTRIBUTED_MODULE)
    except Exception:
        logger.debug("torch.distributed is not importable; skipping no-NCCL guard.")
        return False

    installed_any = _install_no_nccl_torch_backend_guard_on_module(distributed)

    c10d = getattr(distributed, "distributed_c10d", None)
    if c10d is None:
        try:
            c10d = importlib.import_module("torch.distributed.distributed_c10d")
        except Exception:
            c10d = None
    if c10d is not None and c10d is not distributed:
        installed_any = _install_no_nccl_torch_backend_guard_on_module(c10d) or (
            installed_any
        )

    setattr(distributed, _NO_NCCL_TORCH_GUARD_INSTALLED_ATTR, True)
    logger.info("Installed Dynamo no-NCCL torch.distributed backend guard.")
    return installed_any


def _install_no_nccl_torch_backend_guard_on_module(module: Any) -> bool:
    if getattr(module, _NO_NCCL_TORCH_GUARD_INSTALLED_ATTR, False):
        return False

    installed_any = False
    for function_name, backend_arg_index in (
        ("init_process_group", 0),
        ("new_group", 2),
        ("split_group", 2),
    ):
        original = getattr(module, function_name, None)
        if original is None:
            continue
        setattr(
            module,
            f"{_ORIGINAL_TORCH_DIST_PREFIX}no_nccl_{function_name}",
            original,
        )
        setattr(
            module,
            function_name,
            _make_no_nccl_torch_backend_guard(
                f"{module.__name__}.{function_name}",
                original,
                backend_arg_index,
            ),
        )
        installed_any = True

    setattr(module, _NO_NCCL_TORCH_GUARD_INSTALLED_ATTR, True)
    return installed_any


def _make_no_nccl_torch_backend_guard(
    function_name: str,
    original: Callable[..., Any],
    backend_arg_index: int,
) -> Callable[..., Any]:
    @functools.wraps(original)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        backend = _backend_arg(args, kwargs, backend_arg_index)
        normalized = _normalize_backend(backend)
        if normalized is not None and "nccl" in normalized:
            raise RuntimeError(
                "Dynamo no-NCCL vLLM snapshot mode blocked "
                f"{function_name}(backend={backend!r})."
            )
        if function_name.endswith(".init_process_group") and backend is None:
            device_id = _device_id_arg(args, kwargs)
            if device_id is not None and _value_indicates_cuda(device_id):
                raise RuntimeError(
                    "Dynamo no-NCCL vLLM snapshot mode blocked "
                    f"{function_name}(backend=None, device_id={device_id!r}) "
                    "because PyTorch may choose and eager-initialize a CUDA "
                    "process-group backend."
                )
        return original(*args, **kwargs)

    return guarded


def _device_id_arg(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if "device_id" in kwargs:
        return kwargs["device_id"]
    if len(args) > 8:
        return args[8]
    return None


def _install_cuda_communicator_guard() -> bool:
    try:
        cuda_communicator = importlib.import_module(_CUDA_COMMUNICATOR_MODULE)
        communicator_cls = getattr(cuda_communicator, "CudaCommunicator")
    except Exception as exc:
        raise RuntimeError(
            f"{DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES}=1 could not install "
            "the strict FlashInfer-only vLLM CUDA communicator guard."
        ) from exc

    if getattr(communicator_cls, _GUARD_INSTALLED_ATTR, False):
        return False

    setattr(communicator_cls, _ORIGINAL_ALL_REDUCE_ATTR, communicator_cls.all_reduce)
    communicator_cls.all_reduce = _strict_flashinfer_all_reduce

    for method_name in _BLOCKED_CUDA_COLLECTIVES:
        original = getattr(communicator_cls, method_name, None)
        if original is None:
            continue
        setattr(
            communicator_cls,
            f"{_ORIGINAL_COLLECTIVE_PREFIX}{method_name}",
            original,
        )
        setattr(
            communicator_cls,
            method_name,
            _make_strict_blocked_collective(method_name, original),
        )

    setattr(communicator_cls, _GUARD_INSTALLED_ATTR, True)
    logger.info(
        "Installed strict FlashInfer-only vLLM CUDA communicator guard "
        "(blocked collectives: %s).",
        ", ".join(_BLOCKED_CUDA_COLLECTIVES),
    )
    return True


def _install_torch_distributed_guard() -> bool:
    try:
        distributed = importlib.import_module(_TORCH_DISTRIBUTED_MODULE)
    except Exception:
        logger.debug(
            "torch.distributed is not importable; skipping strict "
            "FlashInfer-only torch.distributed CUDA tensor guard."
        )
        return False

    if getattr(distributed, _TORCH_DIST_GUARD_INSTALLED_ATTR, False):
        return False

    installed_any = False
    for function_name in _TORCH_DISTRIBUTED_COLLECTIVES:
        original = getattr(distributed, function_name, None)
        if original is None:
            continue
        setattr(
            distributed,
            f"{_ORIGINAL_TORCH_DIST_PREFIX}{function_name}",
            original,
        )
        setattr(
            distributed,
            function_name,
            _make_strict_torch_distributed_collective_guard(
                distributed, function_name, original
            ),
        )
        installed_any = True

    setattr(distributed, _TORCH_DIST_GUARD_INSTALLED_ATTR, True)
    logger.info(
        "Installed strict FlashInfer-only torch.distributed CUDA tensor guard "
        "(guarded functions: %s).",
        ", ".join(
            name
            for name in _TORCH_DISTRIBUTED_COLLECTIVES
            if hasattr(distributed, name)
        ),
    )
    return installed_any


def _backend_arg(
    args: tuple[Any, ...], kwargs: dict[str, Any], backend_arg_index: int
) -> Any:
    if "backend" in kwargs:
        return kwargs["backend"]
    if len(args) > backend_arg_index:
        return args[backend_arg_index]
    return None


def _validate_no_nccl_vllm_globals() -> None:
    try:
        parallel_state = importlib.import_module("vllm.distributed.parallel_state")
    except Exception:
        parallel_state = None

    if parallel_state is not None and bool(
        getattr(parallel_state, "_ENABLE_CUSTOM_ALL_REDUCE", False)
    ):
        raise RuntimeError(
            "Dynamo no-NCCL vLLM snapshot mode requires "
            "--disable-custom-all-reduce before CudaCommunicator construction."
        )

    try:
        envs = importlib.import_module("vllm.envs")
    except Exception:
        return

    for name, expected in (
        ("VLLM_DISABLE_PYNCCL", True),
        ("VLLM_ALLREDUCE_USE_SYMM_MEM", False),
        ("VLLM_USE_NCCL_SYMM_MEM", False),
    ):
        if bool(getattr(envs, name, not expected)) is not expected:
            raise RuntimeError(
                "Dynamo no-NCCL vLLM snapshot mode requires "
                f"{name}={expected!r} before CudaCommunicator construction."
            )


def _validate_no_nccl_cuda_communicator(communicator: Any) -> None:
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
            "Dynamo no-NCCL vLLM snapshot mode found active NCCL-backed "
            f"communicator(s) {active!r} in "
            f"{getattr(communicator, 'unique_name', '<unnamed>')!r}."
        )


def _set_config_flag(config: Any, name: str, value: bool, reason: str) -> None:
    try:
        setattr(config, name, value)
    except Exception as exc:
        raise ValueError(f"{reason} could not set parallel_config.{name}.") from exc

    if getattr(config, name, None) is not value:
        raise ValueError(
            f"{reason} requires parallel_config.{name}={value!r}, "
            f"got {getattr(config, name, None)!r}."
        )


def _set_engine_arg_if_present(engine_args: Any, name: str, value: bool) -> None:
    if engine_args is None or not hasattr(engine_args, name):
        return
    try:
        setattr(engine_args, name, value)
    except Exception:
        logger.debug("Could not mirror %s=%s onto engine_args.", name, value)


def _validate_strict_all2all_backend(parallel_config: Any, vllm_config: Any) -> None:
    if not _strict_moe_all2all_may_be_active(parallel_config, vllm_config):
        return

    backend = getattr(parallel_config, "all2all_backend", None)
    if backend in FLASHINFER_NVLINK_ALL2ALL_BACKENDS:
        return

    raise ValueError(
        f"{DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES}=1 requires a FlashInfer "
        "NVLink MoE all2all backend when expert-parallel MoE all2all may be "
        "active. Set --all2all-backend to one of "
        f"{sorted(FLASHINFER_NVLINK_ALL2ALL_BACKENDS)!r}; got {backend!r}."
    )


def _strict_moe_all2all_may_be_active(parallel_config: Any, vllm_config: Any) -> bool:
    if not _as_bool(getattr(parallel_config, "enable_expert_parallel", False)):
        return False

    if _strict_model_is_known_non_moe(parallel_config, vllm_config):
        return False

    return (
        _as_int(getattr(parallel_config, "data_parallel_size", 1), default=1) > 1
        and _strict_parallel_world_size(parallel_config) > 1
    )


def _strict_model_is_known_non_moe(parallel_config: Any, vllm_config: Any) -> bool:
    model_hints = (
        getattr(parallel_config, "is_moe_model", None),
        getattr(getattr(vllm_config, "model_config", None), "is_moe", None),
    )
    if any(value is True for value in model_hints):
        return False
    return any(value is False for value in model_hints)


def _strict_parallel_world_size(parallel_config: Any) -> int:
    world_size = 1
    for name in (
        "tensor_parallel_size",
        "prefill_context_parallel_size",
        "data_parallel_size",
    ):
        world_size *= _as_int(getattr(parallel_config, name, 1), default=1)
    return world_size


def _configure_vllm_collective_env(parallel_config: Any, engine_args: Any) -> None:
    for name, value in _REQUIRED_VLLM_ENV.items():
        current = os.environ.get(name)
        if current is not None and current != value:
            logger.warning(
                "%s=1 forcing %s=%s for checkpoint-safe collectives "
                "(was %r).",
                DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES,
                name,
                value,
                current,
            )
        os.environ[name] = value

    if _VLLM_FLASHINFER_ALLREDUCE_THRESHOLDS_ENV not in os.environ:
        tp_size = _tensor_parallel_size(parallel_config, engine_args)
        if tp_size > 1:
            os.environ[_VLLM_FLASHINFER_ALLREDUCE_THRESHOLDS_ENV] = json.dumps(
                {str(tp_size): _DEFAULT_FLASHINFER_ALLREDUCE_THRESHOLD_MB}
            )

    _clear_vllm_env_cache()


def _parse_flashinfer_threshold_overrides() -> dict[int, float]:
    raw = os.environ.get(_VLLM_FLASHINFER_ALLREDUCE_THRESHOLDS_ENV)
    if raw is None or raw == "":
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{_VLLM_FLASHINFER_ALLREDUCE_THRESHOLDS_ENV} must be a JSON "
            "object mapping world size to a positive numeric MB threshold; "
            f"got {raw!r}."
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            f"{_VLLM_FLASHINFER_ALLREDUCE_THRESHOLDS_ENV} must be a JSON "
            "object mapping world size to a positive numeric MB threshold; "
            f"got {type(parsed).__name__}."
        )

    overrides: dict[int, float] = {}
    for raw_world_size, raw_threshold_mb in parsed.items():
        world_size = _parse_positive_int(
            raw_world_size,
            f"{_VLLM_FLASHINFER_ALLREDUCE_THRESHOLDS_ENV} key",
        )
        threshold_mb = _parse_positive_number(
            raw_threshold_mb,
            f"{_VLLM_FLASHINFER_ALLREDUCE_THRESHOLDS_ENV}[{raw_world_size!r}]",
        )
        overrides[world_size] = threshold_mb
    return overrides


def _parse_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer, got {value!r}.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must be a positive integer, got {value!r}."
        ) from exc
    if parsed <= 0 or str(value).strip() != str(parsed):
        raise ValueError(f"{label} must be a positive integer, got {value!r}.")
    return parsed


def _parse_positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{label} must be a positive numeric MB threshold, got {value!r}."
        )
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(
            f"{label} must be a positive numeric MB threshold, got {value!r}."
        )
    return parsed


def _call_vllm_threshold_default(original: Any) -> dict[int, float]:
    defaults = original()
    if defaults is None:
        return {}
    return dict(defaults)


def _avoid_nccl_process_group_enabled() -> bool:
    raw = os.environ.get(_DYN_VLLM_FLASHINFER_AVOID_NCCL_PROCESS_GROUP)
    if os.environ.get("DYN_VLLM_NO_NCCL_SNAPSHOT") == "1":
        if raw is not None and not _as_bool(raw):
            raise ValueError(
                "DYN_VLLM_NO_NCCL_SNAPSHOT=1 requires vLLM platform "
                "distributed backend patching to Gloo; "
                f"{_DYN_VLLM_FLASHINFER_AVOID_NCCL_PROCESS_GROUP}=0 would "
                "allow NCCL process-group initialization."
            )
        return True
    if raw is not None:
        return _as_bool(raw)
    return no_nccl_snapshot_mode_enabled()


def _platform_backend_patch_targets(current_platform: Any) -> tuple[Any, ...]:
    targets: list[Any] = [current_platform]
    platform_cls = current_platform if isinstance(current_platform, type) else type(
        current_platform
    )
    if hasattr(platform_cls, "dist_backend") and platform_cls not in targets:
        targets.append(platform_cls)
    return tuple(targets)


def _qualified_type_name(value: Any) -> str:
    if isinstance(value, type):
        cls = value
    else:
        cls = value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _tensor_parallel_size(parallel_config: Any, engine_args: Any) -> int:
    return _as_int(
        getattr(
            parallel_config,
            "tensor_parallel_size",
            getattr(engine_args, "tensor_parallel_size", 1),
        ),
        default=1,
    )


def _clear_vllm_env_cache() -> None:
    envs = sys.modules.get("vllm.envs")
    if envs is None:
        return
    cache_clear = getattr(getattr(envs, "__getattr__", None), "cache_clear", None)
    if callable(cache_clear):
        cache_clear()


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _strict_flashinfer_all_reduce(self: Any, input_: Any) -> Any:
    world_size = _as_int(getattr(self, "world_size", 1), default=1)
    if world_size <= 1:
        return input_

    if not _is_tensor_parallel_group(self):
        raise RuntimeError(
            "Strict FlashInfer-only vLLM snapshot mode blocked CUDA all_reduce "
            "on a non-TP group because it would use a checkpoint-unsafe "
            "fallback. "
            f"{_strict_allreduce_diagnostics(self, input_, fi_ar_comm=None)}"
        )

    fi_ar_comm = getattr(self, "fi_ar_comm", None)
    fi_disabled = bool(getattr(fi_ar_comm, "disabled", False)) if fi_ar_comm else None
    should_use_fi_ar = False
    should_use_error: BaseException | None = None
    if fi_ar_comm is not None and not fi_disabled:
        try:
            should_use_fi_ar = bool(fi_ar_comm.should_use_fi_ar(input_))
        except Exception as exc:
            should_use_error = exc

    if fi_ar_comm is not None and not fi_disabled and should_use_fi_ar:
        out = fi_ar_comm.all_reduce(input_)
        if out is None:
            raise RuntimeError(
                "Strict FlashInfer-only vLLM snapshot mode rejected CUDA "
                "all_reduce because FlashInfer returned None. "
                f"{_strict_allreduce_diagnostics(self, input_, fi_ar_comm)}"
            )
        return out

    reason = _flashinfer_rejection_reason(
        self,
        fi_ar_comm=fi_ar_comm,
        fi_disabled=fi_disabled,
        should_use_fi_ar=should_use_fi_ar,
        should_use_error=should_use_error,
    )
    raise RuntimeError(
        "Strict FlashInfer-only vLLM snapshot mode rejected CUDA all_reduce "
        "instead of falling back to CustomAllreduce, PyNCCL, NCCL symmetric "
        f"memory, or torch.distributed. {reason}; "
        f"{_strict_allreduce_diagnostics(self, input_, fi_ar_comm)}"
    )


def _make_strict_blocked_collective(
    method_name: str, original: Callable[..., Any]
) -> Callable[..., Any]:
    @functools.wraps(original)
    def blocked(self: Any, *args: Any, **kwargs: Any) -> Any:
        world_size = _as_int(getattr(self, "world_size", 1), default=1)
        if world_size <= 1:
            if method_name == "batch_isend_irecv":
                return None
            return args[0] if args else None
        raise RuntimeError(
            "Strict FlashInfer-only vLLM snapshot mode blocked CUDA "
            f"{method_name} because it implies a checkpoint-unsafe PyNCCL/NCCL "
            f"path for world_size={world_size}. {_communicator_details(self)}"
        )

    return blocked


def _make_strict_torch_distributed_collective_guard(
    distributed: Any, function_name: str, original: Callable[..., Any]
) -> Callable[..., Any]:
    @functools.wraps(original)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        cuda_tensors = tuple(_iter_cuda_tensors(args)) + tuple(
            _iter_cuda_tensors(kwargs)
        )
        if not cuda_tensors:
            return original(*args, **kwargs)

        group = _torch_distributed_group(function_name, args, kwargs)
        backend = _torch_distributed_backend(distributed, group)
        raise RuntimeError(
            "Strict FlashInfer-only vLLM snapshot mode blocked "
            f"torch.distributed.{function_name} with CUDA tensor(s) because "
            "GPU collectives must use FlashInfer and cannot use "
            "torch.distributed, CustomAllreduce, PyNCCL, or NCCL fallback "
            "paths in this mode. "
            f"group={_group_details(group)}, backend={backend!r}, "
            f"tensors={[_tensor_details(tensor) for tensor in cuda_tensors]}"
        )

    return guarded


def _torch_distributed_group(
    function_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    if "group" in kwargs:
        return kwargs["group"]

    index = _TORCH_DIST_GROUP_ARG_INDEX.get(function_name)
    if index is not None and len(args) > index:
        return args[index]

    if function_name == "batch_isend_irecv" and args:
        groups = []
        seen_group_ids: set[int] = set()
        for op in _as_iterable(args[0]):
            group = getattr(op, "group", None)
            if group is None or id(group) in seen_group_ids:
                continue
            groups.append(group)
            seen_group_ids.add(id(group))
        if len(groups) == 1:
            return groups[0]
        if groups:
            return tuple(groups)

    return None


def _torch_distributed_backend(distributed: Any, group: Any) -> str | None:
    backend = _backend_from_group(group)
    if backend is not None:
        return backend

    get_backend = getattr(distributed, "get_backend", None)
    if not callable(get_backend):
        return None

    try:
        return _normalize_backend(get_backend(group))
    except Exception as exc:
        logger.debug("torch.distributed.get_backend(%r) failed: %s", group, exc)
        return None


def _backend_from_group(group: Any) -> str | None:
    if isinstance(group, tuple):
        normalized = tuple(_backend_from_group(item) for item in group)
        normalized = tuple(item for item in normalized if item is not None)
        if not normalized:
            return None
        if len(set(normalized)) == 1:
            return normalized[0]
        return ",".join(sorted(set(normalized)))

    for attr_name in ("backend", "_backend", "backend_name", "name"):
        try:
            value = getattr(group, attr_name, None)
        except Exception:
            continue
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        normalized = _normalize_backend(value)
        if normalized is not None:
            return normalized
    return None


def _normalize_backend(backend: Any) -> str | None:
    if backend is None:
        return None

    name = getattr(backend, "name", None)
    if isinstance(name, str):
        return name.lower()

    text = str(backend)
    if "." in text:
        text = text.rsplit(".", maxsplit=1)[-1]
    text = text.strip().strip("'\"").lower()
    return text or None


def _iter_cuda_tensors(
    value: Any, seen: set[int] | None = None
) -> Iterator[Any]:
    if seen is None:
        seen = set()

    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)

    if _is_cuda_tensor(value):
        yield value
        return

    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_cuda_tensors(item, seen)
        return

    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _iter_cuda_tensors(item, seen)
        return

    for attr_name in ("tensor", "tensors", "input", "output"):
        try:
            attr = getattr(value, attr_name, None)
        except Exception:
            continue
        if attr is not None:
            yield from _iter_cuda_tensors(attr, seen)


def _is_cuda_tensor(value: Any) -> bool:
    if not _looks_tensor_like(value):
        return False

    try:
        is_cuda = getattr(value, "is_cuda", False)
        if callable(is_cuda):
            is_cuda = is_cuda()
        if bool(is_cuda):
            return True
    except Exception:
        pass

    try:
        device = getattr(value, "device", None)
    except Exception:
        return False
    return _value_indicates_cuda(device)


def _looks_tensor_like(value: Any) -> bool:
    return any(
        hasattr(value, attr_name)
        for attr_name in ("dtype", "shape", "is_contiguous", "numel")
    )


def _value_indicates_cuda(value: Any) -> bool:
    device_type = getattr(value, "type", None)
    if isinstance(device_type, str) and device_type.lower() == "cuda":
        return True
    return str(value).lower().startswith("cuda")


def _as_iterable(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, (list, set, frozenset)):
        return tuple(value)
    return (value,)


def _group_details(group: Any) -> str:
    if group is None:
        return "default"
    if isinstance(group, tuple):
        return "[" + ", ".join(_group_details(item) for item in group) + "]"
    return (
        f"{group.__class__.__module__}.{group.__class__.__qualname__}"
        f"(name={getattr(group, 'name', None)!r}, "
        f"backend={getattr(group, 'backend', None)!r}, "
        f"device={getattr(group, 'device', None)!r}, "
        f"device_type={getattr(group, 'device_type', None)!r})"
    )


def _is_tensor_parallel_group(communicator: Any) -> bool:
    unique_name = str(getattr(communicator, "unique_name", "") or "")
    return unique_name.startswith("tp")


def _flashinfer_rejection_reason(
    communicator: Any,
    *,
    fi_ar_comm: Any,
    fi_disabled: bool | None,
    should_use_fi_ar: bool,
    should_use_error: BaseException | None,
) -> str:
    if not getattr(communicator, "use_flashinfer_allreduce", False):
        return "FlashInfer allreduce was not configured on this communicator"
    if fi_ar_comm is None:
        return "FlashInfer allreduce communicator is missing"
    if fi_disabled:
        return "FlashInfer allreduce communicator is disabled"
    if should_use_error is not None:
        return (
            "FlashInfer allreduce eligibility check raised "
            f"{should_use_error.__class__.__name__}: {should_use_error}"
        )
    if not should_use_fi_ar:
        return "FlashInfer allreduce rejected this tensor"
    return "FlashInfer allreduce was unavailable"


def _communicator_details(communicator: Any) -> str:
    return (
        "group="
        f"{getattr(communicator, 'unique_name', None)!r}, "
        f"world_size={getattr(communicator, 'world_size', None)!r}, "
        f"rank={getattr(communicator, 'rank', None)!r}, "
        f"rank_in_group={getattr(communicator, 'rank_in_group', None)!r}, "
        "use_flashinfer_allreduce="
        f"{getattr(communicator, 'use_flashinfer_allreduce', None)!r}"
    )


def _strict_allreduce_diagnostics(
    communicator: Any, tensor: Any, fi_ar_comm: Any
) -> str:
    raw_threshold_env = os.environ.get(_VLLM_FLASHINFER_ALLREDUCE_THRESHOLDS_ENV)
    try:
        effective_thresholds = _effective_flashinfer_thresholds()
    except Exception as exc:
        effective_thresholds = f"<error {exc.__class__.__name__}: {exc}>"
    return (
        f"{_communicator_details(communicator)}, "
        f"backend={_communicator_backend(communicator)!r}, "
        f"flashinfer={_flashinfer_comm_details(fi_ar_comm)}, "
        f"threshold_env_raw={raw_threshold_env!r}, "
        f"threshold_env_effective={effective_thresholds!r}; "
        f"{_tensor_details(tensor)}"
    )


def _effective_flashinfer_thresholds() -> dict[int, float]:
    try:
        compilation_config = importlib.import_module("vllm.config.compilation")
        pass_config_cls = getattr(compilation_config, "PassConfig")
        defaults_fn = getattr(
            pass_config_cls,
            "default_fi_allreduce_fusion_max_size_mb",
        )
        defaults = defaults_fn()
        return dict(defaults or {})
    except Exception:
        return _parse_flashinfer_threshold_overrides()


def _communicator_backend(communicator: Any) -> str | None:
    for attr_name in ("device_group", "cpu_group", "group"):
        backend = _backend_from_group(getattr(communicator, attr_name, None))
        if backend is not None:
            return backend

    try:
        platforms = importlib.import_module("vllm.platforms")
        current_platform = getattr(platforms, "current_platform")
        return _normalize_backend(getattr(current_platform, "dist_backend", None))
    except Exception:
        return None


def _flashinfer_comm_details(fi_ar_comm: Any) -> str:
    if fi_ar_comm is None:
        return "missing"
    return (
        f"disabled={getattr(fi_ar_comm, 'disabled', None)!r}, "
        f"world_size={getattr(fi_ar_comm, 'world_size', None)!r}, "
        f"rank={getattr(fi_ar_comm, 'rank', None)!r}, "
        "max_workspace_size="
        f"{_format_bytes(getattr(fi_ar_comm, 'max_workspace_size', None))}, "
        f"max_num_tokens={getattr(fi_ar_comm, 'max_num_tokens', None)!r}"
    )


def _tensor_details(tensor: Any) -> str:
    contiguous = getattr(tensor, "is_contiguous", None)
    if callable(contiguous):
        try:
            contiguous = contiguous()
        except Exception as exc:
            contiguous = f"<error {exc!r}>"

    numel = _tensor_numel(tensor)
    bytes_ = _tensor_nbytes(tensor, numel)
    return (
        f"dtype={getattr(tensor, 'dtype', None)!r}, "
        f"shape={getattr(tensor, 'shape', None)!r}, "
        f"numel={numel!r}, "
        f"bytes={bytes_!r}, "
        f"MiB={_bytes_to_mib(bytes_)!r}, "
        f"contiguous={contiguous!r}, "
        f"device={getattr(tensor, 'device', None)!r}"
    )


def _tensor_numel(tensor: Any) -> int | None:
    numel = getattr(tensor, "numel", None)
    if callable(numel):
        try:
            value = numel()
            return int(value)
        except Exception:
            return None

    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    try:
        result = 1
        for dim in shape:
            result *= int(dim)
        return result
    except Exception:
        return None


def _tensor_nbytes(tensor: Any, numel: int | None) -> int | None:
    nbytes = getattr(tensor, "nbytes", None)
    if callable(nbytes):
        try:
            return int(nbytes())
        except Exception:
            pass
    elif nbytes is not None:
        try:
            return int(nbytes)
        except Exception:
            pass

    if numel is None:
        return None

    element_size = getattr(tensor, "element_size", None)
    if callable(element_size):
        try:
            return numel * int(element_size())
        except Exception:
            pass

    return _tensor_nbytes_from_dtype(getattr(tensor, "dtype", None), numel)


def _tensor_nbytes_from_dtype(dtype: Any, numel: int) -> int | None:
    dtype_name = str(dtype).lower()
    if dtype_name in {"none", ""}:
        return None
    for token, size in (
        ("bfloat16", 2),
        ("float16", 2),
        ("half", 2),
        ("float32", 4),
        ("float", 4),
        ("int64", 8),
        ("long", 8),
        ("float64", 8),
        ("double", 8),
        ("int32", 4),
        ("int16", 2),
        ("int8", 1),
        ("uint8", 1),
        ("bool", 1),
    ):
        if token in dtype_name:
            return numel * size
    return None


def _format_bytes(value: Any) -> str:
    try:
        bytes_ = int(value)
    except (TypeError, ValueError):
        return repr(value)
    return f"{bytes_} bytes ({_bytes_to_mib(bytes_):.3f} MiB)"


def _bytes_to_mib(value: int | None) -> float | None:
    if value is None:
        return None
    return value / (1024 * 1024)
